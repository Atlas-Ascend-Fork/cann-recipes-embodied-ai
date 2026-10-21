# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""在真实模型上测 System 2 的 generate 耗时，并可选做算子级 profile。

对应 docs/navigation/internvla-n1/infer_with_torch/README.md「性能画像」。要点：评测稳态下每次 generate 要处理
9 张图 / 7056 个图块，而输出往往只有 2~5 个 token —— 耗时由 prefill 主导。
本脚本先把 8 帧历史喂满，再在稳态下测量，避免用「刚 reset、只有 1 帧」的
工况得出错误结论。

用法：
    # A/B：关 / 开视觉塔融合注意力
    ASCEND_RT_VISIBLE_DEVICES=0 python bench/profile_generate.py
    ASCEND_RT_VISIBLE_DEVICES=0 python bench/profile_generate.py --fused-vision-attn

    # 顺带看耗时随历史帧数的增长
    ASCEND_RT_VISIBLE_DEVICES=0 python bench/profile_generate.py --scaling

    # 算子级 profile，产物用 --parse 解析
    ASCEND_RT_VISIBLE_DEVICES=0 python bench/profile_generate.py --fused-vision-attn --profile-out /tmp/prof
    python bench/profile_generate.py --parse /tmp/prof
"""
import argparse
import glob
import logging
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

logger = logging.getLogger(__name__)


class MissingInputError(FileNotFoundError):
    """profile 产物或样例帧缺失；main() 把它转成一行报错退出，不打栈。"""


def parse_kernels(prof_dir, top=20):
    """聚合 torch_npu profiler 产出的 kernel_details.csv。"""
    import collections
    import csv
    paths = glob.glob(os.path.join(prof_dir, "**", "kernel_details.csv"), recursive=True)
    if not paths:
        raise MissingInputError(f"{prof_dir} 下找不到 kernel_details.csv")
    with open(paths[0]) as f:
        rows = list(csv.DictReader(f))
    agg = collections.defaultdict(lambda: [0.0, 0])
    for r in rows:
        try:
            d = float(r["Duration(us)"])
        except (KeyError, ValueError):
            continue
        key = (r.get("Name") or "?").split("_")[0]
        agg[key][0] += d
        agg[key][1] += 1
    total = sum(v[0] for v in agg.values())
    logger.info("%s\nNPU kernel 总耗时 %.1f ms，%d 条记录\n", paths[0], total / 1000, len(rows))
    logger.info("%-34s %7s %11s %7s", "kernel 前缀", "次数", "总us", "占比")
    for k, (d, n) in sorted(agg.items(), key=lambda x: -x[1][0])[:top]:
        logger.info("%-34s %7d %11.0f %6.1f%%", k[:34], n, d, d / total * 100)
    logger.info("\n共 %d 个前缀。", len(agg))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parse", metavar="DIR", help="只解析已有的 profile 产物并退出")
    ap.add_argument("--internnav-root", default=os.environ.get("INTERNNAV_ROOT"))
    ap.add_argument("--model-path", default=None,
                    help="默认 $MODEL_STORE/InternVLA-N1-DualVLN")
    ap.add_argument("--fused-vision-attn", action="store_true",
                    help="启用视觉塔融合注意力（默认关闭，便于 A/B）")
    ap.add_argument("--scaling", action="store_true", help="打印耗时随历史帧数的增长")
    ap.add_argument("--warmup-steps", type=int, default=10, help="喂满历史所需的步数")
    ap.add_argument("--measure-steps", type=int, default=10)
    ap.add_argument("--profile-out", default=None)
    return ap.parse_args()


def patch_model(fused_vision_attn):
    """与 npu_eval.py 相同的两处适配：from_pretrained 换 sdpa，可选启用视觉塔融合注意力。

    返回本次配置的描述。
    """
    from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM

    _orig = InternVLAN1ForCausalLM.from_pretrained.__func__

    @classmethod
    def _fp(cls, *a, **kw):
        kw["attn_implementation"] = "sdpa"
        return _orig(cls, *a, **kw)

    InternVLAN1ForCausalLM.from_pretrained = _fp

    if not fused_vision_attn:
        return "基线（上游 sdpa + 巨型 mask）"
    from compat import npu_vision_attn
    npu_vision_attn.enable()
    return "融合视觉注意力"


def install_generate_timer(stats):
    """给 generate 包一层计时，每次调用向 stats 追加 (输入 token, 图块, 输出 token, 秒)。"""
    import torch
    from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM

    _gen = InternVLAN1ForCausalLM.generate

    def gen(self, *a, **kw):
        # 不能写成 `kw.get("input_ids") or (a[0] if a else None)`：
        # input_ids 是个多元素 Tensor，`or` 会对它取 __bool__，直接抛
        # RuntimeError: Boolean value of Tensor with more than one value is ambiguous。
        ids = kw.get("input_ids")
        if ids is None and a:
            ids = a[0]
        n_in = int(ids.shape[-1]) if ids is not None else -1
        pv = kw.get("pixel_values")
        n_pix = int(pv.shape[0]) if pv is not None else 0
        torch.npu.synchronize()
        t0 = time.perf_counter()
        out = _gen(self, *a, **kw)
        torch.npu.synchronize()
        seq = out if hasattr(out, "shape") else getattr(out, "sequences", None)
        n_out = (int(seq.shape[-1]) - n_in) if (seq is not None and n_in > 0) else -1
        stats.append((n_in, n_pix, n_out, time.perf_counter() - t0))
        return out

    InternVLAN1ForCausalLM.generate = gen


def make_agent(model):
    """返回 (agent, 相机内参)。"""
    import numpy as np
    from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

    class Args:
        device = "npu:0"
        model_path = model
        resize_w = resize_h = 384
        num_history = 8
        plan_step_gap = 4
        camera_intrinsic = np.array([[386.5, 0, 328.9, 0], [0, 386.5, 244, 0],
                                     [0, 0, 1, 0], [0, 0, 0, 1]])

    return InternVLAN1AsyncAgent(Args()), Args.camera_intrinsic


def load_frames(repo):
    """读 InternNav 自带的真机样例：返回 (指令, 最多 40 帧 RGB)。"""
    import numpy as np
    from PIL import Image

    scene = os.path.join(repo, "assets/realworld_sample_data1")
    with open(os.path.join(scene, "instruction.txt")) as f:
        instr = f.read().strip()
    paths = sorted(glob.glob(os.path.join(scene, "debug_raw_*.jpg")))
    if not paths:
        raise MissingInputError(f"{scene} 下没有样例帧，请先解压 assets/realworld_sample_data.tar.gz")
    return instr, [np.asarray(Image.open(p).convert("RGB")) for p in paths[:40]]


def log_scaling(stats):
    logger.info("\n%4s %7s %7s %8s %12s", "step", "in_tok", "图块", "out_tok", "generate(s)")
    for j, (n_in, n_pix, n_out, dt) in enumerate(stats):
        logger.info("%4d %7d %7d %8d %12.3f", j, n_in, n_pix, n_out, dt)


def log_steady(stats):
    ts = sorted(d for *_, d in stats)
    n_in, n_pix, n_out, _ = stats[-1]
    logger.info("\n稳态：输入 %d token / %d 图块 / 输出 %d token", n_in, n_pix, n_out)
    logger.info("generate  n=%d  中位 %.3fs  均值 %.3fs  min %.3fs  max %.3fs",
                len(ts), ts[len(ts) // 2], sum(ts) / len(ts), ts[0], ts[-1])


def profile_once(step, out_dir):
    from torch_npu.profiler import ProfilerActivity, profile, tensorboard_trace_handler
    os.makedirs(out_dir, exist_ok=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                 on_trace_ready=tensorboard_trace_handler(out_dir)):
        step(0)
    logger.info("\nprofile 产物 -> %s", out_dir)
    logger.info("解析：python %s --parse %s", os.path.relpath(__file__), out_dir)


def run(args):
    repo = os.path.abspath(args.internnav_root)
    model = args.model_path or os.path.join(
        os.environ.get("MODEL_STORE", os.path.expanduser("~/models")), "InternVLA-N1-DualVLN")

    sys.path.insert(0, repo)
    os.chdir(repo)

    import numpy as np
    import torch
    import torch_npu  # noqa: F401
    import torch_npu.contrib.transfer_to_npu  # noqa: F401

    logger.info("配置：%s", patch_model(args.fused_vision_attn))
    stats = []
    install_generate_timer(stats)

    agent, intrinsic = make_agent(model)
    instr, frames = load_frames(repo)
    depth = 10 * np.ones(frames[0].shape[:2], np.float32)
    pose = np.eye(4)

    def step(i):
        with torch.no_grad():
            agent.step(frames[i % len(frames)], depth, pose, instr, intrinsic=intrinsic)

    agent.reset()
    for i in range(args.warmup_steps):
        step(i)
    if args.scaling:
        log_scaling(stats)

    stats.clear()
    for i in range(args.warmup_steps, args.warmup_steps + args.measure_steps):
        step(i)
    log_steady(stats)

    if args.profile_out:
        profile_once(step, args.profile_out)


def setup_logging():
    # 只给本脚本的 logger 挂 stdout handler、不配 root：torch_npu 会把每次读环境变量
    # 都记成 root 下的 INFO 日志，root 开到 INFO 会混进基准输出
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def main():
    args = parse_args()
    if not args.parse and not args.internnav_root:
        raise SystemExit("需要 --internnav-root 或环境变量 INTERNNAV_ROOT")
    try:
        if args.parse:
            parse_kernels(args.parse)
        else:
            run(args)
    except MissingInputError as e:
        raise SystemExit(str(e)) from None


if __name__ == "__main__":
    setup_logging()
    main()
