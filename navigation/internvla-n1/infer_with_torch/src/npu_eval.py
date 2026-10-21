# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""InternVLA-N1 在昇腾 NPU 上跑 VLN-CE（Habitat）评测的启动器。

InternNav 上游的评测入口 scripts/eval/eval.py 里有若干处是按 NVIDIA GPU 写死的
（cuda 设备、flash_attention_2、habitat 的 gpu_device_id）。本启动器用 monkeypatch
在运行期替换掉这些点，**不修改 InternNav 源码**，好处是上游仓库工作区保持干净、
后续跟进上游更新时不必反复解冲突。

用法见同目录 scripts/run_eval.sh。
"""
import argparse
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import episode_select  # noqa: E402  isort: skip

# 垫片的导入顺序有讲究，不要重排：
#   eglinit          必须早于任何 habitat_sim 的 GL 上下文创建
#   hvenv_tail       把 habitat venv 挂到 sys.path 末尾，后面才 import 得到 habitat
#   py311_dataclass  必须早于 import habitat
from compat import eglinit  # noqa: F401,E402  isort: skip
from compat import hvenv_tail  # noqa: F401,E402  isort: skip
from compat import py311_dataclass_shim  # noqa: F401,E402  isort: skip

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="InternVLA-N1 VLN-CE 昇腾 NPU 评测")
    p.add_argument("--internnav-root", default=os.environ.get("INTERNNAV_ROOT"),
                   help="InternNav 仓库根目录，默认取环境变量 INTERNNAV_ROOT")
    p.add_argument("--config", default="scripts/eval/configs/habitat_dual_system_cfg.py",
                   help="InternNav 的评测配置（相对 InternNav 根目录）")
    p.add_argument("--habitat-config", default=None,
                   help="覆盖 env_settings.config_path，例如 scripts/eval/configs/vln_rxr.yaml")
    p.add_argument("--data-path", default=None,
                   help="覆盖 habitat dataset.data_path，RxR 走转换后的数据时需要")
    p.add_argument("--output-path", default=None, help="结果输出目录")
    p.add_argument("--max-steps", type=int, default=None, help="覆盖单 episode 最大步数")
    p.add_argument("--sample", type=int, default=0,
                   help=">0 时在全量上等间隔取 N 个 episode，按场景占比均匀覆盖")
    p.add_argument("--limit", type=int, default=0,
                   help=">0 时只跑前 N 个 episode。注意会集中在头几个场景，仅用于冒烟")
    p.add_argument("--shard", default=None, metavar="i/n",
                   help="手工分片，配合多卡多进程；结果用 tools/merge_results.py 合并")
    p.add_argument("--fused-vision-attn", dest="fused_vision_attn",
                   action="store_true", default=True,
                   help="用 npu_fusion_attention 替换视觉塔注意力（默认开启）")
    p.add_argument("--no-fused-vision-attn", dest="fused_vision_attn",
                   action="store_false", help="关闭上述替换，用于 A/B 对比")
    p.add_argument("--save-video", action="store_true",
                   help="为**成功**的 episode 存轨迹视频（RGB + 俯视图 + 预测像素目标），"
                        "落在 <output-path>/vis_<epoch>/<scene>/<episode:04d>.mp4，"
                        "epoch 取自配置，默认 0")
    p.add_argument("--vis-debug", action="store_true",
                   help="为**每条** episode 存第一人称调试视频（含 step/action 叠字），"
                        "失败案例也会保留，用于排查；落在 "
                        "<vis-debug-path>/epoch_<epoch>/<scene>_<episode:04d>.mp4")
    p.add_argument("--vis-debug-path", default=None,
                   help="调试视频目录，需配合 --vis-debug；默认 <output-path>/vis_debug。"
                        "上游默认值是固定的 ./logs/habitat/vis_debug，与输出目录无关，"
                        "多分片会混写到一起")
    p.add_argument("--seed", type=int, default=0,
                   help="随机种子。上游 habitat 评测路径只 seed 了 numpy，"
                        "System 1 的扩散采样会导致同样输入跑两次结果不同；置为 -1 可关闭")
    return p.parse_args()


def _setup_logging():
    # 只给本样例自己的 logger 挂 handler、不配 root：上游和三方库的日志行为保持原样
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    for lg in (logger, episode_select.logger):
        lg.addHandler(handler)
        lg.setLevel(logging.INFO)
        lg.propagate = False


def _seed_host(seed):
    # ---- 适配 0：补上种子 ----
    # 上游 internnav/evaluator/distributed_base.py 只做了 np.random.seed(local_rank)。
    # 但 System 1 的轨迹头是扩散模型，generate_traj 里用 diffusers 的 randn_tensor
    # 从**全局 torch RNG** 取噪声，没有 seed 就导致同一份代码、同一批 episode
    # 跑两次结果不同（实测同一 episode steps 126 vs 105、SPL 0.790 vs 0.694）。
    # 这里补上 torch / npu 的种子，让结果可复现，也让优化前后的对比有意义。
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _seed_device(seed):
    import torch

    # transfer_to_npu 之后 torch.cuda 已指向 npu，这一句才能真正落到 NPU 的 RNG 上
    torch.cuda.manual_seed_all(seed)


def _patch_attention(fused_vision_attn):
    # ---- 适配 2：flash_attention_2 -> sdpa ----
    # 昇腾上没有 flash_attn 实现。这里换成 sdpa 保证能跑通；
    # sdpa 未走融合算子，性能优化见 docs/navigation/internvla-n1/infer_with_torch/README.md。
    from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM

    _orig_from_pretrained = InternVLAN1ForCausalLM.from_pretrained.__func__

    @classmethod
    def _from_pretrained_npu(cls, *a, **kw):
        kw["attn_implementation"] = "sdpa"
        return _orig_from_pretrained(cls, *a, **kw)

    InternVLAN1ForCausalLM.from_pretrained = _from_pretrained_npu

    # ---- 适配 2b：视觉塔注意力换融合算子 ----
    # 上游 Qwen2_5_VLVisionSdpaAttention 会物化一个 seq_length^2 的块对角 mask，
    # 8 帧历史下 seq_length=7056，每层 50MB、32 层各建一次，且 SDPA 收到显式 mask
    # 后无法走融合核。换成 npu_fusion_attention 的 TND 布局即可完全省掉 mask。
    # 实测单次 generate 2.60s -> 0.35s。详见 compat/npu_vision_attn.py
    if fused_vision_attn:
        from compat import npu_vision_attn

        npu_vision_attn.enable()


def _patch_habitat_config(data_path):
    # ---- 适配 3：habitat_sim 不去找 CUDA 设备 ----
    # vln_r2r.yaml / vln_rxr.yaml 里 gpu_device_id: 0，在无 CUDA 的机器上会报
    # "unable to find CUDA device 0 among N EGL devices"。置 -1 走默认 EGL 设备。
    import internnav.habitat_extensions.vln.habitat_vln_evaluator as hve
    from habitat.config import read_write

    _orig_get_habitat_config = hve.get_habitat_config

    def _get_habitat_config(path, *a, **kw):
        cfg = _orig_get_habitat_config(path, *a, **kw)
        with read_write(cfg):
            cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = -1
            if data_path:
                cfg.habitat.dataset.data_path = data_path
        return cfg

    hve.get_habitat_config = _get_habitat_config


def _patch_episode_selection(sample, limit, shard):
    # ---- 适配 4：采样与手工分片 ----
    # 不走 evaluator 自带的 torch.distributed all_gather：每个进程都是 world_size=1，
    # 各写各的 progress.json，跑完用 tools/merge_results.py 离线合并。
    # 指标是逐 episode 的算术平均，分片再合并与整体计算完全等价。
    from internnav.env.habitat_env import HabitatEnv

    _orig_generate = HabitatEnv.generate_episodes

    def _generate_episodes(self):
        # 采样 / 切片 / 过滤已完成的具体逻辑在 episode_select.py，那边有单测。
        # 这里只负责让上游按**原始顺序**返回完整列表：把 output_path 临时指到一个
        # 不存在的目录，使上游自带的断点续跑过滤空转，切片之后再自己过滤。
        real_output_path = self.output_path
        self.output_path = os.path.join(real_output_path, "__unfiltered__")
        try:
            eps = _orig_generate(self)
        finally:
            self.output_path = real_output_path

        return episode_select.select_episodes(
            eps, sample=sample, limit=limit, shard=shard,
            done=episode_select.read_done(real_output_path))

    HabitatEnv.generate_episodes = _generate_episodes


def _load_cfg(args):
    from scripts.eval.eval import load_eval_cfg

    cfg = load_eval_cfg(args.config)
    if args.habitat_config:
        cfg.env.env_settings["config_path"] = args.habitat_config
    if args.output_path:
        cfg.eval_settings["output_path"] = args.output_path
    if args.max_steps:
        cfg.eval_settings["max_steps_per_episode"] = args.max_steps

    # 可视化。top_down_map 这个 measure 是 evaluator 在 __init__ 里无条件加的，
    # 所以只要把开关打开即可；注意上游只为**成功**的 episode 写 vis 视频，
    # 失败案例要看 --vis-debug 那一路。
    if args.save_video:
        cfg.eval_settings["save_video"] = True
    if args.vis_debug:
        cfg.agent.model_settings["vis_debug"] = True
        cfg.agent.model_settings["vis_debug_path"] = (
            args.vis_debug_path
            or os.path.join(cfg.eval_settings["output_path"], "vis_debug"))
    return cfg


def main():
    args = parse_args()
    if not args.internnav_root:
        raise SystemExit("需要 --internnav-root 或环境变量 INTERNNAV_ROOT")
    _setup_logging()
    repo = os.path.abspath(args.internnav_root)
    sys.path.insert(0, repo)
    # InternNav 里多处用相对路径读权重（internvla_n1_arch.py 的 MODEL_PATH_TO="checkpoints"）
    # 和数据（habitat 配置里的 data/...），必须在仓库根目录下运行
    os.chdir(repo)

    if args.seed >= 0:
        _seed_host(args.seed)

    # ---- 适配 1：cuda -> npu ----
    # evaluator 里写死了 torch.device(f"cuda:{local_rank}")、torch.cuda.set_device()、
    # 以及 nccl 后端。transfer_to_npu 会把这些统一转到 npu / hccl 上。
    # 必须在任何 torch.device("cuda") 之前 import。
    import torch_npu.contrib.transfer_to_npu  # noqa: F401

    if args.seed >= 0:
        _seed_device(args.seed)

    _patch_attention(args.fused_vision_attn)
    _patch_habitat_config(args.data_path)
    if args.sample > 0 or args.limit > 0 or args.shard:
        _patch_episode_selection(args.sample, args.limit, args.shard)

    # ---- 跑评测 ----
    sys.argv = [sys.argv[0]]  # InternNav 的 eval.py 自己会 parse_args
    from internnav.evaluator import Evaluator

    cfg = _load_cfg(args)
    logger.info("[cfg] habitat=%s out=%s max_steps=%s save_video=%s vis_debug=%s",
                cfg.env.env_settings["config_path"],
                cfg.eval_settings["output_path"],
                cfg.eval_settings["max_steps_per_episode"],
                cfg.eval_settings.get("save_video", False),
                cfg.agent.model_settings.get("vis_debug", False))

    Evaluator.init(cfg).eval()


if __name__ == "__main__":
    main()
