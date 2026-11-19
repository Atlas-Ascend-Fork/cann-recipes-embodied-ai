# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""bench 脚本共用部分：数据集下载兜底、模型与输入构建、单次推理与 minADE。

数据加载与采样调用参照上游 NVlabs/alpamayo 样例（Apache-2.0）改写。

这些脚本都要在装好 torch_npu 的昇腾环境里运行，且能 import 到上游 alpamayo 源码
（安装步骤见样例 README）。
"""

import copy
import logging
import time

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)
RENAMED_SENSOR = {"metadata/sensor_presence.parquet": "metadata/feature_presence.parquet"}

DEFAULT_CLIP = "030c760c-ae38-49aa-9ad8-f5650a545d26"
DEFAULT_T0_US = 5_100_000


def patch_dataset_download():
    """hf-mirror 不支持 get_paths_info 预检，改走 hf_hub_download；新版本改名时回退原名。"""
    import physical_ai_av.utils.hf_interface as hf_interface

    original = hf_interface.HfRepoInterface.download_file

    def download_file(self, filename, **kwargs):
        try:
            return self.api.hf_hub_download(
                filename=filename,
                cache_dir=self.cache_dir,
                local_dir=self.local_dir,
                **self.repo_snapshot_info,
                **kwargs,
            )
        except Exception:
            renamed = RENAMED_SENSOR.get(filename)
            if renamed:
                try:
                    return self.api.hf_hub_download(
                        filename=renamed,
                        cache_dir=self.cache_dir,
                        local_dir=self.local_dir,
                        **self.repo_snapshot_info,
                        **kwargs,
                    )
                except Exception as error:  # 新名字也不存在时回落到原名
                    LOGGER.warning("renamed sensor file %s unavailable: %s", renamed, error)
            return original(self, filename, **kwargs)

    hf_interface.HfRepoInterface.download_file = download_file


def build_model(model_path):
    from alpamayo_r1 import helper
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.qwen_patches import apply_qwen3vl_patches

    apply_qwen3vl_patches()
    model = AlpamayoR1.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda")
    model.eval()
    return model, helper.get_processor(model.tokenizer)


def conversation(data):
    """把一组图像帧转成 processor 需要的消息列表。"""
    from alpamayo_r1 import helper

    return helper.create_message(data["image_frames"].flatten(0, 1))


def tokenize(processor, messages):
    """按样例的生成口径把消息编码成 token。"""
    options = {
        "add_generation_prompt": False,
        "continue_final_message": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    return processor.apply_chat_template(messages, tokenize=True, **options)


def prepare_inputs(model, processor, clip_id=DEFAULT_CLIP, t0_us=DEFAULT_T0_US, avdi=None):
    """读一个 clip，返回送进模型的输入与原始数据。"""
    from alpamayo_r1 import helper
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    data = load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi)
    tokens = tokenize(processor, conversation(data))
    history = {name: data[name] for name in ("ego_history_xyz", "ego_history_rot")}
    return helper.to_device(dict(history, tokenized_data=tokens), "cuda"), data


def build_batch(model_inputs):
    """每次推理前重建一份输入：token 深拷贝，历史轨迹直接复用。"""
    batch = dict(model_inputs)
    batch["tokenized_data"] = copy.deepcopy(model_inputs["tokenized_data"])
    return batch


def sample(model, batch, graph: bool, prefetch: bool, num_traj: int):
    """在 bf16 autocast 下做一次采样。"""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model.sample_trajectories_from_data_with_vlm_rollout(
            data=batch,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=num_traj,
            max_generation_length=256,
            return_extra=True,
            enable_graph=graph,
            enable_prefetch=prefetch,
        )


def run_inference(model, model_inputs, graph=False, prefetch=False, num_traj=10, seed=42):
    """一次完整推理，返回 (pred_xyz, pred_rot, extra, 耗时秒)。"""
    torch.npu.manual_seed_all(seed)
    torch.npu.synchronize()
    started = time.time()
    pred_xyz, pred_rot, extra = sample(model, build_batch(model_inputs), graph, prefetch, num_traj)
    torch.npu.synchronize()
    return pred_xyz, pred_rot, extra, time.time() - started


def ground_truth_xy(data):
    return data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()


def min_ade(pred_xyz, gt_xy):
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    distances = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    return float(distances.min())


def serialized_trajectory(pred_xyz):
    return np.round(pred_xyz.cpu().numpy(), 6).tolist()


def coc_text(extra):
    return {key: np.asarray(value).tolist() for key, value in extra.items()}
