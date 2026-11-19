# LingBot-VA 在线推理（Ascend NPU）

本样例提供 LingBot-VA 的 NPU 推理服务和 LIBERO 客户端。源码补丁位于 `patches/`，推理优化位于 `src/`。客户端通过双相机观测和语言指令请求动作，并将执行后的观测回传给服务端更新 KV Cache。

## 环境与版本

测试环境为 Atlas A3（逻辑设备 64 GiB，aarch64）、CANN 9.0.0、驱动 26.1.1、Python 3.11.15、PyTorch 2.10.0+cpu、torch_npu 2.10.0。驱动报告设备为 Ascend910，型号标识为 9382；模型计算使用 torch_npu 后端。

| 项目 | 版本 |
|---|---|
| LingBot-VA | `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb` |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |

源码和补丁的校验值见 [`source_lock.json`](configs/source_lock.json)，权重版本和文件哈希见 [`checkpoints.json`](configs/checkpoints.json)。

四个 suite 的成功率：Spatial 95.0%、Object 99.0%、Goal 96.6%、Long 98.0%，平均 97.15%。耗时数据与优化项见[优化文档](../../../docs/manipulation/lingbot-va_optimization.md)。

## 安装

服务端和 LIBERO 客户端分别安装在独立环境中，依赖版本见 `requirements.lock`。无显示设备时，客户端使用 OSMesa 渲染。以下命令在本样例目录执行。

```bash
git clone https://github.com/Robbyant/lingbot-va.git /path/to/lingbot-va
git -C /path/to/lingbot-va checkout 7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /path/to/LIBERO
git -C /path/to/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
python3 -m venv --system-site-packages /path/to/model-env
/path/to/model-env/bin/python -m pip install -r requirements.lock
/path/to/model-env/bin/python prepare.py source /path/to/lingbot-va
/path/to/model-env/bin/python prepare.py checkpoints --models-root /path/to/models --download
```

`prepare.py source` 校验源码版本并应用补丁。`prepare.py checkpoints` 校验权重文件，写入样例配置和元数据，并创建共享组件的软链接；加 `--download` 时先下载权重。准备操作通过文件锁串行执行。

## 启动服务

按 suite 选权重目录，Long 不设 `NORM_STATS`。

| suite | /path/to/models 下的目录 | NORM_STATS |
|---|---|---|
| libero_spatial | lingbot-va-libero-spatial | 该目录/norm_stats.json |
| libero_object | lingbot-va-libero-object | 该目录/norm_stats.json |
| libero_goal | lingbot-va-libero-goal | 该目录/norm_stats.json |
| libero_10（Long） | lingbot-va-posttrain-libero-long | 不设置，沿用锁定配置 |

```bash
export LINGBOT_VA_ROOT=/path/to/lingbot-va
export MODEL_ROOT=/path/to/models/lingbot-va-libero-spatial
export NORM_STATS="$MODEL_ROOT/norm_stats.json"
export PYTHON=/path/to/model-env/bin/python
export ASCEND_RT_VISIBLE_DEVICES=0 PORT=29976 MASTER_PORT=29881
bash run_server.sh
```

三个社区 suite 的配置把 `attn_mode` 从 `flex` 改成 `torch`，Long 沿用原配置；四个 suite 都关闭了融合 RMSNorm。

## 健康检查与单回合

在客户端环境配置 LIBERO 的 assets、bddl_files 和 init_states 路径。

```bash
export PYTHONPATH="/path/to/lingbot-va:/path/to/lingbot-va/wan_va:/path/to/LIBERO:$PYTHONPATH"
export LINGBOT_EVAL_HOST=127.0.0.1 LINGBOT_EVAL_PORT=29976
/path/to/client-env/bin/python src/evaluate_libero_npu.py --health-check
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
/path/to/client-env/bin/python src/evaluate_libero_npu.py \
  --suite libero_spatial --task 0 --episode 0 --seed 100 \
  --out outputs/smoke/libero_spatial/task00/episode00
```

`--health-check` 读取服务元数据并发送 reset 请求。单回合结果写入 `--out` 指定的目录，包括 `record.json`、`requests.jsonl` 和 `action_*.npy`。

## 复现四个 suite

每个 suite 使用对应权重启动服务，依次评测。以下以 Spatial 为例：

```bash
suite=libero_spatial
run_root=outputs/libero-run-001
for task in $(seq 0 9); do
  for episode in $(seq 0 49); do
    seed=$((100 + task * 50 + episode))
    printf -v case_dir '%s/%s/task%02d/episode%02d' "$run_root" "$suite" "$task" "$episode"
    /path/to/client-env/bin/python src/evaluate_libero_npu.py \
      --suite "$suite" --task "$task" --episode "$episode" \
      --seed "$seed" --out "$case_dir" || exit 1
  done
done
```

`libero_object`、`libero_goal` 和 `libero_10` 换权重和归一化配置后重复这个过程。每个 suite 是 10 个 task、每个 task 50 个回合，共 500 回合；成功率按 LIBERO 的 `done` 统计，最终结果取四个 suite 的平均值。

## 注意事项

- 补丁或哈希校验失败时，先确认上游提交与 LIBERO 版本是否与 `configs/source_lock.json` 一致。
- suite 结果异常时，先查权重目录、`norm_stats` 和相机配置。
- 每轮使用新的输出目录，避免覆盖已有结果。
