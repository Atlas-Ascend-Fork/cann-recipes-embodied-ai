# Being-H0.5 MuJoCo 仿真与视频评估

> 外部资源与 Docker：[安装和发布说明](../resources/README.md)。镜像与资源包已发布至社区 OBS，支持 HTTPS 下载和离线安装。

[简体中文](README.md) | [English](README_en.md)

本目录提供纯 MuJoCo 的闭环策略验证：Being-H0.5 根据双相机图像、语言指令和
状态预测 wrist + `z_gesture`，NativeVAE 将 `z_gesture` 解码为灵巧手原生关节，
MuJoCo 执行动力学并输出成功率、轨迹、耗时、元数据和 episode 视频。不依赖
Isaac Gym，也不包含第二套 Being-H0.5 或 NativeVAE 实现。

本模块只维护 MuJoCo 环境、场景、任务、回放和评估入口。策略推理编排复用
`infer_with_torch/code/vae/integration/`，NativeVAE 实现复用 `vae/code/vae/`。

相关模块文档：

- [公共代码与组装层](../common/README.md)
- [NPU 在线推理](../infer_with_torch/README.md)
- [VAE 训练与后训练](../vae/README.md)

## 代码结构

```text
being-h0.5/
├── infer_with_torch/code/vae/integration/
│   ├── beingh_shadow_grasp_eval.py       # 闭环策略编排的 canonical 实现
│   ├── benchmark_beingh_retargeting.py   # VAE/几何迁移推理对比
│   └── diagnose_cross_hand_cameras.py    # 跨手型相机诊断
└── sim/
    ├── requirements.txt
    └── code/vae/
        ├── sim/                          # 环境、场景、任务和评估客户端
        │   └── assets/
        │       ├── environment/          # 桌面、相机、灯光和物理参数
        │       └── demos/                # 可独立运行的最小场景
        ├── examples/
        │   └── beingh_shadow_grasp_eval.py # 保持原 CLI 的薄入口
        ├── scripts/                      # smoke、查看、回放和 demo 入口
        ├── evaluation/                   # 便携 episode 清单
        ├── tests/test_sim.py
        └── sim/                          # Python API 与坐标约定实现
```


## 1. 组装运行目录

从仓库根目录执行。组装脚本不会覆盖非空目标目录：

```bash
export WORK_DIR=/path/to/workdir
export BEINGH_DIR="$WORK_DIR/beingh05_vae"

bash manipulation/being-h0.5/vae/scripts/setup.sh
```

脚本依次合并 `common`、NPU 推理、VAE 和 MuJoCo 层。后续命令都在组装目录
中执行：

```bash
cd "$BEINGH_DIR"
python -m pip install -r vae/requirements.txt
python -m pip install -r /path/to/repo/manipulation/being-h0.5/sim/requirements.txt
python -m pip install -e vae
```

已有可用环境时不必重复安装。确认三个关键入口可以加载：

```bash
python vae/examples/beingh_shadow_grasp_eval.py --help
python vae/scripts/benchmark_beingh_retargeting.py --help
python vae/scripts/diagnose_cross_hand_cameras.py --help
```

## 2. 准备模型、数据与 NPU

至少需要以下三个路径：

```bash
export MODEL_PATH=/path/to/beingh_posttrained_checkpoint
export VAE_CHECKPOINT=/path/to/native_vae_inference.pt
export CONTROL_DATASET=/path/to/matching_shadow_lerobot_dataset
```

- `MODEL_PATH`：包含 `config.json`、tokenizer、模型 safetensors 和后训练元数据。
- `VAE_CHECKPOINT`：NativeVAE 推理 checkpoint。
- `CONTROL_DATASET`：与 checkpoint 匹配的 LeRobot 数据集，用于初始状态和训练
  分布运动限制；它不是在评估时回放的专家动作。

在 Ascend NPU 上先加载 CANN 环境。若只暴露物理 NPU 1，程序内设备仍写
`npu:0`：

```bash
source /usr/local/Ascend/driver/bin/setenv.bash
source /path/to/Ascend/cann/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=1
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
```

CUDA 可将后续命令的 `--device npu:0` 改为 `cuda:0`。无 GPU/NPU 的环境只能
运行 MuJoCo smoke 或使用可在 CPU 加载的轻量策略。

## 3. 推理一个 episode 并生成视频

仓库自带最小 TissueBox 场景、物体 mesh 和 episode manifest。下面命令执行
一次完整闭环推理，并默认记录主视角与腕部视角：

```bash
export RESULT_DIR="$BEINGH_DIR/results/tissuebox_demo"

bash vae/scripts/run_tissuebox_policy_demo.sh \
  --model-path "$MODEL_PATH" \
  --vae-checkpoint "$VAE_CHECKPOINT" \
  --control-reference-dataset "$CONTROL_DATASET" \
  --device npu:0 \
  --output-dir "$RESULT_DIR"
```

这不是专家轨迹回放。episode 的第一个参考状态仅用于初始化；从第 1 个控制步
开始，动作来自 Being-H0.5，手指关节由 NativeVAE 解码。TissueBox 默认成功
判据是抬升至少 0.08 m。

输出目录会按 checkpoint、手型和推理配置继续分层。查找结果：

```bash
find "$RESULT_DIR" -type f \( \
  -name 'rollout.mp4' -o \
  -name 'rollout_wrist.mp4' -o \
  -name 'rollout.npz' -o \
  -name 'metadata.json' \
\) -print
```

每个 episode 的产物为：

```text
episode_000000/
├── rollout.mp4          # ego_opposite 主视角 H.264 视频
├── rollout_wrist.mp4    # wrist 腕部视角 H.264 视频
├── rollout.npz          # 状态、动作、物体位姿、模型查询及耗时
└── metadata.json        # 成功标记、最大抬升、配置、视频路径和耗时摘要
```

指定单 episode 的主视角文件名时，可直接调用评估器：

```bash
python vae/examples/beingh_shadow_grasp_eval.py \
  --deployment-profile safe_smooth \
  --model-path "$MODEL_PATH" \
  --vae-checkpoint "$VAE_CHECKPOINT" \
  --control-reference-dataset "$CONTROL_DATASET" \
  --dataset vae/evaluation/object_episodes/tissuebox_idx83_shadow.jsonl \
  --scene-xml vae/sim/assets/demos/tissuebox_idx83_shadow/scene.xml \
  --scene-contains-object \
  --episode 0 \
  --hand shadow_hand_right \
  --instruction 'Lift the tissue box from the table.' \
  --success-profile dex-demo \
  --latent-observation-mode commanded \
  --device npu:0 \
  --output "$RESULT_DIR/episode0/rollout.npz" \
  --output-video "$RESULT_DIR/episode0/rollout.mp4" \
  --output-metadata "$RESULT_DIR/episode0/metadata.json"
```

双相机策略会自动在指定的 `rollout.mp4` 旁生成 `rollout_wrist.mp4`。

## 4. 批量运行 10 个 episode 并统计成功率

数据集 manifest 中引用的普通物体需要完整资产包：

```text
vae/sim/assets/templates/<hand>.xml
vae/sim/assets/objects/<object_id>/visual/simplified.obj
vae/sim/assets/objects/<object_id>/collision/convex_piece_*.obj
```

准备好 manifest、资产和匹配的控制参考数据集后运行：

```bash
export EVAL_MANIFEST=/path/to/object_episodes.jsonl
export RESULT_DIR="$BEINGH_DIR/results/bottle_10ep"

python vae/examples/beingh_shadow_grasp_eval.py \
  --deployment-profile safe_smooth \
  --model-path "$MODEL_PATH" \
  --vae-checkpoint "$VAE_CHECKPOINT" \
  --dataset "$EVAL_MANIFEST" \
  --episode-range 0 9 \
  --control-reference-dataset "$CONTROL_DATASET" \
  --hand shadow_hand_right \
  --device npu:0 \
  --latent-observation-mode commanded \
  --success-profile loose \
  --output-dir "$RESULT_DIR" \
  --summary-output "$RESULT_DIR/summary.json"
```

批量模式只加载一次 Being-H0.5 和 NativeVAE。`summary.json` 包含：

- `num_episodes`、`num_successes`、`success_rate`
- `successful_episode_indices`、`failed_episode_indices`
- `mean_max_lift_m`、`max_lift_m`
- `total_steps`、`total_queries`
- 每个 episode 的 `success`、`timing`、`videos` 和安全事件

快速打印成功率和成功 episode：

```bash
python - "$RESULT_DIR/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"success: {s['num_successes']}/{s['num_episodes']} "
      f"({100 * s['success_rate']:.1f}%)")
print("successful episodes:", s["successful_episode_indices"])
PY
```

查看全部 episode 视频：

```bash
find "$RESULT_DIR" -type f \( \
  -name 'rollout.mp4' -o -name 'rollout_wrist.mp4' \
\) -print | sort
```

如只测指定编号，使用 `--episodes 0 3 7`；测 manifest 中全部任务，使用
`--all-episodes`。只有明确不需要视频、希望降低内存占用时才添加
`--no-record-images`。

## 5. 验证仿真环境

在加载大模型前，先运行快速回归：

```bash
# 三种手型的状态、动作、物理步进和 RGB 渲染
python vae/scripts/sim_smoke_test.py --steps 10

# 不渲染图像
python vae/scripts/sim_smoke_test.py --no_render --steps 10

# 仿真单元测试
python vae/tests/test_sim.py
```

查看环境或数据集 episode：

```bash
python vae/scripts/sim_view_environment.py --hand shadow_hand_right
python vae/scripts/sim_view_dataset_episode.py \
  --dataset "$EVAL_MANIFEST" --episode 0 --hand shadow_hand_right
```

## 6. 仿真资产、场景清单与来源

`code/vae/evaluation/object_episodes/` 保存 MuJoCo 评估使用的便携对象配置。
JSONL 记录 evaluation/source/original episode index、对象 ID/类别、任务文本、mesh
scale、世界坐标位置、`wxyz` MuJoCo quaternion 和 episode frame count，不包含图像、
策略 checkpoint 或专家动作轨迹。

主要清单包括：

| Manifest | 内容 |
| --- | --- |
| `shadow_grasp_0725.jsonl` | 1,207 个评估对象场景 |
| `shadow_grasp_0725_core_bottle_1071.jsonl` | 同一 bottle 的 52 个场景 |
| `tissuebox_idx83_shadow.jsonl` | 仓库自带 TissueBox 单 episode demo |

`object_episodes/tissuebox_idx83_shadow.jsonl` 是仓库自带 TissueBox demo 的单
episode 清单。它只保留源 episode 的第一个 native target 作为 reset pose；闭环
执行从第一个控制步开始全部使用 Being-H0.5 输出，latent checkpoint 的手指动作
由 NativeVAE 解码。

该 JSONL 必须逐行解析为 JSON 对象，不能插入注释；数据内保留
`source_repository` 和 `source_commit` 来源字段。本次内部资产的已授权确认
见[授权记录](../resources/INTERNAL_ASSET_AUTHORIZATION.md)，第三方资产的许可
说明见[第三方许可说明](../resources/THIRD_PARTY_NOTICES.md)。

核心配置仍随源码提供，其余资源通过运行资源包安装到组装目录：

- `code/vae/sim/assets/environment/base_scene.xml`：相机、桌面、灯光和物理参数；
- `vae/sim/assets/demos/tissuebox_idx83_shadow/scene.xml`（外部资源包）：Shadow Hand 与
  TissueBox 独立场景；
- demo 使用的 38 个 OBJ 文件，总计约 2.03 MB；
- `beingh05-runtime-v2.tar.gz`：三手模板、手部 mesh 和 22 组对象资产，
  由资源安装脚本校验后恢复到组装目录；不再在 Git 中保存嵌套压缩包。

TissueBox 场景来自 `sicloud2333/dex_project` 提交
`0d2f89042627eb6c4f4207e1d687e5e708591e95` 中的
`ohra_shadow_grasp_0_table_scene.xml`。没有复制专家轨迹、非 MuJoCo runner、URDF、
未使用 mesh 或生成结果。成功判据沿用源 benchmark：物体抬升至少 0.08 m。

该固定版本源仓库没有根许可证文件；2026-10-31 项目方已确认所提供内部资产的社区再分发授权，见[授权记录](../resources/INTERNAL_ASSET_AUTHORIZATION.md)。手模型的原始第三方许可证按[逐文件核查](../resources/THIRD_PARTY_NOTICES.md)保留；Being-H 和 MuJoCo 的来源说明见
[common 模块 README](../common/README.md#source-and-reproducibility)。

## 7. 动作、状态、坐标与 Python API

三种 MuJoCo 环境统一采用：

```text
environment action/state = [world wrist xyz, absolute wrist Euler XYZ, native joint q]
```

| Hand | Wrist | Native joints | 总维度 |
| --- | ---: | ---: | ---: |
| Shadow | 6 | 22 | 28 |
| Gaia | 6 | 15 | 21 |
| Sharpa | 6 | 22 | 28 |

数据配置可以声明 `WRIST_WORLD_ORIGIN`。当前双相机 Shadow 数据使用
`[0, 0, 0.4]` m：

```text
policy_xyz = world_xyz - wrist_world_origin
world_xyz  = policy_xyz + wrist_world_origin
```

Euler RPY 是绝对旋转，不做平移。`step()` 返回同顺序实际 state、主 RGB
`image`、按相机名索引的 `images`、物体 pose/velocity 和仿真时间。默认控制频率
30 Hz，MuJoCo 使用 2 ms substep。

### 环境接口

```python
from sim import GraspEnv, GraspEnvConfig

with GraspEnv(
    GraspEnvConfig(
        hand="shadow_hand_right",
        observation_cameras=("ego_opposite", "wrist"),
    )
) as env:
    obs, info = env.reset()
    action = policy.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
```

如果实验首个 action 定义了特定腕部初始位姿，使用
`reset(initial_action=...)`。交互 GUI 可通过 `env.launch_gui()`、
`env.sync_gui()` 和 `env.gui_is_running` 管理。

### 对象场景组合

同一个 manifest 记录可以与任一支持的手型组合：

```bash
python vae/scripts/sim_view_dataset_episode.py \
  --dataset vae/evaluation/object_episodes/shadow_grasp_0725.jsonl \
  --episode 20 \
  --hand gaia_hand_right
```

也可直接读取原始 LeRobot 数据集；加载器会读取对象 ID、scale、rotmat 和 world
xy，并根据旋转后的 collision mesh 计算桌面高度：

```bash
python vae/scripts/sim_view_dataset_episode.py \
  --dataset /path/to/shadow_grasp_0725 \
  --episode 20 \
  --hand shadow_hand_right
```

### Gesture policy 适配

```text
policy input/output = [wrist 6D, z_gesture 24D]
environment action  = [wrist 6D, decoded target native q]
```

```python
from native_vae import NativeVAE
from sim import CallablePolicy, GesturePolicyAdapter, PolicyEvaluationClient

vae = NativeVAE.from_pretrained(device="npu:0")
policy = GesturePolicyAdapter(
    CallablePolicy(model_predict),
    vae=vae,
    target_hand="sharpa_hand_right",
)
result = PolicyEvaluationClient(env, policy).run(
    initial_action=initial_action,
    max_steps=300,
)
print(result.success, result.max_lift_m)
```

六维 wrist 输出必须已经位于目标 MuJoCo wrist frame。默认成功规则是连续 10 个
policy frame 抬升至少 0.20 m；TissueBox demo 单独使用源 benchmark 的 0.08 m。

回放 `[T,D]` native action 并保存视频：

```bash
python vae/scripts/sim_replay_actions.py \
  --hand shadow_hand_right \
  --actions trajectory.npz \
  --episode 0 \
  --output_video outputs/shadow_episode0.mp4
```

MP4 输出需要系统 `ffmpeg`。可通过 `--scene_xml` 指定具体 hand/object 场景；共享
base scene 仍会提供一致的桌面、相机、灯光和物理参数。

## 8. 常见问题

- `Unknown or incomplete object`：缺少对应 object 的 visual 或 collision mesh。
- 找不到手型 XML：检查 `vae/sim/assets/templates/` 是否包含目标手型模板。
- `libEGL` 警告：无头服务器优先设置 `MUJOCO_GL=egl`；若 EGL 不可用，可测试
  `MUJOCO_GL=osmesa`。
- 没有生成视频：确认未传 `--no-record-images`，并检查 `metadata.json` 的
  `videos` 字段。视频由 PyAV 写为 H.264。
- NPU 设备错误：先运行 `npu-smi info`；设置 `ASCEND_RT_VISIBLE_DEVICES` 后，
  `--device` 使用重新编号后的逻辑设备。
- checkpoint 与数据不匹配：以 checkpoint 自带的 `data_config_name` 和后训练
  元数据为准，并使用同一任务的 `CONTROL_DATASET`。

## 补充：CUDA 基线入口

以下命令在组装后的工作区及兼容 CUDA 环境中执行；昇腾主流程使用上文对应的已验证环境。

MuJoCo 与 Sharpa/Gaia zero-shot 评估：

```bash
python vae/examples/beingh_shadow_grasp_eval.py \
  --model-path outputs/<training-run>/0040000 \
  --dataset vae/evaluation/object_episodes/<manifest>.jsonl \
  --episode-range 0 99 \
  --hand sharpa_hand_right \
  --device cuda:0
```

Shadow physical-joint checkpoint 的 geometry retargeting baseline：

```bash
python vae/examples/beingh_shadow_grasp_eval.py \
  --model-path outputs/<joint-baseline-run>/0040000 \
  --dataset vae/evaluation/object_episodes/<manifest>.jsonl \
  --episode-range 0 99 \
  --hand sharpa_hand_right \
  --joint-retargeting geometry \
  --geometry-action-chunk-mode batch \
  --latent-observation-mode commanded \
  --device cuda:0

python vae/scripts/benchmark_beingh_retargeting.py --help
```
