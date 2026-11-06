# Being-H0.5 公共代码与组装层

[简体中文](README.md) | [English](README_en.md)

`common` 保存 Being-H0.5 各功能模块共同使用的新增代码、配置和最小差异补丁。
它不是可单独运行的样例，而是 NPU 推理、VAE 后训练和 MuJoCo 仿真三条路径的
共享组装输入。

其他功能模块见 [NPU 推理](../infer_with_torch/README.md)、
[VAE 训练](../vae/README.md) 和 [MuJoCo 仿真](../sim/README.md)。

## 模块定位

未修改的 Being-H0.5 官方源码位于公共 submodule
`../third_party/Being-H`，固定到提交
[`9443a4082bce013cc61e6523596940134179b23e`](https://github.com/BeingBeyond/Being-H/commit/9443a4082bce013cc61e6523596940134179b23e)。

本模块只维护以下内容：

- `upstream_files.txt`：从 `third_party/Being-H/Being-H05` 复制的精确白名单；
- `code/`：本项目新增且被多条功能路径共同使用的文件；
- `patches/`：对固定上游文件的最小功能差异；
- 共享配置模板、数据配置注册、兼容工具和轻量 smoke test。

## 目录结构

```text
common/
├── README.md
├── upstream_files.txt            # 固定上游源码复制白名单
├── code/
│   ├── BeingH/                   # Being-H 共享新增文件
│   ├── configs/                  # 路径模板和数据配置索引
│   └── scripts/                  # smoke test 和实验索引工具
└── patches/
    ├── beingvla.patch            # BeingVLA 共享差异
    ├── data_config.patch         # 数据配置差异
    ├── qwen2_npu_import.patch    # NPU 导入兼容
    ├── siglip_npu_import.patch   # NPU 导入兼容
    └── video_backend_fallback.patch # 视频后端回退
```

## 组装顺序

最终工作区必须按以下顺序生成：

1. 复制 `upstream_files.txt` 中列出的固定版本 Being-H 文件；
2. 合并 `common/code/`，应用 `common/patches/`；
3. 合并 `infer_with_torch/code/` 和 NPU 推理差异；
4. 合并 `vae/code/`、NativeVAE 和 VAE 后训练差异；
5. 合并 `sim/code/` 中的 MuJoCo 环境与评估入口。

## 使用方法

### 1. 初始化固定版本上游源码

从仓库根目录执行：

```bash
git submodule update --init manipulation/being-h0.5/third_party/Being-H
git -C manipulation/being-h0.5/third_party/Being-H rev-parse HEAD
```

第二条命令应输出 `9443a4082bce013cc61e6523596940134179b23e`；
两个组装入口也会检查该提交，不匹配时停止组装。

需要初始化仓库中全部 submodule 时使用：

```bash
git submodule update --init --recursive
```

### 2. 选择组装入口

只需要生产 NPU 推理闭包：

```bash
bash manipulation/being-h0.5/infer_with_torch/download_code_and_data.sh \
  --work-root /path/to/workdir
```

需要 VAE、后训练和 MuJoCo 的完整工作区：

```bash
export WORK_DIR=/path/to/workdir
bash manipulation/being-h0.5/vae/scripts/setup.sh
```

两个入口都会读取 `common`，但交付边界不同。推理入口不会复制训练、VAE、仿真、
profiling 或测试文件；完整组装入口才会生成这些路径。

## 验证

完整工作区组装完成后，可以先运行不依赖数据下载的源码和入口检查：

```bash
cd "${WORK_DIR}/beingh05_vae"
PYTHONPATH="$PWD" python scripts/smoke_test_beingh05.py
```

需要同时检查配置导入和本地数据时：

```bash
PYTHONPATH="$PWD" python scripts/smoke_test_beingh05.py \
  --import-config \
  --require-local-data
```

第二条命令要求数据和 checkpoint 已按对应功能指南准备，不属于纯源码检查。

## 运行环境

请分别按[推理指南](../infer_with_torch/README.md)或[训练指南](../vae/README.md)准备环境；两条路径的已验证 Python、PyTorch 版本不同。以下共享下载命令请在仓库之外的工作目录执行。

## 模型和数据下载

Being-H0.5 基础 checkpoint 来自
[官方 Hugging Face collection](https://huggingface.co/collections/BeingBeyond/being-h05)：

```bash
hf download BeingBeyond/Being-H05-2B --local-dir ckpts/Being-H05-2B
hf download OpenGVLab/InternVL3_5-2B --local-dir ckpts/InternVL3_5-2B
hf download Qwen/Qwen3-0.6B --local-dir ckpts/Qwen3-0.6B
```

Shadow 主训练数据、策略模型、NativeVAE 和 physical-joint baseline：

```bash
hf download zju/shadow_grasp_bottle22249179_aug100_2cam \
  --repo-type dataset \
  --local-dir data/shadow_grasp_bottle22249179_aug100_2cam
hf download zju/Being-H05-shadow-grasp-2cam-rot6d-zraw \
  --local-dir ckpts/Being-H05-shadow-grasp-2cam-rot6d-zraw
hf download zju/Being-H05-native-vae native_n2_epoch800_inference.pt \
  --local-dir vae/checkpoints
hf download zju/Being-H05-shadow-grasp-2cam-joints \
  --local-dir ckpts/Being-H05-shadow-grasp-2cam-joints
```

Sharpa/Gaia 跨手评估数据和 baseline：

```bash
hf download zju/sharpa_grasp_bottle22249179_geo_visual100_2cam \
  --repo-type dataset \
  --local-dir data/sharpa_grasp_bottle22249179_geo_visual100_2cam
hf download zju/gaia_grasp_bottle22249179_geo_visual100_2cam \
  --repo-type dataset \
  --local-dir data/gaia_grasp_bottle22249179_geo_visual100_2cam
hf download zju/Being-H05-sharpa-joint-baseline \
  --local-dir ckpts/Being-H05-sharpa-joint-baseline
hf download zju/Being-H05-gaia-joint-baseline \
  --local-dir ckpts/Being-H05-gaia-joint-baseline
```

模型、数据集和 checkpoint 必须放在 Git 仓库之外，不要提交 `.pt`、
`.safetensors`、LeRobot 数据集或训练输出。

## 功能入口

- [VAE 与 Being-H0.5 后训练](../vae/README.md)
- [MuJoCo 评估与视频](../sim/README.md)

## 配置索引

Being-H0.5 的主要训练配置族如下；NativeVAE 使用 `vae` 模块自己的配置，不在
这里混用：

| 配置族 | 组装后位置 | 用途 |
| --- | --- | --- |
| Shadow grasp | `configs/posttrain/shadow_grasp/` | wrist/`z_gesture`、Rot6D 和 physical-joint 变体 |
| LIBERO | `configs/posttrain/libero/` | LIBERO 后训练 |
| RoboCasa | `configs/posttrain/robocasa/` | Human RoboCasa 后训练 |
| Cross-embodiment | `configs/posttrain/cross-embodiment/` | 混合本体训练 |

Shadow 配置名后缀表示动作变换：`_q99` 为百分位归一化，`_minmax` 为完整
state/action min-max，`_wrist_minmax_zraw` 为 wrist min-max 加原始 latent，
`_wrist_euler_minmax_zraw`/`_wrist_rot6d_minmax_zraw` 区分腕部旋转表示，
`_wrist_rot6d_minmax_joints` 输出物理关节。

训练与评估入口：

- 训练：`scripts/train/train_shadow_grasp_npu.sh`
- TissueBox 闭环演示：`vae/scripts/run_tissuebox_policy_demo.sh`
- MuJoCo 评估：`vae/examples/beingh_shadow_grasp_eval.py`

<a id="source-and-reproducibility"></a>

## 源码版本与许可

- 官方 Being-H 仓库：<https://github.com/BeingBeyond/Being-H.git>
- 公共 submodule 固定提交：`9443a4082bce013cc61e6523596940134179b23e`
- 上游根目录：`third_party/Being-H/Being-H05`
- 复制白名单：`common/upstream_files.txt`
- 共享扩展代码：`common/code/`

NativeVAE 实现位于 `vae/code/vae/native_vae/`。复现实验时，记录本仓库提交、
Being-H 上游提交、Hugging Face 模型 revision 和数据集 revision。

第三方许可：

| 组件 | 来源 | 许可与说明 |
| --- | --- | --- |
| Being-H / Being-H0.5 | [官方固定提交](https://github.com/BeingBeyond/Being-H/commit/9443a4082bce013cc61e6523596940134179b23e) | Apache-2.0；版本见上方 submodule 提交 |
| MuJoCo | <https://mujoco.org/> | Apache-2.0；仅作为可选运行依赖，仓库不包含 MuJoCo 二进制 |

本样例不包含模型 checkpoint 或数据集。TissueBox mesh 的独立来源和再分发提示
记录在 [sim 模块 README](../sim/README.md#6-仿真资产场景清单与来源)。

## 维护约束

- 上游未修改文件应通过 `upstream_files.txt` 引用，不应复制到 `code/`。
- 修改固定上游文件时，应维护最小 patch，而不是提交完整替代文件。
- 只被单一模块使用的代码应留在对应模块，不应放入 `common`。
- 数据集、checkpoint、训练输出和本机绝对路径不得提交到仓库。
- 更新 submodule commit、白名单或 patch 后，应同时验证最小推理组装和完整组装。
