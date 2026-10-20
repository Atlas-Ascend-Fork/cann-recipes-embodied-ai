# AnchorSplat 昇腾训练与评测样例

AnchorSplat 是一种利用三维几何先验的前馈式 3D Gaussian Splatting
场景重建方法。模型从多视图图像预测相机、深度和稀疏三维锚点，再从锚点特征
解码高斯属性。

- 论文：[AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](https://arxiv.org/abs/2604.07053)
- 代码：[Zhang-Xiaoxue/AnchorSplat](https://github.com/Zhang-Xiaoxue/AnchorSplat)

AnchorSplat 的昇腾实现已包含在上游仓库中。因此本样例不复制模型源码或补丁，
只提供固定版本的环境准备、训练和评测启动脚本。

## 固定版本

样例默认使用以下 AnchorSplat commit：

```text
7f1625fccc431bd53c961649312a1b2a077e39f4
```

下文实现说明中的源码路径均相对此版本的 AnchorSplat 上游仓库。

## 模型流程

1. Depth Anything 3 从多视图图像预测深度与相机信息；
2. CNN 编码图像、深度与 Plücker ray；
3. 体素采样器生成稀疏三维锚点；
4. Point Transformer 聚合跨视角特征；
5. Gaussian head 解码高斯属性，并通过可微光栅化监督训练。

## 昇腾执行路径

- Point Transformer 在 `src/depth_anything_3/model/point_net.py` 中调用
  `torch_npu.npu_fusion_attention`；
- 训练器使用 HCCL DistributedDataParallel，并支持 BF16、FP16 和 FP32；本仓库训练使用BF16.
- `src/depth_anything_3/model/utils/rasterizer.py` 调用
  `meta_gauss_render` 完成投影、排序、调度和渲染；
- `src/depth_anything_3/model/utils/projection_three_dims_gaussian_fused.py`
  保留投影算子的自动求导封装。

## 环境准备

使用 Python 3.10，并提前安装与目标 CANN 版本匹配的 PyTorch、torchvision 和
torch-npu。加载 CANN 环境后，在 cann-recipes 仓库根目录执行：

```bash
conda create -n anchorsplat python=3.10 -y
conda activate anchorsplat
# 按 CANN/torch-npu 配套关系安装 torch、torchvision 和 torch-npu。
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./3d_vision/anchorsplat/src/scripts/setup.sh
```

脚本默认将 AnchorSplat clone 到 cann-recipes 仓库的同级目录：

```text
workspace/
├── cann-recipes-embodied-ai/
└── AnchorSplat/
```

如果已经存在本地仓库，可通过 `ANCHORSPLAT_ROOT` 指定：

```bash
ANCHORSPLAT_ROOT=/path/to/AnchorSplat \
  ./3d_vision/anchorsplat/src/scripts/setup.sh
```

NPU 渲染还依赖 CANN Python 包 `acl` 和与 AnchorSplat 接口匹配的
`meta_gauss_render`。

### Gaussian 算子兼容性

固定版本使用的 `meta_gauss_render` Python 接口包括：

- `get_render_schedule_cpp`；
- 九参数形式的 `flash_gaussian_build_mask`；
- `gaussian_sort(mask, depths)`。

## 数据与权重

在上游
`src/depth_anything_3/configs/anchorsplat_mix.yaml`
中修改 Depth Anything 3 权重路径和数据集根目录。数据格式、场景列表和配置项以
[AnchorSplat 上游 README](https://github.com/Zhang-Xiaoxue/AnchorSplat#dataset-format)
为准。

## 训练

默认启动 8 个 NPU 进程：

```bash
./3d_vision/anchorsplat/src/scripts/run_train.sh
```

可覆盖上游目录、卡数、配置和输出目录：

```bash
ANCHORSPLAT_ROOT=/path/to/AnchorSplat \
NPUS=4 \
CONFIG=/path/to/anchorsplat_mix.yaml \
OUTPUT=/path/to/output \
  ./3d_vision/anchorsplat/src/scripts/run_train.sh
```

运行前可先检查最终命令：

```bash
DRY_RUN=1 ./3d_vision/anchorsplat/src/scripts/run_train.sh
```

## 评测

```bash
CKPT=/path/to/epoch_xxxx.ckpt \
  ./3d_vision/anchorsplat/src/scripts/run_test.sh
```

## 验证建议

1. 确认 `torch`、`torch_npu`、`acl` 和 `meta_gauss_render` 均可导入；
2. 先执行单卡小数据前向、反向和 checkpoint 保存；
3. 再执行多卡 HCCL 训练；
4. 固定数据和 checkpoint，记录 PSNR、SSIM、LPIPS；
5. 记录稳态 step 时间、吞吐量与峰值显存。

## 目录说明

```text
3d_vision/anchorsplat/
├── README.md
└── src/scripts/
    ├── setup.sh
    ├── run_train.sh
    └── run_test.sh
```
