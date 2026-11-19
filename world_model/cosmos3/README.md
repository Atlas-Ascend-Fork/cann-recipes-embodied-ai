# Cosmos3 昇腾 NPU 适配使用指南

本样例在昇腾 A2/A3 上运行 Cosmos3-Nano，支持文生视频（T2V）、图生视频（I2V）、视频生视频（V2V）、文本生图像（T2I）和文本生带音轨视频（T2VS）。适配文件包括 NPU attention 后端、音频模块兼容处理和输入检查。

## 代码仓拉取与适配文件覆盖

先拉取 CANN 适配仓和固定版本的 NVIDIA 原始代码仓，再将 `world_model/cosmos3` 下的适配文件覆盖到 Cosmos3 仓根目录。

```bash
# 进入需要放置代码仓的本地目录，建议让 cann-recipes-embodied-ai 与 cosmos-framework 保持同级
git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git

git clone https://github.com/NVIDIA/cosmos-framework.git && cd cosmos-framework && git checkout a61b292

# 回到两个代码仓的共同上级目录
cd ../

# 将 CANN 仓中的 Cosmos3 适配文件覆盖到 Cosmos3 仓根目录
cp -rf cann-recipes-embodied-ai/world_model/cosmos3/* ./cosmos-framework
```

完成覆盖后，`cosmos-framework` 根目录下应包含 `npu_adapt.sh`、`pyproject.toml` 等适配文件。

## Cosmos3 在昇腾 A2/A3 上的运行环境配置

### 与昇腾平台相关的环境配置

实测环境使用 CANN 9.0.0、Python 3.13.16、torch 2.10.0+cpu 和 torch_npu 2.10.0。A3 实测设备为 Ascend910_9362，A2 为 Ascend910B3；每次推理使用一张卡。单卡实测统计及边界见下文“实测结果”。

从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0)下载同版本 toolkit 和对应硬件的算子包。本次 aarch64 实测安装包如下：

|平台|toolkit|算子包|
|---|---|---|
|A2|Ascend-cann-toolkit_9.0.0_linux-aarch64.run|Ascend-cann-910b-ops_9.0.0_linux-aarch64.run|
|A3|Ascend-cann-toolkit_9.0.0_linux-aarch64.run|Ascend-cann-A3-ops_9.0.0_linux-aarch64.run|

```bash
# 按平台选择算子包；CANN_INSTALL_PATH 为独立安装目录。
export CANN_INSTALL_PATH=/path/to/Ascend
bash Ascend-cann-toolkit_9.0.0_linux-aarch64.run --check
bash Ascend-cann-toolkit_9.0.0_linux-aarch64.run --install --install-path="$CANN_INSTALL_PATH" --quiet
source "$CANN_INSTALL_PATH/cann/set_env.sh"
# A2 示例；A3 替换为表内 A3-ops 包。
bash Ascend-cann-910b-ops_9.0.0_linux-aarch64.run --check
bash Ascend-cann-910b-ops_9.0.0_linux-aarch64.run --install --install-path="$CANN_INSTALL_PATH" --quiet
```

每次新建终端均需加载对应 `set_env.sh`。存在多套 CANN 时，应使用干净终端，检查实际运行进程的库路径，避免混入系统旧版本。复测时记录驱动、设备、Python 依赖版本和实际加载的 CANN 库路径。

### 安装 uv

已有 uv 时跳过此步。

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

### Python 运行环境安装

```bash
cd cosmos-framework
uv sync --python 3.13
```



## 模型权重下载

本样例使用 Cosmos3-Nano 权重进行推理验证。请从 [nvidia/Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano/tree/main) 下载模型权重，并将推理命令中的 `COSMOS_CHECKPOINT` 指向本地权重目录。

此外，视频生成还需要 Wan2.2 VAE 权重。请从 [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/tree/main) 下载 `Wan2.2_VAE.pth`，只需要该 VAE 权重文件，无需下载完整 Wan2.2-TI2V-5B 模型，并将其放置到 Cosmos3-Nano 本地权重目录下。

```bash
# 示例：将 Cosmos3-Nano 权重放置在本地目录 /mnt/workspace/cosmos3/cosmos3-nano
export COSMOS_CHECKPOINT=/mnt/workspace/cosmos3/cosmos3-nano

# Wan2.2 VAE 权重应放置为如下路径
ls ${COSMOS_CHECKPOINT}/Wan2.2_VAE.pth
```

如果部署环境无法直接访问 Hugging Face，可在可联网环境下载完整 Cosmos3-Nano 权重目录和 `Wan2.2_VAE.pth` 后拷贝到昇腾服务器，再使用本地路径作为 `--checkpoint-path`。

## 执行 NPU 适配脚本

```bash
cd cosmos-framework

bash npu_adapt.sh
```

## 推理验证示例

完成适配后，可在 `cosmos-framework` 根目录下执行以下命令进行基础场景验证。`COSMOS_CHECKPOINT` 用于指定本地权重目录；如不设置，可直接将命令中的 `--checkpoint-path` 替换为实际权重路径。

```bash
export COSMOS_CHECKPOINT=/mnt/workspace/cosmos3/cosmos3-nano
export COSMOS_RESOLUTION=480
export COSMOS_SEED=0
export COSMOS_NPUS=1
```

### 输入 JSON 配置说明

推理命令中的 `-i inputs/omni/t2v.json` 用于指定单条样例输入。常用字段如下：

- `model_mode`：任务类型，例如 `text2video`、`image2video`、`video2video`、`text2image`。
- `prompt`：文本提示词；完整固定样例直接使用仓库中对应 JSON。
- `enable_sound`：T2VS 设置为 `true`，与 `model_mode="text2video"` 配合生成音轨；T2I 不启用。
- `vision_path`：I2V/V2V 的输入图片或视频路径，仅图生视频、视频生视频需要。

`vision_path` 可以写远程 URL，也可以写本地文件路径。若服务器无法访问 GitHub/Hugging Face，或遇到证书、代理、内网限制等网络问题，请先手动下载输入图片/视频到本地，然后在 JSON 中改成本地绝对路径，例如：

```json
{
  "model_mode": "image2video",
  "prompt": "A robot arm moves smoothly in a lab.",
  "vision_path": "/mnt/workspace/cosmos3/inputs/robot_153.jpg"
}
```

T2V 示例 JSON 可简化为：

```json
{
  "model_mode": "text2video",
  "name": "t2v",
  "prompt": "A realistic video of molten metal being poured in a steel mill."
}
```


五种场景的输入 JSON 均由锁定的 NVIDIA `cosmos-framework` 提交 `a61b292` 提供，拉取上游后即可使用，无需从适配仓复制。T2I 使用 `model_mode="text2image"`；T2VS 使用 `model_mode="text2video"` 和 `enable_sound=true`。下面仅演示字段，复现实测请使用上游 [t2i.json](https://github.com/NVIDIA/cosmos-framework/blob/a61b292ccbc337c25c0c82d2a8cca75feef96cee/inputs/omni/t2i.json)、[t2vs.json](https://github.com/NVIDIA/cosmos-framework/blob/a61b292ccbc337c25c0c82d2a8cca75feef96cee/inputs/omni/t2vs.json) 中完整的结构化 prompt，不以缩写替换：

```json
{"model_mode": "text2image", "name": "t2i", "prompt": "A modern robotics research laboratory."}
```

```json
{"model_mode": "text2video", "name": "t2vs", "enable_sound": true, "prompt": "A video with synchronized environmental sound."}
```

T2I、T2VS 仅需文本，无外部输入媒体。I2V/V2V 固定输入来自 NVIDIA 官方 `cosmos-dependencies` 仓库的提交 `2b17a2413bd86b2cf9b03823637108851e4ddf2d`：

- [I2V 输入图片 robot_153.jpg](https://github.com/nvidia-cosmos/cosmos-dependencies/raw/2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/vision/robot_153.jpg)
- [V2V 输入视频 robot_pouring.mp4](https://github.com/nvidia-cosmos/cosmos-dependencies/raw/2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/vision/robot_pouring.mp4)

可以使用 JSON 中的官方 URL。需要离线运行时，在可联网机器执行以下命令，校验通过后将媒体复制到运行机器，并把 `inputs/omni/i2v.json`、`inputs/omni/v2v.json` 的 `vision_path` 分别改为对应文件的本地绝对路径。其余字段保持不变。

```bash
mkdir -p sample_media
curl --fail --location --retry 3 \
  https://github.com/nvidia-cosmos/cosmos-dependencies/raw/2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/vision/robot_153.jpg \
  -o sample_media/robot_153.jpg
curl --fail --location --retry 3 \
  https://github.com/nvidia-cosmos/cosmos-dependencies/raw/2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/vision/robot_pouring.mp4 \
  -o sample_media/robot_pouring.mp4
sha256sum --check <<'EOF'
886e6cfa8a0d6c4d96459cfe79b51c56aff4d63e1c523f8366f1771ea9c2788d  sample_media/robot_153.jpg
75b38640974fd5ad8245fa610acee9b0efe1f29d3c8fea829799d7b8fa799be8  sample_media/robot_pouring.mp4
EOF
```


### T2V 文生视频

```bash
torchrun --nproc-per-node=${COSMOS_NPUS} -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i inputs/omni/t2v.json \
    -o outputs/t2v \
    --checkpoint-path ${COSMOS_CHECKPOINT} \
    --resolution=${COSMOS_RESOLUTION} \
    --seed=${COSMOS_SEED} \
    --no-guardrails
```

### I2V 图生视频

```bash
torchrun --nproc-per-node=${COSMOS_NPUS} -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i inputs/omni/i2v.json \
    -o outputs/i2v \
    --checkpoint-path ${COSMOS_CHECKPOINT} \
    --resolution=${COSMOS_RESOLUTION} \
    --seed=${COSMOS_SEED} \
    --no-guardrails
```

### V2V 视频生视频

```bash
torchrun --nproc-per-node=${COSMOS_NPUS} -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i inputs/omni/v2v.json \
    -o outputs/v2v \
    --checkpoint-path ${COSMOS_CHECKPOINT} \
    --resolution=${COSMOS_RESOLUTION} \
    --seed=${COSMOS_SEED} \
    --no-guardrails
```

### T2I 文本生图像

```bash
torchrun --nproc-per-node=${COSMOS_NPUS} -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i inputs/omni/t2i.json \
    -o outputs/t2i \
    --checkpoint-path ${COSMOS_CHECKPOINT} \
    --resolution=${COSMOS_RESOLUTION} \
    --seed=${COSMOS_SEED} \
    --no-guardrails
```

输出为 `vision.jpg`；`resolution=480` 是上游档位，固定样例实际生成 640×640 图像。每次复测使用未运行过的输出目录，避免上游跳过已有结果。

### T2VS 文本生带音轨视频

```bash
torchrun --nproc-per-node=${COSMOS_NPUS} -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i inputs/omni/t2vs.json \
    -o outputs/t2vs \
    --checkpoint-path ${COSMOS_CHECKPOINT} \
    --resolution=${COSMOS_RESOLUTION} \
    --seed=${COSMOS_SEED} \
    --no-guardrails
```

使用包含 sound tokenizer 的 Cosmos3-Nano checkpoint；Edge 不支持带音频场景。输出 `vision.mp4` 含视频轨和音频轨。音频 tokenizer 在首次音频请求时加载，缺失 AVAE 文件或 AAC 编码依赖会报错。NPU 音频预处理仅将 Hann 窗在 CPU 构造后传回 waveform 设备，STFT 和生成流程保持上游实现。

### 多卡并行

当前适配支持 CP、CFGP 和 FSDP 多卡推理。运行前请将 `COSMOS_NPUS` 设置为实际使用的 NPU 数量，并保证各并行度与进程数匹配。

| 并行方式 | 主要参数 | 适用目的与约束 |
| --- | --- | --- |
| CP（Context Parallel） | `--cp-size` | 沿 token 序列切分 Attention 计算，适合长序列并降低激活显存；当前 CP 范围为 1～32 |
| CFGP（Classifier-Free Guidance Parallel） | `--cfgp-size` | 将有条件与无条件 CFG 分支分配到不同设备；CFGP 仅支持 1 或 2，更大规模可与 CP/FSDP 组合 |
| FSDP | `--dp-shard-size` | 按进程数切分模型参数，优先降低单卡权重显存 |

FSDP 的 DP 通信组与 CP/CFGP 通信组相互独立，通信域大小满足：

```text
dp-shard-size × dp-replicate-size = WORLD_SIZE
WORLD_SIZE % (cp-size × cfgp-size) = 0
```


CP：

```bash
torchrun --nproc-per-node=${COSMOS_NPUS} -m cosmos_framework.scripts.inference \
    --parallelism-preset=throughput \
    --dp-shard-size=1 --cp-size=${COSMOS_NPUS} --cfgp-size=1 \
    -i inputs/omni/t2v.json \
    -o outputs/t2v_cp \
    --checkpoint-path ${COSMOS_CHECKPOINT} \
    --resolution=${COSMOS_RESOLUTION} \
    --seed=${COSMOS_SEED} \
    --no-guardrails
```

CFGP（单独启用时设置 `COSMOS_NPUS=2`）：

```bash
torchrun --nproc-per-node=${COSMOS_NPUS} -m cosmos_framework.scripts.inference \
    --parallelism-preset=throughput \
    --dp-shard-size=1 --cp-size=1 --cfgp-size=2 \
    -i inputs/omni/t2v.json \
    -o outputs/t2v_cfgp \
    --checkpoint-path ${COSMOS_CHECKPOINT} \
    --resolution=${COSMOS_RESOLUTION} \
    --seed=${COSMOS_SEED} \
    --no-guardrails
```

FSDP：

```bash
torchrun --nproc-per-node=${COSMOS_NPUS} -m cosmos_framework.scripts.inference \
    --parallelism-preset=throughput \
    --dp-shard-size=${COSMOS_NPUS} --cp-size=1 --cfgp-size=1 \
    -i inputs/omni/t2v.json \
    -o outputs/t2v_fsdp \
    --checkpoint-path ${COSMOS_CHECKPOINT} \
    --resolution=${COSMOS_RESOLUTION} \
    --seed=${COSMOS_SEED} \
    --no-guardrails
```


## 样例输出展示

以下为在昇腾 NPU 上运行上述基础场景得到的示例输出，可用于快速查看生成效果。

| 场景 | 示例输出 |
| --- | --- |
| T2V 文生视频 | ![t2v_vision.gif](https://raw.gitcode.com/user-images/assets/10199195/600da397-f263-4e82-96e9-4589de9ebc2a/t2v_vision.gif) |
| I2V 图生视频 | ![i2v_vision.gif](https://raw.gitcode.com/user-images/assets/10199195/a493e495-4dc5-4a83-9517-5d34f9d7e757/i2v_vision.gif) |
| V2V 视频生视频 | ![v2v_vision.gif](https://raw.gitcode.com/user-images/assets/10199195/bf6f811d-a96d-4511-af23-0a6ca65380a2/v2v_vision.gif) |
| T2I 文本生图像 | ![T2I 实测输出](https://raw.gitcode.com/user-images/assets/7380116/a575db7e-af8c-4cdc-b08e-cde845b01875/cosmos3_t2i_a2.jpg) |
| T2VS 文本生带音轨视频 | ![T2VS 无声预览](https://raw.gitcode.com/user-images/assets/7380116/b4260cfe-6fb3-4737-932f-48526358eb5a/cosmos3_t2vs_preview.gif) [下载原始带音轨 MP4](https://raw.gitcode.com/user-images/assets/7380116/151fbdbf-ac2c-45a9-8ac5-9ae484604058/cosmos3_t2vs_a2.mp4) |

T2I 图片和 T2VS 视频来自 A2 实测；GIF 为原视频降采样的无声预览，听取音轨请下载并播放原始 MP4。生成效果随模型、输入和参数变化。

## 实测结果

以下是固定上游 `a61b292`、CANN 9.0.0、Python 3.13.16、torch/torch_npu 2.10.0 的历史单卡测量。本轮仅整理文档及合入文件范围，未重新执行设备全矩阵。每组独立预热一次、正式三次；统计请求准备、生成、解码及保存的同步墙钟时间，不含模型加载，关闭 debug/profiler。不同设备批次不拼接成性能对照。

|平台|场景|参考实现正式中位（秒）|场景适配实现正式中位（秒）|参考三次最大（秒）|
|---|---|---:|---:|---:|
|A3 Ascend910_9362|T2V|286.404035|286.235047|286.422887|
|A3 Ascend910_9362|I2V|285.981710|286.294036|286.718152|
|A3 Ascend910_9362|V2V|276.315432|276.193762|276.491707|
|A2 Ascend910B3，同卡配对|T2V|257.838199|257.996490|257.899051|

|平台|场景|场景适配实现正式中位（秒）|
|---|---|---:|
|A3 Ascend910_9362|T2I|24.443827|
|A3 Ascend910_9362|T2VS|292.477118|
|A2 Ascend910B3|T2VS|269.827472|

A3 三项对照满足“适配中位不超过参考三次最大”判据。A2 T2V 初轮超过该边界 0.4353%，同卡配对超过 0.097439 秒（0.037782%）；该差值已获任务用户接受，但仍超出原判据；现有样本不足以确定差异来源。

|验证内容|A2/A3 实测结果及限制|
|---|---|
|T2V/I2V/V2V 对照|原始张量差异元素为 0、最大绝对误差为 0，解码像素一致|
|T2I|生成有效图像，固定输入对照通过|
|T2VS|非空、非全零的 48kHz 双声道音轨；音画有效时长差约 5ms，实际播放正常|
|内存|五种场景均有 profile_memory 记录；A2 初始化后六次相同 T2I 的结束 allocated 在 34608014848 / 34608223232 bytes 间交替，实测窗口内无持续增长|

连续音频请求曾出现 allocated 增长，上述内存稳定结果仅适用于已测 T2I 序列，不代表任意混合调用序列均无增长。实现细节见 [Cosmos 适配与优化说明](../../docs/world_model/Cosmos/README.md#5-cosmos3-场景适配)。

## 引用

```bibtex
% Citation
@misc{nvidia2026cosmos3omnimodalworldmodels,
      title={Cosmos 3: Omnimodal World Models for Physical AI},
      author={NVIDIA: Aditi, Niket Agarwal, Arslan Ali, Jon Allen, Martin Antolini, Adeline Aubame, Alisson Azzolini, Junjie Bai, Maciej Bala, Yogesh Balaji, Josh Bapst, Aarti Basant, Mukesh Beladiya, Mohammad Qazim Bhat, Zaid Pervaiz Bhat, Dan Blick, Vanni Brighella, Han Cai, Tiffany Cai, Eric Cameracci, Jiaxin Cao, Yulong Cao, Mark Carlson, Carlos Casanova, Ting-Yun Chang, Yan Chang, Yu-Wei Chao, Prithvijit Chattopadhyay, Roshan Chaudhari, Chieh-Yun Chen, Junyu Chen, Ke Chen, Qizhi Chen, Wenkai Chen, Xiaotong Chen, Yu Chen, An-Chieh Cheng, Click Cheng, Xiu Chia, Jeana Choi, Chaeyeon Chung, Wenyan Cong, Yin Cui, Magdalena Dadela, Nalin Dadhich, Wenliang Dai, Joyjit Daw, Alperen Degirmenci, Rodrigo Vieira Del Monte, Robert Denomme, Sameer Dharur, Marco Di Lucca, Ke Ding, Wenhao Ding, Yifan Ding, Yuzhu Dong, Nicole Drumheller, Yilun Du, Aigul Dzhumamuratova, Aleksandr Efitorov, Hamid Eghbalzadeh, Naomi Eigbe, Imad El Hanafi, Hassan Eslami, Benedikt Falk, Jiaojiao Fan, Jim Fan, Amol Fasale, Sergiy Fefilatyev, Liang Feng, Francesco Ferroni, Sanja Fidler, Xiao Fu, Vikram Fugro, Prashant Gaikwad, TJ Galda, Katelyn Gao, Yihuai Gao, Wenhang Ge, Sreyan Ghosh, Arushi Goel, Vivek Goel, Akash Gokul, Rama Govindaraju, Jinwei Gu, Miguel Guerrero, Elfie Guo, Aryaman Gupta, Siddharth Gururani, Hugo Hadfield, Song Han, Ankur Handa, Zekun Hao, Mohammad Harrim, Ali Hassani, Nathan Hayes-Roth, Yufan He, Chris Helvig, Cyrus Hogg et al. (195 additional authors not shown)},
      year={2026},
      eprint={2606.02800},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.02800},
}
```
