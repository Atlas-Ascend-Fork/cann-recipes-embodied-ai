# HY-Embodied-0.5-X NPU 推理与 EmbSpatial-Bench 评测

本样例使用腾讯混元（Tencent Hunyuan）的 HY-Embodied-0.5-X 模型，在 Atlas A2 / A3 上提供 BF16 图像与文本推理。支持直接回答（direct）和先推理后回答（thinking）两种模式，以及单图和串行批量输入；同时提供 EmbSpatial-Bench 评测、断点续跑和结果重新统计分数工具。

每个推理实例在一张 NPU 上加载完整模型。视觉编码器、MoT 多模态路由、语言主干、注意力键值缓存（KV Cache）和输出层的计算在 NPU 上执行；图片解码、分词和结果文件处理由 CPU 完成。使用方式为命令行调用。

## 目录说明

样例按模型加载、NPU 适配、推理与评测组织。命令从仓库根目录执行，下面列出样例目录内的文件及作用。

```text
infer_with_torch/
├── README.md                     # 安装、推理、评测及重新统计分数说明
├── requirements.txt              # Python 依赖及固定源码来源
├── configs/
│   ├── asset-lock.json           # 模型、数据版本和文件信息
│   ├── request-single.json       # 单图推理配置
│   ├── request-batch.json        # 串行批量推理配置
│   └── evaluation.json           # EmbSpatial-Bench 评测配置
├── src/
│   ├── infer.py                  # 推理入口、参数与输入检查
│   ├── model_loader.py           # 依赖校验、模型加载及适配接入
│   ├── command_io.py             # JSON 配置读取与命令输出
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── attention.py          # Attention 的 NPU 适配
│   │   ├── gelu.py               # 模型实例级 erf GELU 适配
│   │   ├── transformers_patch.py # Transformers 补丁与备份
│   │   └── graph_decode.py       # 图解码、模态缓存及包装清理
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── generation.py         # 输入准备、随机种子、生成及解码
│   │   └── monitoring.py         # 耗时和内存统计
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluator.py          # 逐题评测、异常记录及续跑
│       ├── protocol.py           # 数据格式、问题构造及严格计分
│       └── results.py            # 结果存储、续跑核对、重新统计分数及导出
├── tools/
│   ├── evaluate_embspatial.py     # EmbSpatial-Bench 评测命令入口
│   └── score_results.py          # CPU 重新统计分数及 JSON、CSV 导出入口
├── tests/
│   ├── test_protocol.py          # CPU 问题构造、解析和计分测试
│   └── test_runtime.py           # CPU 参数、输入、续跑及重新统计分数测试
└── bench/
    ├── check_ops.py               # NPU Attention、GELU 检查
    └── check_model.py             # 推理功能、状态隔离及可选采集
```

`model_loader.py`、`adapters/`、`inference/` 组成核心推理路径；`evaluation/` 调用这条路径执行评测。适配原理和测试目的见 [NPU 适配与验证说明](../../../docs/manipulation/hy-embodied-0.5-x/infer_with_torch/npu-adaptation.md)。

## 运行环境

| 组件 | 配置 |
| --- | --- |
| 昇腾计算平台 | Atlas A2 / Atlas A3，单卡推理 |
| 操作系统架构 | Linux aarch64 |
| Python | 3.12.15 |
| 昇腾 CANN 工具包 | 9.1.0 |
| 昇腾 NPU 驱动（Ascend HDK Driver） | 26.1.1 或 25.5.0；安装包需与 Atlas 设备型号及 CANN 9.1.0 配套 |
| PyTorch / 昇腾 PyTorch 扩展 torch_npu | 2.10.0 / 2.10.0.post6 |
| Transformers | 腾讯混元指定的源码提交 [`9293856c419762ebf98fbe2bd9440f9ce7069f1a`](https://github.com/huggingface/transformers/commit/9293856c419762ebf98fbe2bd9440f9ce7069f1a) |
| 模型权重（腾讯混元） | Hugging Face：`tencent/HY-Embodied-0.5-X` |
| 评测数据（EmbSpatial-Bench） | Hugging Face：`Phineas476/EmbSpatial-Bench`，human-verified evaluation |

相关依赖见 [requirements.txt](requirements.txt)。模型和数据的 revision、文件校验信息见 [configs/asset-lock.json](configs/asset-lock.json)。

注：本样例实现依赖了未合入Transformers主仓的提交版本，该提交由腾讯混元指定，详情见：https://github.com/huggingface/transformers/commit/9293856c419762ebf98fbe2bd9440f9ce7069f1a

昇腾 NPU 驱动安装在宿主机，容器使用宿主机提供的设备和驱动。可以用以下命令查看 NPU 状态及驱动版本：

```bash
npu-smi info
```

## 快速开始：运行一张自己的图片

### 安装模型运行环境

以下命令均从仓库根目录执行，即包含 `manipulation/` 和 `docs/` 的目录。创建独立 Python 3.12 环境并加载已安装的 CANN，将 `/path/to/` 替换为实际路径。

```bash
python3.12 -m venv manipulation/hy-embodied-0.5-x/infer_with_torch/.venv
source manipulation/hy-embodied-0.5-x/infer_with_torch/.venv/bin/activate
source /path/to/cann/set_env.sh
python -m pip install \
  -r manipulation/hy-embodied-0.5-x/infer_with_torch/requirements.txt
export PYTHONPATH="$PWD/manipulation/hy-embodied-0.5-x/infer_with_torch/src:${PYTHONPATH:-}"
```

腾讯混元的 [安装脚本](https://github.com/Tencent-Hunyuan/HY-Embodied-0.5-X/blob/cf23678a76dc9c445ab7f7aea7a9a8f60e910a63/setup_env.sh#L63-L65) 指定 Transformers 源码提交 `9293856c419762ebf98fbe2bd9440f9ce7069f1a`。[requirements.txt](requirements.txt) 同时固定这一提交和腾讯混元 HY-Embodied-0.5-X 源码提交 `cf23678a76dc9c445ab7f7aea7a9a8f60e910a63`。Transformers 的包内版本字符串为 `4.57.0`，安装时请使用依赖文件中指定的源码来源。

首次加载会在该独立环境的 Transformers 模型文件中应用 NPU 适配，并保存 `.py.hy-original` 备份。该环境需要对模型文件所在目录有写权限；补丁内容、恢复方法及进程使用条件见[NPU 适配与验证说明](../../../docs/manipulation/hy-embodied-0.5-x/infer_with_torch/npu-adaptation.md)。

### 模型准备与加载

优先使用已有的同一版本权重。需要下载时执行：

```bash
hf download tencent/HY-Embodied-0.5-X \
  --revision 79f8301d8a80dd03ba778c58d6429e414e9ef5d3 \
  --local-dir /path/to/model
```

将 `/path/to/model` 替换为实际模型目录，保留目录中的权重、配置、tokenizer、processor、chat template 和 LICENSE 文件。

上述命令将模型文件下载到本地。执行下一节的推理命令时，程序会从该目录加载模型，并将模型放到指定 NPU 上。

### 运行单图推理

下面以 EmbSpatial-Bench 样本 `mp3d_13` 为例。输入是一张 640×480 的室内图片，问题询问抽屉柜与镜子的相对位置。需要使用同一张图片时，可按[从评测数据中提取单个样本](../../../docs/manipulation/hy-embodied-0.5-x/infer_with_torch/embspatial-bench-evaluation.md#从评测数据中提取单个样本)准备图片和问题。

![EmbSpatial-Bench 样本 mp3d_13 的原始输入图片](https://raw.gitcode.com/cann/cann-recipes-embodied-ai/attachment/uploads/a6d3fb29-c788-4ced-afcd-a4172589b3da/mp3d_13.jpg)

图片来源：[Phineas476/EmbSpatial-Bench](https://huggingface.co/datasets/Phineas476/EmbSpatial-Bench)，样本 `mp3d_13`，原始尺寸为 640×480。

将模型目录和图片路径替换为实际路径，执行以下命令：

```bash
python manipulation/hy-embodied-0.5-x/infer_with_torch/src/infer.py \
  --model-path /path/to/model \
  --image /path/to/mp3d_13.jpg \
  --prompt 'What is the spatial configuration between dresser and mirror in relation to each other within the image?
A. The dresser is touching the mirror.
B. The dresser is above the mirror.
C. The dresser is blocking the mirror.
D. The dresser is below the mirror.
Select the best answer. Put only one letter (A, B, C or D) in the final answer.' \
  --device npu:0 \
  --dtype bfloat16 \
  --decode-backend graph \
  --disable-thinking \
  --max-new-tokens 32768 \
  --temperature 0.05 \
  --top-p 1.0 \
  --seed 55
```

**输入参数说明**

| 参数 | 本例设置 | 说明 |
| --- | --- | --- |
| `--model-path` | `/path/to/model` | 本地模型目录 |
| `--image` | `/path/to/mp3d_13.jpg` | 输入图片的本地路径 |
| `--prompt` | 上述完整问题及四个选项 | 模型需要回答的问题，单引号内的多行内容作为一个参数传入 |
| `--device` | `npu:0` | 使用逻辑 NPU 0 |
| `--dtype` | `bfloat16` | 模型使用 BF16 精度 |
| `--decode-backend` | `graph` | 使用图解码后端 |
| `--disable-thinking` | 开关，无需取值 | 使用直接回答模式 |
| `--max-new-tokens` | `32768` | 最多生成的 token 数 |
| `--temperature` | `0.05` | 调节采样分布的温度 |
| `--top-p` | `1.0` | 核采样的累计概率阈值 |
| `--seed` | `55` | 本次推理使用的随机种子 |

**输出结果**

程序为每张输入图片输出一行 JSON。以下为该样本一次 NPU 推理的输出示例，展开排版显示：

```json
{
  "index": 0,
  "seed": 55,
  "thinking": false,
  "device": "npu:0",
  "dtype": "BF16",
  "decode_backend": "graph",
  "input_tokens": 408,
  "output_tokens": 6,
  "output": "<answer>\nD\n</answer>",
  "error": null
}
```

本例的模型回答为 **D**，对应“The dresser is below the mirror.”，即“抽屉柜在镜子下方”。

**输出字段说明**

| 字段 | 本例值 | 说明 |
| --- | --- | --- |
| `index` | `0` | 当前请求中的输入序号，从 0 开始 |
| `seed` | `55` | 本次实际使用的随机种子 |
| `thinking` | `false` | 本次使用直接回答模式 |
| `device` | `npu:0` | 执行推理的设备 |
| `dtype` | `BF16` | 模型推理使用的数据类型 |
| `decode_backend` | `graph` | 本次实际使用的解码后端 |
| `input_tokens` | `408` | 图片和文本经过处理后的输入序列 token 数 |
| `output_tokens` | `6` | 本次生成的输出 token 数 |
| `output` | `<answer>\nD\n</answer>` | 模型生成的原文，其中 `\n` 表示换行 |
| `error` | `null` | 本次推理正常结束；发生异常时保存错误信息 |

查看全部命令行参数：

```bash
python manipulation/hy-embodied-0.5-x/infer_with_torch/src/infer.py --help
```

## 参数配置与运行方式

以下命令均从仓库根目录执行，使用快速开始中准备好的 Python 环境和 `PYTHONPATH`。配置中的相对路径也以命令执行时的仓库根目录为基准。

### 参数配置方法

推理和评测都支持三种方式：直接传入命令行参数、使用 JSON 配置，以及在 JSON 配置上叠加命令行参数。取值优先级为：**显式命令行参数 > JSON 配置 > 程序默认值**。

| 方式 | 示例 | 用途 |
| --- | --- | --- |
| 命令行参数 | `python manipulation/hy-embodied-0.5-x/infer_with_torch/src/infer.py --model-path ... --image ... --prompt ...` | 临时指定输入和生成参数 |
| JSON 配置 | `python manipulation/hy-embodied-0.5-x/infer_with_torch/src/infer.py --config manipulation/hy-embodied-0.5-x/infer_with_torch/configs/request-single.json` | 保存并复用一组参数 |
| JSON 加命令行覆盖 | `python manipulation/hy-embodied-0.5-x/infer_with_torch/src/infer.py --config manipulation/hy-embodied-0.5-x/infer_with_torch/configs/request-single.json --disable-thinking --seed 55` | 复用配置并修改部分参数 |

JSON 字段使用下划线命名，例如 `model_path`、`max_new_tokens`。`thinking` 使用 JSON 布尔值 `true` 或 `false`；`image`、`prompt` 可以使用字符串或字符串数组。

例如，单图配置可以写为：

```json
{
  "model_path": "/path/to/model",
  "image": "/path/to/image.jpg",
  "prompt": "Describe this image.",
  "thinking": false,
  "max_new_tokens": 32768,
  "temperature": 0.05,
  "top_p": 1.0,
  "seed": 42,
  "decode_backend": "graph"
}
```

样例提供的配置文件如下。使用前修改其中的本地路径。

| 配置文件 | 用途与初始模式 | 运行命令 |
| --- | --- | --- |
| [request-single.json](configs/request-single.json) | 单图，thinking | `python manipulation/hy-embodied-0.5-x/infer_with_torch/src/infer.py --config manipulation/hy-embodied-0.5-x/infer_with_torch/configs/request-single.json` |
| [request-batch.json](configs/request-batch.json) | 串行批量，direct | `python manipulation/hy-embodied-0.5-x/infer_with_torch/src/infer.py --config manipulation/hy-embodied-0.5-x/infer_with_torch/configs/request-batch.json` |
| [evaluation.json](configs/evaluation.json) | EmbSpatial-Bench 完整评测，thinking | `python manipulation/hy-embodied-0.5-x/infer_with_torch/tools/evaluate_embspatial.py --config manipulation/hy-embodied-0.5-x/infer_with_torch/configs/evaluation.json` |

### 推理参数

| 命令行参数 | JSON 字段 | 默认值或要求 |
| --- | --- | --- |
| `--model-path` | `model_path` | 必填，本地模型目录 |
| `--image` | `image` | 必填，一个或多个图片路径 |
| `--prompt` | `prompt` | 必填，一个共享问题或与图片等量的问题 |
| `--device` | `device` | `npu:0` |
| `--dtype` | `dtype` | `bfloat16`，也接受 `BF16` |
| `--decode-backend` | `decode_backend` | `graph`，也支持 `native` |
| `--enable-thinking` / `--disable-thinking` | `thinking` | `false`，命令行通过开关选择 |
| `--max-new-tokens` | `max_new_tokens` | `32768` |
| `--temperature` | `temperature` | `0.05`；设为 `0` 时贪心生成 |
| `--top-p` | `top_p` | `1.0` |
| `--seed` | `seed` | `42` |
| `--timeout-seconds` | `timeout_seconds` | `7200`，单次生成超时秒数 |
| `--config` | — | 读取 JSON 配置文件 |

### 单图与串行批量

单图推理见快速开始。串行批量复用一个模型，按 `--image` 的顺序逐张处理。下面两张图片使用同一个问题：

```bash
python manipulation/hy-embodied-0.5-x/infer_with_torch/src/infer.py \
  --model-path /path/to/model \
  --image /path/to/image-1.jpg /path/to/image-2.jpg \
  --prompt 'Describe this image.'
```

逐图指定问题时，问题数量应与图片数量相同：

```bash
python manipulation/hy-embodied-0.5-x/infer_with_torch/src/infer.py \
  --model-path /path/to/model \
  --image /path/to/image-1.jpg /path/to/image-2.jpg \
  --prompt 'Describe the first image.' 'What objects are visible in the second image?'
```

每条输入输出一行 JSON，`index` 与输入顺序对应。从 0 开始的第 i 条输入使用基础 seed 加 i，并为该条输入创建独立的生成状态和 KV Cache。

### direct 与 thinking

| 模式 | 命令行设置 | JSON 设置 | 生成内容 |
| --- | --- | --- | --- |
| direct | `--disable-thinking` | `"thinking": false` | 直接回答 |
| thinking | `--enable-thinking` | `"thinking": true` | 推理过程及最终回答 |

下面复用单图配置，并显式启用 thinking：

```bash
python manipulation/hy-embodied-0.5-x/infer_with_torch/src/infer.py \
  --config manipulation/hy-embodied-0.5-x/infer_with_torch/configs/request-single.json \
  --enable-thinking
```

腾讯混元的输入模板提供 `<think>` 开头，thinking 输出包含后续推理、`</think>` 和最终回答。达到 token 上限且未输出结束标记时，保留已生成文本，并将 `error` 记为 `GenerationTokenLimit`。

### graph 与 native

解码是模型逐步生成输出 token 的过程；这里的后端决定这些计算的执行方式。

| 后端 | 设置 | 执行方式 |
| --- | --- | --- |
| `graph` | `--decode-backend graph` | 默认路径，使用 NPU 图执行，并缓存文本/视觉分支的判断结果 |
| `native` | `--decode-backend native` | 保留必要的 NPU 适配，调用腾讯混元模型的 forward 方法及 KV Cache 更新 |

例如，使用原生路径运行同一份单图配置：

```bash
python manipulation/hy-embodied-0.5-x/infer_with_torch/src/infer.py \
  --config manipulation/hy-embodied-0.5-x/infer_with_torch/configs/request-single.json \
  --decode-backend native
```

后端在模型加载前选择，每个进程使用一种后端。需要修改解码实现时，可查看[两种解码方式的实现说明](../../../docs/manipulation/hy-embodied-0.5-x/infer_with_torch/npu-adaptation.md#解码方式与资源释放)；需要恢复环境中的 Transformers 文件时，按[补丁恢复方法](../../../docs/manipulation/hy-embodied-0.5-x/infer_with_torch/npu-adaptation.md#transformers-文件补丁与恢复)操作。

## EmbSpatial-Bench 评测

模型目录及 Python 环境沿用快速开始中的准备结果，以下命令均从仓库根目录执行。

### 准备评测数据

下载 [Phineas476/EmbSpatial-Bench](https://huggingface.co/datasets/Phineas476/EmbSpatial-Bench) 的 human-verified evaluation 数据：

```bash
# 下载固定版本的评测数据到 /path/to/dataset
hf download Phineas476/EmbSpatial-Bench embspatial_bench.json \
  --repo-type dataset \
  --revision e2733cacf0f20371518c687176e881d79644e61e \
  --local-dir /path/to/dataset
```

`--repo-type dataset` 选择数据仓库，`--revision` 指定数据版本，`--local-dir` 指定本地保存目录。该文件包含 3,640 道题，图片已编码在 JSON 中，评测程序会直接读取。需要检查其中的问题、选项或标准答案时，查看[评测数据文件的字段说明](../../../docs/manipulation/hy-embodied-0.5-x/infer_with_torch/embspatial-bench-evaluation.md#评测数据文件)。

### 运行评测

先查看本次运行会使用和生成的文件：

```text
仓库根目录/
└── manipulation/hy-embodied-0.5-x/infer_with_torch/configs/
    ├── evaluation.json                  # 修改模型、数据、结果路径和生成参数
    └── asset-lock.json                  # 随样例提供，程序自动读取版本和文件信息

/path/to/
├── model/                               # 输入：快速开始准备的完整模型目录
├── dataset/
│   └── embspatial_bench.json             # 输入：下载的完整评测数据
└── results/
    ├── embspatial.jsonl                  # 评测生成：逐题结果
    ├── embspatial.protocol.json          # 评测生成：续跑及重新统计分数使用的运行配置
    ├── embspatial.summary.json           # 评测生成：本次运行汇总
    └── summary/                         # 重新统计分数时生成
        ├── summary.json                 # 重新计算的总分和分类统计
        └── per-question.csv             # 逐题明细表
```

编辑 [evaluation.json](configs/evaluation.json)，将三个路径改为实际位置：

| JSON 字段 | 对应命令行参数 | 设置说明 |
| --- | --- | --- |
| `model_path` | `--model-path` | 完整模型目录，例如 `/path/to/model` |
| `dataset_path` | `--dataset-path` | 评测数据文件，例如 `/path/to/dataset/embspatial_bench.json` |
| `output_path` | `--output-path` | 逐题结果文件，例如 `/path/to/results/embspatial.jsonl` |
| `device` | `--device` | 使用的逻辑 NPU，默认 `npu:0` |
| `thinking` | `--enable-thinking` 或 `--disable-thinking` | 仓库配置为 `true`，即先推理后回答；`false` 表示直接回答 |
| `decode_backend` | `--decode-backend` | 默认 `graph`，也支持 `native`，见[解码后端说明](#graph-与-native) |

需要调整输出长度、采样温度等设置时，先查看[生成参数的含义](#推理参数)，再修改配置中的对应字段。`evaluation.json` 全部字段的类型、默认值和命令行写法见[评测配置文件说明](../../../docs/manipulation/hy-embodied-0.5-x/infer_with_torch/embspatial-bench-evaluation.md#评测配置文件)。显式命令行参数会覆盖 JSON 中对应的设置。

```bash
# --config 读取编辑后的配置，运行全部 3,640 题
python manipulation/hy-embodied-0.5-x/infer_with_torch/tools/evaluate_embspatial.py \
  --config manipulation/hy-embodied-0.5-x/infer_with_torch/configs/evaluation.json
```

仓库配置使用 BF16、thinking、temperature 0.05、top-p 1.0、基础 seed 42 和最多 32768 个输出 token；模型默认 top-k 为 50。每题使用基础 seed 加原始索引作为随机种子。错答、答案解析失败、内存不足和超时均计错，完整评测保留全部题目作为分母。

首次运行时，可以先检查前十题：

```bash
# --limit 10 只处理前十题；--output-path 将检查结果保存到单独的文件
python manipulation/hy-embodied-0.5-x/infer_with_torch/tools/evaluate_embspatial.py \
  --config manipulation/hy-embodied-0.5-x/infer_with_torch/configs/evaluation.json \
  --output-path /path/to/results/development.jsonl \
  --limit 10
```

### 查看评测结果

运行结束后，先打开 `embspatial.summary.json` 查看总分和完成数量。需要查看各类正确数、错误列表或内存统计时，按[本次评测汇总文件说明](../../../docs/manipulation/hy-embodied-0.5-x/infer_with_torch/embspatial-bench-evaluation.md#本次评测汇总文件)阅读对应字段。

- `total` 是本次选择的题目数，`completed` 是已保存结果的题目数，包含错答和运行异常。
- `correct` 是正确题数，`accuracy` 为 `correct / total × 100`，例如 `80.0` 表示 80%。
- `complete` 为 `true` 表示本次选择的题目均已有结果；`false` 表示仍有缺失。
- `fullDatasetRun` 为 `true` 表示选择了完整数据集，`false` 表示只选择部分题目。

例如，前十题全部完成后，`total` 和 `completed` 都是 `10`，`complete` 为 `true`，`fullDatasetRun` 为 `false`。

需要检查某道题时，在 `embspatial.jsonl` 中找到该题的 `question_id`，查看模型原文、解析出的选项和错误信息；各字段的含义见[逐题结果文件说明](../../../docs/manipulation/hy-embodied-0.5-x/infer_with_torch/embspatial-bench-evaluation.md#逐题结果文件)。这些字段由程序生成。

需要分析耗时或内存时，查看汇总中的 `performance` 对象。它保存首 token 耗时、端到端耗时、吞吐及内存峰值；单位和计算方式见[耗时和内存字段说明](../../../docs/manipulation/hy-embodied-0.5-x/infer_with_torch/embspatial-bench-evaluation.md#耗时和内存字段)。

### 中断后续跑

确认原评测进程已退出后，保留中断前的完整命令、配置内容和输出路径，仅补上 `--resume`。最初按完整配置运行时，续跑命令为：

```bash
# --resume 读取已有结果，核对配置后从下一道未完成题目继续
python manipulation/hy-embodied-0.5-x/infer_with_torch/tools/evaluate_embspatial.py \
  --config manipulation/hy-embodied-0.5-x/infer_with_torch/configs/evaluation.json \
  --resume
```

原命令额外指定的 `--output-path`、`--limit` 和生成参数也应保留。续跑使用原 `.jsonl` 和配套的 `.protocol.json`；后者记录首次运行的配置，具体内容见[运行配置记录文件说明](../../../docs/manipulation/hy-embodied-0.5-x/infer_with_torch/embspatial-bench-evaluation.md#运行配置记录文件)。程序核对配置后跳过已保存的题目，错误记录也会跳过；配置不一致时会列出具体字段差异。

本次选择的题目均已完成时，`--resume` 直接根据已有结果更新汇总文件，无需再次运行模型推理。若只有 SSH 连接中断，先确认原进程是否仍在运行。修改模型计算或答案解析逻辑后，使用新的输出路径开展评测。

### 重新统计分数并导出表格

评测结束后，使用 `score_results.py` 读取已有逐题结果，再计算一次准确率并生成表格。程序重新读取模型原文中的最终选项，与原数据的标准答案比较；这一步在 CPU 上使用 Python 标准库完成，无需加载模型。全部命令参数见[重新统计分数的命令参数](../../../docs/manipulation/hy-embodied-0.5-x/infer_with_torch/embspatial-bench-evaluation.md#重新统计分数的命令参数)。

```bash
# 读取已有结果和配套配置，用原始数据核对标准答案，输出到 summary/ 目录
python manipulation/hy-embodied-0.5-x/infer_with_torch/tools/score_results.py \
  --results-path /path/to/results/embspatial.jsonl \
  --protocol-path /path/to/results/embspatial.protocol.json \
  --dataset-path /path/to/dataset/embspatial_bench.json \
  --output-dir /path/to/results/summary
```

- `--results-path`：已有逐题结果文件。
- `--protocol-path`：与结果配套的运行配置，用于核对回答模式、题号顺序和解析规则。
- `--dataset-path`：原始评测数据，用于独立核对题号、空间关系类别和标准答案。
- `--output-dir`：保存 `summary.json` 和 `per-question.csv` 的目录。

命令完成后，打开 `summary/summary.json` 查看重新统计的总分及标准答案核对结果，各项含义见[重新统计后的汇总文件说明](../../../docs/manipulation/hy-embodied-0.5-x/infer_with_torch/embspatial-bench-evaluation.md#重新统计后的汇总文件)。这里的总准确率始终按“正确题数 / 3640 × 100%”计算；`complete` 只有在结果覆盖全部 3,640 题且六类题数完整时才为 `true`，少量题目的结果为 `false`。

需要逐题查看或筛选错答时，用表格软件打开 `summary/per-question.csv`。CSV 是以逗号分隔各列的文本表格，每行对应一道题；各列的名称、答案编号和空值含义见[逐题表格文件说明](../../../docs/manipulation/hy-embodied-0.5-x/infer_with_torch/embspatial-bench-evaluation.md#逐题表格文件)。

如果要把这一步放到另一台 CPU 机器执行，可以从原始数据导出一个较小的 JSON 文件，只保存数据版本、题号、类别、标准答案和题目顺序，再用它代替完整评测数据。需要这种用法时，按[导出并使用标准答案文件](../../../docs/manipulation/hy-embodied-0.5-x/infer_with_torch/embspatial-bench-evaluation.md#导出并使用标准答案文件)操作。

## 已有评测与性能数据

### 已有完整评测记录

以下结果对应使用 `torch.nn.GELU` 的 NPU 实现，覆盖 EmbSpatial-Bench human-verified evaluation 的全部 3,640 题。原始输出经过重新解析，并使用从 EmbSpatial-Bench 数据导出的标准答案文件核对。

| 空间关系 | 样本数 | 正确 | 普通错答 | 解析失败 | 达到生成上限 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| close | 612 | 460 | 152 | 0 | 0 | 75.1634% |
| far | 594 | 404 | 187 | 3 | 0 | 68.0135% |
| above | 596 | 518 | 74 | 0 | 4 | 86.9128% |
| under | 602 | 515 | 81 | 1 | 5 | 85.5482% |
| left | 616 | 574 | 40 | 0 | 2 | 93.1818% |
| right | 620 | 552 | 60 | 0 | 8 | 89.0323% |
| 合计 | 3,640 | 3,023 | 594 | 4 | 19 | 83.0495% |

普通错答指能够解析出合法选项、但与标准答案不同的输出。解析失败和达到生成上限的结果均计错，全部题目保留在分母中。总体 Accuracy 为 `3023 / 3640 × 100%`。检查已有结果的分数时，使用[重新统计分数并导出表格](#重新统计分数并导出表格)中的命令。

### 已有性能与内存记录

以下测量对应使用 `torch.nn.GELU` 的 NPU 实现，平台为 Atlas A3，模型精度为 BF16，单次推理样本数为 1。

| 指标 | 测量结果 | 测量范围 |
| --- | ---: | --- |
| 平均首 token 延迟 | 0.284255 秒 | 50 次独占设备完整生成，关闭性能分析工具和每步输出的 NaN/Inf 检查 |
| 平均端到端耗时 | 14.910801 秒 | 50 次独占设备完整生成，关闭性能分析工具和每步输出的 NaN/Inf 检查 |
| 输出吞吐 | 21.552275 token/s | 50 次独占设备完整生成，关闭性能分析工具和每步输出的 NaN/Inf 检查 |
| NPU 分配器峰值 | 10,141,768,192 字节 | 完整评测，单进程统计 |

需要查看延迟分布、CPU 内存和清理后内存时，阅读[已有性能测量数据](../../../docs/manipulation/hy-embodied-0.5-x/infer_with_torch/embspatial-bench-evaluation.md#已有性能测量数据)，其中说明了各数值的测量范围。

## 测试与故障排查

### CPU 逻辑测试

```bash
# 检查参数、输入、严格计分、结果文件和续跑逻辑
python -m unittest discover \
  -s manipulation/hy-embodied-0.5-x/infer_with_torch/tests \
  -v
```

### NPU 功能检查

```bash
# 检查逻辑 NPU 0 上的 Attention 和 GELU 适配
python manipulation/hy-embodied-0.5-x/infer_with_torch/bench/check_ops.py -v

# 使用指定模型与数据执行五项推理功能检查，结果写入新的输出目录
python manipulation/hy-embodied-0.5-x/infer_with_torch/bench/check_model.py \
  --model-path /path/to/model \
  --dataset-path /path/to/dataset/embspatial_bench.json \
  --output-dir /path/to/checks/smoke \
  --device npu:0 \
  --decode-backend graph
```

`--model-path` 和 `--dataset-path` 分别指定模型及数据，`--output-dir` 指定检查结果目录；`--device` 和 `--decode-backend` 选择执行设备及解码后端。需要了解某项检查在验证什么时，查看[测试用例说明](../../../docs/manipulation/hy-embodied-0.5-x/infer_with_torch/npu-adaptation.md#测试用例说明)。

修改解码实现后，可以[比较 graph 与 native 的输出](../../../docs/manipulation/hy-embodied-0.5-x/infer_with_torch/npu-adaptation.md#比较两种解码方式的输出)，检查相同输入的结果是否一致。排查连续处理图片后的状态或内存问题时，使用[连续推理检查方法](../../../docs/manipulation/hy-embodied-0.5-x/infer_with_torch/npu-adaptation.md#检查连续推理后的结果和内存)。

需要查看 NPU 实际执行了哪些算子、分配了多少内存时，按[记录 NPU 算子和内存使用](../../../docs/manipulation/hy-embodied-0.5-x/infer_with_torch/npu-adaptation.md#记录-npu-算子和内存使用)中的命令采集。

### 常见问题

| 现象 | 处理方式 |
| --- | --- |
| 图片路径错误或图片无法解码 | 按错误中的序号检查路径、读取权限和图片格式 |
| 依赖版本或模型接口不匹配 | 按本目录 `requirements.txt` 安装指定依赖 |
| NPU 不可用或加载运行库失败 | 执行 `npu-smi info`，加载当前 CANN 的 `set_env.sh`，并核对 torch_npu 配套版本 |
| `Resume configuration differs` | 对照列出的字段，使用首次配置继续原结果，或选择新输出路径 |

遇到文件权限、后端切换、结果文件损坏或标准答案不一致的错误时，按[常见错误处理](../../../docs/manipulation/hy-embodied-0.5-x/infer_with_torch/npu-adaptation.md#常见错误处理)中的对应条目排查。
