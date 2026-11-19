# EmbSpatial-Bench 评测参数与结果文件说明

运行评测、断点续跑和重新统计分数的步骤见[样例使用说明](../../../../manipulation/hy-embodied-0.5-x/infer_with_torch/README.md#embspatial-bench-评测)。本文按输入文件、配置文件和结果文件组织，说明每个文件由谁生成、何时使用，以及各字段的含义。

文中的命令均从仓库根目录执行，使用样例使用说明中准备的 Python 环境和 `PYTHONPATH`。`embspatial_bench.json` 包含图片、问题、选项和标准答案；`reference-answers.json` 是可选的较小文件，只保存数据版本、题号、类别、标准答案和题目顺序，用于在另一台机器重新统计分数。

## 文件存放位置

下图中的 `/path/to/` 表示用户选择的存放位置。`model/` 使用快速开始下载的完整模型目录；`results/` 及其中的结果文件由程序创建。

```text
仓库根目录/
└── manipulation/hy-embodied-0.5-x/infer_with_torch/configs/
    ├── evaluation.json                 # 用户编辑：评测路径与生成参数
    └── asset-lock.json                 # 随样例提供：模型、数据版本与文件信息

/path/to/
├── model/                              # 输入：完整模型权重、配置及文本/图像处理文件
├── dataset/
│   ├── embspatial_bench.json            # 输入：完整评测数据
│   └── reference-answers.json           # 可选输入：从完整数据导出的标准答案文件
└── results/
    ├── embspatial.jsonl                 # 评测生成：每行一道题的结果
    ├── embspatial.protocol.json         # 评测生成：本次运行配置
    ├── embspatial.summary.json          # 评测生成：本次运行的汇总
    └── summary/                        # 重新统计分数时由 --output-dir 指定
        ├── summary.json                # 重新统计分数生成：按完整评测题数统计的汇总
        └── per-question.csv            # 重新统计分数生成：逐题明细表
```

| 文件或目录 | 由谁准备 | 用来做什么 |
| --- | --- | --- |
| `model/` | 用户下载 | 保存模型权重及加载模型需要的配置文件。用 `model_path` 指向这个目录。 |
| `embspatial_bench.json` | 用户下载 | 保存全部评测题目及图片。用 `dataset_path` 指向这个文件。 |
| `evaluation.json` | 随样例提供，用户编辑 | 填写模型目录、数据文件和结果保存位置。评测命令通过 `--config` 读取它。 |
| `asset-lock.json` | 随样例提供 | 列出要求使用的模型和数据版本，以及检查文件是否正确所需的信息。由程序自动读取。 |
| `embspatial.jsonl` | 评测程序生成 | 每完成一道题就写入一行结果。中断后继续往这个文件追加。 |
| `embspatial.protocol.json` | 评测程序生成 | 保存第一次运行的设置。续跑时，程序用它检查本次设置是否一致。 |
| `embspatial.summary.json` | 评测程序生成 | 保存本次评测的总分、分类成绩和错误列表。 |
| `reference-answers.json` | 用户按本文命令导出，可选 | 只保存题号、标准答案等计分信息。用 `--reference-path` 指定，可代替包含图片的完整数据文件。 |
| `summary/summary.json` | `score_results.py` 生成 | 保存根据已有模型输出重新计算的成绩。 |
| `summary/per-question.csv` | `score_results.py` 生成 | 保存一行一道题的明细，可用表格软件查看或筛选错答。 |

结果配套文件按 `output_path` 替换扩展名得到。例如，`output_path` 为 `/path/to/results/embspatial.jsonl`，对应配置和汇总文件就是上图中的 `embspatial.protocol.json`、`embspatial.summary.json`。

`asset-lock.json` 的字段如下，评测程序自动使用这些信息检查下载的数据文件：

| 字段 | 说明 |
| --- | --- |
| `schemaVersion` | 这个 JSON 文件采用的格式版本，由样例提供。 |
| `model.repository` | 模型在 Hugging Face 上的仓库名称。 |
| `model.revision` | 要求使用的模型权重版本。 |
| `model.files` | 列出模型文件的名称，以及每个文件的内容校验值。 |
| `dataset.repository` | 评测数据在 Hugging Face 上的仓库名称。 |
| `dataset.revision` | 要求使用的评测数据版本。 |
| `dataset.file` | 评测数据的文件名，即 `embspatial_bench.json`。 |
| `dataset.bytes` | 正确数据文件应有的字节数。 |
| `dataset.sha256` | 正确数据文件的内容校验值。程序用它检查文件内容是否符合要求。 |

## 评测数据文件

文件名为 `embspatial_bench.json`，由用户从 EmbSpatial-Bench 数据仓库下载。

`embspatial_bench.json` 是一个包含 3,640 道题的 JSON 数组。评测直接读取其中的图片、问题和选项；图片已编码在文件中，运行评测时可以直接使用该 JSON 文件。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `question_id` | 字符串 | 这道题的唯一编号，用来对应题目和结果，例如 `mp3d_13`。 |
| `image` | 字符串 | 图片内容转换成的 Base64 文本。程序把它还原为图片后交给模型。 |
| `question` | 字符串 | 这道题的问题原文。 |
| `answer_options` | 包含四项的数组 | 四个选项的文字，按 A、B、C、D 排列。 |
| `answer` | 整数 | 标准答案编号：`0=A`、`1=B`、`2=C`、`3=D`。它只用于计分。 |
| `relation` | 字符串 | 这道题考查哪一种空间关系。可取 `close`、`far`、`above`、`under`、`left` 或 `right`。 |

六类题目数量分别为 close 612、far 594、above 596、under 602、left 616、right 620。程序将问题原文、四个选项和固定的最终答案要求组成输入，并按数据文件的原始顺序处理。

## 评测配置文件

文件为样例中的 `configs/evaluation.json`，用户在运行前编辑；评测程序通过 `--config` 读取。

编辑 `evaluation.json` 中的 `model_path`、`dataset_path`、`output_path`，使其对应实际位置。完整配置方式为：**显式命令行参数 > JSON 配置 > 程序默认值**。JSON 中的相对路径以命令执行时的仓库根目录为基准。

下表的“示例配置”指仓库提供的 `evaluation.json`；省略字段时才采用程序默认值。

| JSON 字段 | 命令行写法 | 类型 | 说明 |
| --- | --- | --- | --- |
| `model_path` | `--model-path` | 字符串 | 模型目录。必填，例如 `/path/to/model`。 |
| `dataset_path` | `--dataset-path` | 字符串 | 评测数据文件。必填，例如 `/path/to/dataset/embspatial_bench.json`。 |
| `output_path` | `--output-path` | 字符串 | 逐题结果保存到哪里。必填，例如 `/path/to/results/embspatial.jsonl`。 |
| `device` | `--device` | 字符串 | 用哪张逻辑 NPU。默认 `npu:0`。 |
| `dtype` | `--dtype` | 字符串 | 模型计算的数据类型。使用 `bfloat16` 或 `BF16`，两种写法都表示 BF16。 |
| `decode_backend` | `--decode-backend` | 字符串 | 选择逐步生成回答的执行方式：`graph` 使用 NPU 图执行，`native` 使用模型原有的计算路径。默认 `graph`。 |
| `model_revision` | `--model-revision` | 字符串 | 记录下载模型权重时使用的版本号。默认 `79f8301d8a80dd03ba778c58d6429e414e9ef5d3`。 |
| `dataset_revision` | `--dataset-revision` | 字符串 | 评测数据的版本号。默认 `e2733cacf0f20371518c687176e881d79644e61e`，程序会检查是否符合样例要求。 |
| `thinking` | `--enable-thinking` 或 `--disable-thinking` | 布尔值 | `true` 表示先推理后回答，`false` 表示直接回答。仓库配置填的是 `true`；没有提供此设置时，程序默认 `false`。 |
| `max_new_tokens` | `--max-new-tokens` | 正整数 | 每道题最多生成多少个新 token。默认 `32768`。 |
| `temperature` | `--temperature` | 非负数 | 控制采样的随机程度，默认 `0.05`。数值越小，越倾向于选择概率高的 token；设为 `0` 时固定选择概率最高的 token。 |
| `top_p` | `--top-p` | 数值 | 采样前，按概率从高到低保留累计概率达到该值的一组候选 token。默认 `1.0`；取值应大于 0、小于等于 1。 |
| `seed` | `--seed` | 非负整数 | 随机种子的起始值，默认 `42`。每题使用这个值加上题目在数据文件中的位置编号，位置从 0 开始；每题种子都需小于 `2**32`。 |
| `timeout_seconds` | `--timeout-seconds` | 正数 | 每道题允许生成的最长时间，单位秒。默认 `7200`。 |
| `limit` | `--limit` | 正整数 | 只处理数据文件开头的多少道题。省略时处理全部题目。 |
| `resume` | `--resume` | 布尔值 | 是否从原结果继续。默认 `false`；设为 `true` 后，程序核对设置并跳过已保存的题目。 |
| — | `--config` | 文件路径 | 从指定 JSON 文件读取评测设置。 |
| — | `--help` | 开关 | 显示命令帮助后退出。 |

## 逐题结果文件

文件名由 `output_path` 指定，例如 `embspatial.jsonl`。评测程序每完成一道题就写入一行 JSON。

JSONL 文件每行保存一个 JSON 对象，对应一道题。答案编号在结果中统一为 `0=A`、`1=B`、`2=C`、`3=D`；`null` 表示没有对应值。

| 字段 | 类型或单位 | 说明 |
| --- | --- | --- |
| `question_id` | 字符串 | 当前结果对应哪道题，与数据文件中的题号一致。 |
| `relation` | 字符串 | 这道题所属的空间关系类别，例如 `close`。 |
| `seed` | 整数 | 这道题实际使用的随机种子。 |
| `thinking` | 布尔值 | `true` 表示先推理后回答，`false` 表示直接回答。 |
| `ground_truth` | 整数 0～3 | 标准答案编号，`0=A`、`1=B`、`2=C`、`3=D`。 |
| `raw_output` | 字符串 | 模型生成的原文。重新统计分数时，程序从这段原文重新读取最终选项。 |
| `parsed_answer` | 整数 0～3 或 `null` | 从模型原文读出的选项编号。无法读出最终选项，或发生运行错误时，写为 `null`。 |
| `correct` | 布尔值 | 本题是否答对。正常结束且选项与标准答案相同才是 `true`；答错、解析失败和运行错误都是 `false`。 |
| `error` | 字符串或 `null` | 运行错误的名称或信息。没有运行错误时为 `null`；单纯答错也会是 `null`。 |
| `input_tokens` | 整数 | 模型处理后的图片和文本共占多少个输入 token。 |
| `output_tokens` | 整数 | 模型为这道题生成了多少个输出 token。 |
| `ttftSeconds` | 秒或 `null` | 从开始生成，到 CPU 收到第一个输出 token，经过多少秒。 |
| `generationSeconds` | 秒 | 模型生成回答用了多少秒，包括生成结束后等待 NPU 计算完成的时间。 |
| `endToEndSeconds` | 秒 | 处理本题输入、生成回答和把输出 token 转成文本，一共用了多少秒。 |
| `peakHostRssBytes` | 字节或 `null` | 处理这道题期间，采样到的进程常驻内存最大值。已加载模型等进程内存也包含在内。 |
| `peakNpuAllocatedBytes` | 字节或 `null` | 处理这道题期间，PyTorch NPU 分配器记录的最高已分配内存。 |
| `npuAllocatedAfterCleanupBytes` | 字节 | 释放本题输入输出后，NPU 仍在使用的已分配内存。模型此时仍然加载着。 |
| `npuReservedAfterCleanupBytes` | 字节 | 清理本题后，NPU 分配器仍保留的总内存，包含可重复使用的缓存。 |
| `hostRssAfterCleanupBytes` | 字节 | 清理本题后，进程仍占用的常驻内存。 |

若异常发生在耗时和内存统计或清理之前，部分统计字段可能缺省或为 `null`。`GenerationTimeout` 表示生成超时，`GenerationTokenLimit` 表示达到输出上限且尚未生成结束标记。`error` 为 `null`、但 `parsed_answer` 为 `null` 的题目属于答案解析失败；汇总时以 `UnparseableAnswer` 标记。可以解析但选项错误的题目记为正常生成但答错。

## 运行配置记录文件

文件名与逐题结果对应，例如 `embspatial.protocol.json`，由评测程序自动生成。

该文件自动生成，与逐题结果配套保存。续跑时程序读取其中的配置，并列出与当前设置不一致的字段。

| 字段 | 说明 |
| --- | --- |
| `schemaVersion` | 这个运行配置 JSON 采用的格式版本，程序自动填写。 |
| `run_config` | 保存本次实际使用的模型、数据、生成参数和题目顺序。 |
| `configHash` | 运行设置的校验值。程序用它检查已保存的设置是否被改动。 |
| `provenance` | 程序自动填写的运行环境信息。 |

`run_config` 的字段如下。命令行与 JSON 的取值合并后，程序将实际设置写入此对象。

| `run_config` 内的字段 | 说明 |
| --- | --- |
| `mode` | `generation` 表示让模型生成回答，再从回答中读取选项。 |
| `implementation` | 生成这些结果的评测程序标识，程序自动填写。 |
| `modelPath` | 本次加载的模型目录，保存为绝对路径。 |
| `datasetPath` | 本次读取的数据文件，保存为绝对路径。 |
| `dtype` | 本次模型计算的数据类型，本样例为 BF16。 |
| `device` | 本次使用的逻辑 NPU，例如 `npu:0`。 |
| `decodeBackend` | 生成回答时使用的执行方式，`graph` 或 `native`。 |
| `assets.model` | 模型仓库名称，即 `tencent/HY-Embodied-0.5-X`。 |
| `assets.modelRevision` | 本次记录的模型权重版本。 |
| `assets.dataset` | 评测数据仓库名称，即 `Phineas476/EmbSpatial-Bench`。 |
| `assets.datasetRevision` | 本次读取的评测数据版本。 |
| `assets.datasetSha256` | 本次数据文件的内容校验值，用于确认读取的是要求使用的文件。 |
| `upstream.hy` | 腾讯混元模型 Python 依赖的源码版本。 |
| `upstream.transformers` | Transformers 依赖的源码版本。 |
| `thinking` | `true` 表示先推理后回答，`false` 表示直接回答。 |
| `maxNewTokens` | 本次为每题设置的最大输出 token 数。 |
| `temperature` | 本次使用的采样温度。设为 `0` 时固定选择概率最高的 token。 |
| `topP` | 本次采样时使用的累计概率阈值，对应配置里的 `top_p`。 |
| `topK` | 控制采样时保留多少个高概率候选 token，本样例为 `50`。 |
| `seed` | 随机种子的起始值。 |
| `seedPolicy` | 记录每题种子的计算方法：起始值加题目在原始数据中的位置编号，位置从 0 开始。 |
| `timeoutSeconds` | 为每题设置的生成超时时间，单位秒。 |
| `answerParser` | 把模型回答转成选项时使用的解析规则版本。 |
| `fullDatasetRun` | `true` 表示选择了全部 3,640 题，`false` 表示只选择了部分题目。它描述选择了多少题，不表示这些题已经完成。 |
| `selectedIds` | 本次要处理的题号列表，按程序处理的先后顺序保存。 |
| `prompt` | 记录输入问题的组成方式：问题原文、四个选项，再加要求输出最终选项的固定文字。 |
| `imagePreprocessing` | 记录把图片和问题转成模型输入时使用的方法：腾讯混元 AutoProcessor 和模型目录中的对话模板。 |

## 本次评测汇总文件

文件名与逐题结果对应，例如 `embspatial.summary.json`，汇总本次选择的题目。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `total` | 整数 | 本次要处理多少道题。完整评测为 `3640`，设置 `--limit 10` 时为 `10`。 |
| `completed` | 整数 | 已经保存了多少道题的结果。答错、解析失败和运行错误都计入这个数量。 |
| `correct` | 整数 | 已经答对多少道题。发生运行错误的题目不计入正确数。 |
| `accuracy` | 数值 | 正确题数除以本次题目总数，再乘 100。例如 10 题答对 8 题，这里是 `80.0`，表示 80%。 |
| `complete` | 布尔值 | 本次要处理的题目是否都已有结果。有结果的题数等于 `total` 时为 `true`，仍有缺失时为 `false`。 |
| `fullDatasetRun` | 布尔值 | 本次是否选择了全部 3,640 题。选择全部题目为 `true`，只选部分题目为 `false`。 |
| `relations` | 对象 | 分别保存六类题目的成绩。例如，`relations.close` 保存 close 类题目的成绩。 |
| `errors` | 数组 | 列出发生运行错误或无法解析答案的题目。每项保存题号 `question_id` 和错误说明 `error`；正常生成但答错的题目不会进入这个列表。 |
| `missingIds` | 数组 | 列出本次还没有结果的题号。`[]` 表示每道题都已经有结果。 |
| `performance` | 对象 | 保存本次的耗时、每秒输出 token 数和内存峰值。每个字段的单位及计算方法见本文件的“耗时和内存字段”。 |

每种空间关系的成绩都包含以下三个字段：

| 字段 | 说明 |
| --- | --- |
| `total` | 本次选中的这类题目有多少道，包括尚未得到结果的题目。 |
| `correct` | 这类题目已经答对多少道。运行错误的题目不算答对。 |
| `accuracy` | 这类题目的正确数除以这类题目的总数，再乘 100。 |

下面仅演示文件格式：假设本次选中了四道 close 类题目，其中三道答对，`relations` 中对应部分会写成：

```json
{
  "close": {
    "total": 4,
    "correct": 3,
    "accuracy": 75.0
  }
}
```

要读取示例中的题目总数，先进入 `relations`，再进入 `close`，最后读取 `total`。这里的 `75.0` 表示这四道 close 类题目有 75% 答对。

例如，前十题全部完成后，此文件的 `total`、`completed` 均为 `10`，`complete` 为 `true`，`fullDatasetRun` 为 `false`。这表示十题检查已完成；完整评测需选择并完成全部 3,640 题。完成数量包含错误记录，准确率由 `correct` 单独计算。

程序也会在终端逐题输出进度对象：`completed` 和 `total` 表示已完成数与本次题目数，`question_id`、`parsed_answer`、`error` 分别表示刚完成题目的题号、解析答案和异常信息。运行结束后输出汇总对象。

## 耗时和内存字段

评测程序与分数统计脚本生成的汇总文件使用同一组性能字段。时间单位为秒，内存单位为字节；没有可用测量值时使用 `null`。

| 字段 | 说明 |
| --- | --- |
| `ttftSeconds` | 汇总每道题等待第一个输出 token 的时间，保存测量题数、平均值、中位数和 95 分位值。 |
| `endToEndSeconds` | 汇总每道题从处理输入到获得回答的时间，保存测量题数、平均值、中位数和 95 分位值。 |
| `outputTokensPerGenerationSecond` | 每秒生成多少个输出 token。用参与统计题目的输出 token 总数，除以它们的生成耗时总和。 |
| `generationThroughputPopulation` | 说明哪些题目被用于计算每秒输出 token 数。当前只使用没有运行错误、且生成耗时大于 0 的题目。 |
| `peakHostRssBytes` | 所有已保存题目中，记录到的最大进程常驻内存，单位字节。 |
| `peakNpuAllocatedBytes` | 所有已保存题目中，记录到的最大 NPU 已分配内存，单位字节。 |

| 耗时统计对象中的字段 | 说明 |
| --- | --- |
| `count` | 有这项耗时记录的题目数。 |
| `mean` | 平均耗时，单位秒。把这些题目的耗时相加，再除以 `count`。 |
| `p50` | 耗时从小到大排列后的中间值，即中位数，单位秒。 |
| `p95` | 把耗时从小到大排列后，取接近 95% 位置的值，单位秒。位置落在两条记录之间时，按两条耗时的差值和位置比例计算。 |

耗时统计纳入具有相应测量值的记录；吞吐另按表中的成功记录范围统计。准确率使用所声明的全部题目作为分母，运行错误仍计错。

## 重新统计分数的命令参数

这些参数属于 `tools/score_results.py`。该脚本读取已有模型输出、重新计算正确数量，并保存汇总和逐题表格。

推荐同时提供原评测数据，核对结果中的题号、空间关系类别和标准答案：

```bash
python manipulation/hy-embodied-0.5-x/infer_with_torch/tools/score_results.py \
  --results-path /path/to/results/embspatial.jsonl \
  --protocol-path /path/to/results/embspatial.protocol.json \
  --dataset-path /path/to/dataset/embspatial_bench.json \
  --output-dir /path/to/results/summary
```

| 参数 | 是否必填 | 说明 |
| --- | --- | --- |
| `--results-path` | 是 | 要重新统计分数的逐题结果文件，例如 `embspatial.jsonl`。 |
| `--output-dir` | 是 | 新的汇总和表格保存到哪个目录。程序在这里写入 `summary.json` 和 `per-question.csv`。 |
| `--protocol-path` | 否，推荐提供 | 第一次评测生成的运行配置文件。提供后，程序检查回答模式、题号顺序和答案解析规则是否一致。 |
| `--dataset-path` | 否，推荐提供 | 原始 `embspatial_bench.json`。程序从这里读取标准答案，核对结果文件中的题号、类别和答案。 |
| `--reference-path` | 否 | 从原始数据导出的 `reference-answers.json`。它可代替 `--dataset-path`，两个参数选择一个。 |
| `--help` | 否 | 显示命令帮助后退出。 |

只提供结果文件时，重新统计分数使用文件内的标准答案。提供 `--dataset-path` 或 `--reference-path` 时，重新统计分数先与原始数据或标准答案文件核对；题号、类别或标准答案不一致会报错。

## 导出并使用标准答案文件

如果要在另一台 CPU 机器重新统计分数，可以导出一个只含计分信息的 JSON 文件，避免复制包含图片的完整数据。生成的 `reference-answers.json` 保存全部 3,640 道题的题号、类别、标准答案和原始顺序，并记录数据版本：

```bash
python - <<'PY'
import json
from pathlib import Path
from evaluation.protocol import load_dataset

rows, info = load_dataset("/path/to/dataset/embspatial_bench.json")
reference = {
    "datasetRevision": info["datasetRevision"],
    "questions": [
        {
            "question_id": row["question_id"],
            "relation": row["relation"],
            "ground_truth": row["answer"],
            "originalIndex": index,
        }
        for index, row in enumerate(rows)
    ],
}
Path("/path/to/dataset/reference-answers.json").write_text(
    json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8"
)
PY
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `datasetRevision` | 字符串 | 导出标准答案时使用的数据版本。 |
| `questions` | 数组 | 保存全部 3,640 道题的标准答案记录，顺序与原数据文件一致。 |
| `questions[].question_id` | 字符串 | 当前标准答案属于哪道题。 |
| `questions[].relation` | 字符串 | 这道题的空间关系类别，例如 `close`。 |
| `questions[].ground_truth` | 整数 0～3 | 这道题的标准答案编号，`0=A`、`1=B`、`2=C`、`3=D`。 |
| `questions[].originalIndex` | 整数 | 这道题在原数据文件中排第几个，从 `0` 开始，到 `3639` 结束。 |

使用这个标准答案文件时，提供 `--reference-path`，完整命令如下：

```bash
python manipulation/hy-embodied-0.5-x/infer_with_torch/tools/score_results.py \
  --results-path /path/to/results/embspatial.jsonl \
  --protocol-path /path/to/results/embspatial.protocol.json \
  --reference-path /path/to/dataset/reference-answers.json \
  --output-dir /path/to/results/summary
```

## 重新统计后的汇总文件

文件为 `--output-dir` 目录中的 `summary.json`，由 `score_results.py` 生成。

重新统计分数的总准确率始终使用全部 3,640 题作为分母，缺失题目保留在分母中。此文件与评测程序生成的 `embspatial.summary.json` 分别保存。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `total` | 整数 | 完整数据集的题目总数，固定为 `3640`。只提供部分结果时，这个值仍是 `3640`。 |
| `completed` | 整数 | 结果文件中已经有结果的题目数。答错、解析失败和运行错误都算已有结果。 |
| `correct` | 整数 | 从模型原文重新读取选项后，确认答对的题目数。 |
| `accuracy` | 数值 | 正确题数除以 `3640`，再乘 100，结果表示百分比。还没有结果的题目也包含在这 3,640 题中。 |
| `complete` | 布尔值 | 结果是否已经包含全部 3,640 道题。题号完整、六类数量也正确时为 `true`，否则为 `false`。 |
| `relations` | 对象 | 分别保存六类题目的成绩。例如，`relations.close` 保存 close 类题目的结果及错误数量。 |
| `errors` | 数组 | 运行错误和答案解析失败的题目列表。每项保存题号 `question_id`、类别 `relation` 和错误信息 `error`。 |
| `answerParser` | 字符串 | 本次从模型回答中读取选项时，使用的解析规则版本。 |
| `thinking` | 布尔值 | 按哪一种回答格式读取选项：`true` 为先推理后回答，`false` 为直接回答。一份结果文件只能使用一种模式。 |
| `labelSource` | 字符串 | 计分时的标准答案从哪里读取。`reference` 表示原始数据或单独的标准答案文件；`result` 表示逐题结果文件。 |
| `labelsCheckedAgainstReference` | 布尔值 | 是否用原始数据或标准答案文件核对过结果记录。核对通过时为 `true`；只使用结果自带的标准答案时为 `false`。 |
| `labelVerification` | 字符串 | 用文字说明标准答案从哪里读取。`independent compact reference` 表示已使用单独提供的数据核对；`embedded result labels only` 表示只使用结果文件内的标准答案。 |
| `protocolVerified` | 布尔值 | 是否提供并检查了第一次评测的运行配置文件。提供 `--protocol-path` 且检查通过时为 `true`，没有提供时为 `false`。 |
| `protocolValidationFields` | 数组 | 列出程序支持检查的设置名称，例如回答模式和题号顺序。这是项目清单，不表示每个项目都已实际检查。 |
| `performance` | 对象 | 使用逐题结果中已有的耗时和内存数值重新生成统计。它不会再次测量模型。 |

提供原始数据或标准答案文件后，程序会核对结果记录中的题号、空间关系类别和标准答案。例如，结果内标准答案为 D，数据集中的答案也是 D，核对通过；模型若回答 C，这道题仍计错。完整覆盖和标准答案核对结果分别由 `complete`、`labelsCheckedAgainstReference` 表示，正确数量由 `correct` 表示。

`protocolValidationFields` 中的项目名称对应如下：

| 项目名称 | 何时检查、检查什么 |
| --- | --- |
| `answerParser` | 提供 `--protocol-path` 时，检查结果采用的答案解析规则是否受当前程序支持。 |
| `mode` | 提供 `--protocol-path` 时，检查第一次评测使用的是生成文本后读取选项的方式。 |
| `thinking` | 检查各题的回答模式是否一致。提供 `--protocol-path` 时，还会检查与配置文件是否一致。 |
| `selectedIds` | 提供 `--protocol-path` 时，检查已保存题目是否按第一次选择的题号顺序排列。 |
| `datasetRevision with reference` | 同时提供运行配置文件和标准答案来源时，检查两者记录的数据版本是否一致。 |
| `seed with reference` | 同时提供运行配置文件和标准答案来源时，检查每题种子是否等于基础种子加原始位置编号。 |

每种空间关系都有下面七个字段。这里按完整数据集统计：例如，close 类共有 612 题，即使只提供了其中几题的结果，close 类的总题数仍为 612。

| 每类成绩中的字段 | 说明 |
| --- | --- |
| `total` | 完整数据集中，这类题目共有多少道。close 为 612、far 为 594、above 为 596、under 为 602、left 为 616、right 为 620。 |
| `completed` | 结果文件中已经保存了多少道这类题目的结果。 |
| `correct` | 这类题目中答对了多少道。 |
| `accuracy` | 这类题目的正确数除以这类题目的完整总数，再乘 100。close 类始终除以 612，其他类别使用各自的完整总数。 |
| `answerMismatch` | 成功读出最终选项，但选项答错的题目数。 |
| `unparseable` | 没有运行错误，但从回答中读不出合法最终选项的题目数。 |
| `executionErrors` | 发生运行错误的题目数，包括内存不足、超时和达到生成上限后仍未结束等情况。 |

对只含前十题的结果进行重新统计分数时，本文件的 `completed` 为 `10`，`total` 为 `3640`，`complete` 为 `false`。这与十题评测自身汇总中的 `complete: true` 分别描述“完整数据覆盖情况”和“本次所选题目完成情况”。

## 逐题表格文件

文件为 `--output-dir` 目录中的 `per-question.csv`，每行对应一道题，可用表格软件打开。

CSV 使用 UTF-8 编码并带 BOM，可用于表格软件查看。其 `parsed_answer`、`correct` 由重新统计分数程序重新计算；其余列从逐题结果读取。空值在 CSV 中留空，布尔值写为 `True` 或 `False`。

| CSV 列名 | 说明 |
| --- | --- |
| `question_id` | 这一行对应哪道题。 |
| `relation` | 这道题的空间关系类别，例如 `close`。 |
| `ground_truth` | 标准答案编号，`0=A`、`1=B`、`2=C`、`3=D`。 |
| `parsed_answer` | 重新从模型原文读出的选项编号。读不出合法选项时留空。 |
| `correct` | 本题是否答对，写为 `True` 或 `False`。运行错误的题目写为 `False`。 |
| `error` | 原结果记录的运行错误信息。没有运行错误时留空；答案解析失败会表现为 `parsed_answer` 留空，并在汇总中标记。 |
| `seed` | 这道题实际使用的随机种子。 |
| `input_tokens` | 模型处理后的图片和文本共占多少个输入 token。 |
| `output_tokens` | 模型为这道题生成了多少个输出 token。 |
| `ttftSeconds` | 生成开始后，等到第一个输出 token 用了多少秒。 |
| `generationSeconds` | 模型生成回答用了多少秒。 |
| `endToEndSeconds` | 处理本题输入、生成回答和转成文本一共用了多少秒。 |
| `peakHostRssBytes` | 处理本题时，记录到的最大进程常驻内存，单位字节。 |
| `peakNpuAllocatedBytes` | 处理本题时，记录到的最大 NPU 已分配内存，单位字节。 |

## 已有性能测量数据

以下测量对应使用 `torch.nn.GELU` 的 NPU 实现，平台为 Atlas A3，模型精度为 BF16，单次推理样本数为 1。

延迟和吞吐采用 50 次独占设备、关闭 profiler 与逐步有限值检查的完整生成结果：

| 指标 | 平均值 | P50 | P95 |
| --- | ---: | ---: | ---: |
| 首 token 延迟（秒） | 0.284255 | 0.284051 | 0.288930 |
| 端到端耗时（秒） | 14.910801 | 12.468212 | 37.239775 |

| 指标 | 测量结果 | 测量范围 |
| --- | ---: | --- |
| 输出吞吐 | 21.552275 token/s | 上述 50 次完整生成 |
| NPU 分配器峰值 | 10,141,768,192 字节 | 完整评测，单进程统计 |
| CPU 侧进程常驻内存（RSS）峰值 | 7,522,078,720 字节 | 模型加载阶段 |
| 清理后活跃 NPU 内存 | 7,582,910,976 字节 | 20 题重复两轮的结束值 |

## 从评测数据中提取单个样本

下面使用 [Phineas476/EmbSpatial-Bench](https://huggingface.co/datasets/Phineas476/EmbSpatial-Bench) 中的 `mp3d_13`：一张 640×480 的室内图片，问题询问抽屉柜与镜子的相对位置。

先从已下载的数据文件中提取图片和完整问题，保存到示例目录：

```bash
python - <<'PY'
import base64
import json
from pathlib import Path
from evaluation.protocol import prompt

dataset = Path("/path/to/dataset/embspatial_bench.json")
rows = json.loads(dataset.read_text(encoding="utf-8"))
row = next(item for item in rows if item["question_id"] == "mp3d_13")
folder = Path("/path/to/example")
folder.mkdir(parents=True, exist_ok=True)
(folder / "mp3d_13.jpg").write_bytes(base64.b64decode(row["image"]))
(folder / "prompt.txt").write_text(prompt(row), encoding="utf-8")
PY
```

`prompt.txt` 的内容为：

```text
What is the spatial configuration between dresser and mirror in relation to each other within the image?
A. The dresser is touching the mirror.
B. The dresser is above the mirror.
C. The dresser is blocking the mirror.
D. The dresser is below the mirror.
Select the best answer. Put only one letter (A, B, C or D) in the final answer.
```

使用相同的图片、问题和生成参数执行推理：

```bash
python manipulation/hy-embodied-0.5-x/infer_with_torch/src/infer.py \
  --model-path /path/to/model \
  --image /path/to/example/mp3d_13.jpg \
  --prompt "$(cat /path/to/example/prompt.txt)" \
  --device npu:0 \
  --dtype bfloat16 \
  --enable-thinking \
  --max-new-tokens 32768 \
  --temperature 0.05 \
  --top-p 1.0 \
  --seed 55
```

命令按 JSON 行输出结果，`output` 字段保存生成文本。以下为该样本一次 thinking 模式运行的 `output` 内容；`<think>` 起始标记由腾讯混元输入模板提供：

```text
Got it, let's see. The mirror is on the wall above the dresser, right? So the dresser is
below the mirror. Let's check the options: D says that. Yeah, that makes sense. The mirror
is mounted higher on the wall, the dresser is under it, so they are vertically aligned,
dresser below mirror.
</think>
<answer>
D
</answer>
```

该示例的最终答案为 `D`，对应“抽屉柜在镜子下方”。
