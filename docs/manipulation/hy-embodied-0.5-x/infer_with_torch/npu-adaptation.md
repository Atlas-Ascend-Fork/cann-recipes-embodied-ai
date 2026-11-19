# HY-Embodied-0.5-X NPU 适配与验证说明

本文说明样例怎样复用腾讯混元模型接口，以及每项 NPU 适配的边界。安装和操作顺序见[样例使用说明](../../../../manipulation/hy-embodied-0.5-x/infer_with_torch/README.md)。

本文中的命令均从仓库根目录执行，沿用样例使用说明中配置的 Python 环境和 `PYTHONPATH`。需要查询评测配置或结果文件中的字段时，查看[EmbSpatial-Bench 评测参数与结果文件说明](embspatial-bench-evaluation.md)。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| [src/infer.py](../../../../manipulation/hy-embodied-0.5-x/infer_with_torch/src/infer.py) | 推理参数、输入检查、单图与串行批量命令 |
| [src/model_loader.py](../../../../manipulation/hy-embodied-0.5-x/infer_with_torch/src/model_loader.py) | 检查依赖版本和必要接口，加载图像/文本处理器及 BF16 模型，并安装 NPU 适配。 |
| [src/command_io.py](../../../../manipulation/hy-embodied-0.5-x/infer_with_torch/src/command_io.py) | JSON 默认参数读取与 JSON 标准输出 |
| [src/adapters/transformers_patch.py](../../../../manipulation/hy-embodied-0.5-x/infer_with_torch/src/adapters/transformers_patch.py) | 安装和备份 Transformers 兼容补丁 |
| [src/adapters/attention.py](../../../../manipulation/hy-embodied-0.5-x/infer_with_torch/src/adapters/attention.py) | 把模型的注意力调用连接到 NPU，并处理输入排列、因果遮挡和查询头与键值头数量不同的情况。 |
| [src/adapters/gelu.py](../../../../manipulation/hy-embodied-0.5-x/infer_with_torch/src/adapters/gelu.py) | 模型实例级 erf GELU 适配 |
| [src/adapters/graph_decode.py](../../../../manipulation/hy-embodied-0.5-x/infer_with_torch/src/adapters/graph_decode.py) | 记录并重复执行解码计算，缓存文本/视觉分支判断，并在模型关闭时释放相应资源。 |
| [src/inference/generation.py](../../../../manipulation/hy-embodied-0.5-x/infer_with_torch/src/inference/generation.py) | 把图片和问题转成模型输入，设置随机种子，调用生成接口并将输出转成文本。 |
| [src/inference/monitoring.py](../../../../manipulation/hy-embodied-0.5-x/infer_with_torch/src/inference/monitoring.py) | 按需采集耗时与内存并汇总计时记录 |
| [src/evaluation/evaluator.py](../../../../manipulation/hy-embodied-0.5-x/infer_with_torch/src/evaluation/evaluator.py) | 题目顺序、逐题生成、异常计错及续跑 |
| [src/evaluation/protocol.py](../../../../manipulation/hy-embodied-0.5-x/infer_with_torch/src/evaluation/protocol.py) | 数据格式、问题构造、严格解析和计分 |
| [src/evaluation/results.py](../../../../manipulation/hy-embodied-0.5-x/infer_with_torch/src/evaluation/results.py) | 保存逐题结果、检查续跑设置，并根据模型原文重新计分和生成表格。 |
| [tools/evaluate_embspatial.py](../../../../manipulation/hy-embodied-0.5-x/infer_with_torch/tools/evaluate_embspatial.py) | 评测命令入口 |
| [tools/score_results.py](../../../../manipulation/hy-embodied-0.5-x/infer_with_torch/tools/score_results.py) | CPU 重新统计分数命令入口 |

模型加载、NPU 适配和推理模块协作完成核心推理。评测模块调用核心路径，标准答案保留在评测与重新统计分数层。

## 依赖检查与模型加载

`requirements.txt` 指定腾讯混元 HY-Embodied-0.5-X 和 Transformers 的源码版本；Transformers 的包内版本字符串为 `4.57.0`。加载器检查依赖版本，以及模型、文本/视觉分支选择函数、旋转位置编码函数和 `HyEmbodiedPipeline` 需要的接口。

模型加载使用本地 checkpoint，保留 processor、chat template、BF16 和 top-k 50。参数放到选定 NPU；checkpoint 键、形状或设备不匹配时报告错误。原生成参数、Python/NumPy/PyTorch/NPU seed 初始化和输出解码参数由 inference/generation.py 共用。

## 注意力与 GELU 适配

注意力计算根据查询（Q）、键（K）和值（V）决定当前 token 如何使用上下文。组查询注意力（GQA）让多个查询头共享一组键和值。表中的 BSND 表示批次、序列长度、注意力头数和每头维度；TND 表示将多条序列拼接后按 token、头数和每头维度存放。

| 调用行为 | 适配 |
| --- | --- |
| 固定长度或不同长度的序列 | 分别接收 BSND 或 TND 形式的输入。多条序列拼接时，按每条序列的起止位置分别计算。 |
| 多个查询头共享一组键和值 | 按查询头与键值头的数量比例，把每个查询头对应到它使用的键和值。 |
| 使用因果遮挡，且查询序列与键序列长度不同 | 先把查询对齐到键序列末尾，再屏蔽每个查询之后的位置。 |
| 某个查询看不到任何键 | 该查询对应的输出全部为 0。 |
| 文本和视觉位置 | 文本只能使用当前及更早的位置，视觉可使用同一序列中的全部位置；具体哪些位置属于文本或视觉，由腾讯混元模型决定。 |

Attention 数值计算和 Q/K/V 在 NPU，CPU 仅读取整数分段边界。当前 torch_npu 2.10.0.post6、CANN 9.1.0 环境下，none GELU 通过实例级 torch_npu.npu_gelu(approximate="none") 执行 erf 公式。完整模型预期替换 29 处，保留形状、dtype、参数与显式 tanh 模块。

## 解码方式与资源释放

graph 为默认后端，使用单 token 解码的专用分支、模态判断缓存，以及归一化、投影和前馈网络（MLP）的图捕获与重放。Attention 和 KV Cache 更新在图区域外执行。图包装检查单样本、纯文本 token、线程、NPU 执行流、数据类型和张量形状等条件；其他输入使用模型原有的 NPU 层实现。

native 安装必要 Attention 和 GELU 适配，调用腾讯混元模型的 forward 方法与 KV Cache 更新。它关闭自定义模态判断缓存、图包装及单 token 解码的专用分支。

每个进程在导入 HY 模型前选择后端。graph 模式一次管理一个 pipeline；close_pipeline() 关闭其持有的包装。后端选择进入评测 run_config，防止续跑混合不同执行路径。

## Transformers 文件补丁与恢复

加载器修改独立环境的 transformers/models/hunyuan_vl_mot/modeling_hunyuan_vl_mot.py，将 FlashAttention 导入连接至本样例的 NPU 接口；graph 后端还应用单 token 解码的专用分支。原文件保存为 modeling_hunyuan_vl_mot.py.hy-original。补丁逐处核对预期源码，重复使用相同后端保持幂等；版本冲突和写权限问题明确报错。

模型加载后的后端切换使用新的进程。并发使用不同后端时，为各进程准备独立环境。运行期 close_pipeline() 恢复包装；结束使用该环境的推理进程后，可从备份恢复磁盘源码：

```python
from pathlib import Path
import transformers

target = Path(transformers.__file__).parent / "models/hunyuan_vl_mot/modeling_hunyuan_vl_mot.py"
backup = target.with_suffix(".py.hy-original")
target.write_bytes(backup.read_bytes())
```

## 评测配置检查与运行统计

`run_config` 保存模型和数据路径、版本、原始题目顺序、seed 策略、生成参数、解码后端、prompt 和解析规则，用于续跑时核对配置。修改模型计算或解析逻辑后，使用新的输出目录开展评测。

每题完整保存并 fsync 后进入下一题。已完成 ID 必须是原顺序的唯一前缀，seed 与原始位置编号一致。解析失败、OOM、超时与长度截断计错；标准答案只用于计分。重新统计分数时，可以使用结果文件中的标准答案，也可以用原始数据核对这些答案。

| 指标 | 测量范围 |
| --- | --- |
| 首 token 延迟（TTFT） | 从开始生成到 CPU 收到第一个输出 token，经过多少秒。 |
| 生成耗时 | 从开始生成到生成结束、NPU 计算完成，经过多少秒。 |
| 端到端耗时 | 处理图片和文本输入、生成回答，再把输出转成文本，一共经过多少秒。 |
| 进程常驻内存（Host RSS） | 进程实际占用的常驻内存，每 50 毫秒读取一次，单位字节。 |
| NPU 内存峰值 | 本题运行期间，PyTorch NPU 分配器记录的最高已分配字节数。 |
| 清理后内存 | 释放本题输入输出后读取的内存数值；此时模型仍保持加载。 |

耗时和内存统计由评测程序启用。吞吐使用没有运行错误且生成耗时大于 0 的记录计算，准确率仍保留全部题目分母。缺少内存测量时使用 `null`。

## 测试用例说明

CPU 测试位于 tests/，NPU 检查位于 bench/ 并显式执行。下表给出每个函数的输入范围和判定目的；完整模型的 Accuracy 另由全量评测记录计算。

### [tests/test_protocol.py](../../../../manipulation/hy-embodied-0.5-x/infer_with_torch/tests/test_protocol.py)

| 函数 | 验证目的与通过条件 |
| --- | --- |
| `test_prompt_has_no_label_or_bbox` | 检查发给模型的内容只包含图片、问题和选项，不带标准答案或对象坐标。 |
| `test_thinking_does_not_scan_reasoning` | 回答的推理过程即使出现选项字母，也只读取推理结束后的最终答案。 |
| `test_ambiguous_answer_is_failure` | 回答里出现多个可能的最终选项时，应判为解析失败。 |
| `test_direct_answer_wrapper` | 直接回答模式下，检查带有规定答案标记的输出能否正确读出选项。 |
| `test_final_line_after_closed_thinking` | 回答包含完整的推理结束标记时，应能从最后一行读取最终选项。 |
| `test_failed_missing_count_against_total` | 部分题目发生错误或没有结果时，总题数保持完整，准确率的分母不减少。 |
| `test_repeated_ids_rejected` | 题目列表中出现重复题号时，应报错。 |
| `test_sft_rejected` | 把训练用数据当作评测输入时，应报错。 |
| `test_all_options` | 分别检查 A、B、C、D 能否转换成答案编号 0、1、2、3。 |
| `test_reasoning_answer_is_not_scored` | 回答只有推理过程、没有合法最终答案时，不能计为答对。 |
| `test_truncated_wrapper` | 推理或答案的标记没有闭合时，应判为解析失败。 |
| `test_error_never_counts_correct` | 发生运行错误的题目，即使输出里有正确字母，也必须计错。 |
| `test_independent_parser_does_not_trust_record` | 故意修改已保存的解析答案；重新统计时应从模型原文发现不一致并报错。 |
| `test_changed_ground_truth_rejected` | 结果文件中的标准答案与数据集不同，应报错。 |
| `test_duplicate_results_rejected` | 结果文件中出现重复题号时，重新统计分数应报错。 |

### [tests/test_runtime.py](../../../../manipulation/hy-embodied-0.5-x/infer_with_torch/tests/test_runtime.py)

| 函数 | 验证目的与通过条件 |
| --- | --- |
| `test_core_inference_imports_without_evaluation_or_model_stack` | 导入核心推理模块时，不应连带导入评测模块或加载模型运行库。 |
| `test_config_and_cli_precedence` | 同一个参数同时写在 JSON 和命令行中时，应使用命令行指定的值。 |
| `test_invalid_configuration_rejected` | 配置字段拼错、图片与问题数量不匹配或生成参数非法时，应在加载模型前报错。 |
| `test_generation_parameter_boundaries` | 检查温度 0 和 top-p 1 可用；NaN、非正输出上限和非法超时时间应报错。 |
| `test_serial_batch_uses_per_input_seed` | 逐张处理图片，并检查每张图使用的种子等于基础种子加图片序号。 |
| `test_shared_processor_contract` | 检查图片像素、问题、模板参数和目标设备是否原样传给腾讯混元的输入处理接口。 |
| `test_generation_settings_reach_model_unchanged` | 检查采样参数、输入和 KV Cache 设置传给模型时保持原样，文本解码也使用指定选项。 |
| `test_help_does_not_load_model_packages` | 查看命令帮助或导入结果工具时，不应加载 PyTorch、torch_npu 或模型。 |
| `test_source_commit_not_just_version` | 依赖的源码版本与样例要求不一致时，应在加载模型前报错。 |
| `test_archive_editable_sources_and_required_interfaces` | 源码归档和可编辑安装都检查版本及模型接口；版本错误或缺少接口时应报错。 |
| `test_source_urls_do_not_expose_credentials` | 安装地址包含认证信息时，写入记录前应移除这些信息。 |
| `test_bad_images_are_rejected_before_model_import` | 图片不存在、路径指向目录或图片损坏时，应在加载模型前报出具体路径。 |
| `test_batch_validation_and_one_pipeline` | 先检查所有输入图片，再加载一次模型，按输入顺序输出各图结果。 |
| `test_help_and_scoring_without_site_packages` | 禁用第三方包目录后，帮助命令和 CPU 分数统计仍能执行。 |
| `test_same_protocol_resumes_once` | 续跑时保留已完成结果，从下一道题继续，已完成题目不会重复写入。 |
| `test_dataset_revision_and_file_identity` | 数据版本或文件内容与要求的完整评测数据不一致时，应报错。 |
| `test_different_configuration_is_rejected` | 续跑设置改变，或已保存设置的校验值损坏时，应报错并保留原结果文件。 |
| `test_duplicate_order_and_seed_drift_rejected` | 续跑结果存在重复题号、题目乱序或种子错误时，应报错。 |
| `test_jsonl_read_is_read_only` | 读取结果时保持文件原样；原文件最后一行缺少换行符时，追加的新结果仍应另起一行。 |
| `test_provenance_changes_do_not_change_resume_settings` | 续跑只比较实际运行设置；改变生成参数或解码方式时，应指出具体差异。 |
| `test_completed_resume_does_not_load_model` | 全部题目已有结果时，续跑只重新生成汇总，既不加载模型也不增加逐题结果。 |
| `test_nested_protocol_and_independent_labels` | 使用原始数据核对标准答案后，应记录核对通过；标准答案不一致时应报错。 |
| `test_unknown_memory_is_not_reported_as_zero` | 没有内存测量值时应报告未知，不能写成 0；已有峰值应正确保留。 |
| `test_direct_mode_and_mixed_results` | 直接回答模式可以重新计分；一份文件混用两种回答模式时应报错。 |
| `test_profile_window_stops_after_three_output_tokens` | 记录处理输入和前三个输出 token 的计算，正常结束或发生异常时都只停止一次采集。 |
| `test_timeout_and_non_eos_cap_are_errors` | 生成超时或达到长度上限但未正常结束时应记错；生成结束标记时应保留正常结果。 |
| `test_oom_record_is_not_scored_correct` | 模拟内存不足后，应保留错误信息；即使答案字母正确也计错。 |
| `test_independent_export_uses_raw_answer_and_full_denominator` | 重新统计时从原文读取选项，保留出错题目的分母；伪造解析答案时应报错。 |
| `test_json_logging_preserves_stdout` | 输出保持为一行合法 JSON，日志处理结束后仍可继续使用原输出流。 |

### [bench/check_ops.py](../../../../manipulation/hy-embodied-0.5-x/infer_with_torch/bench/check_ops.py)

| 函数 | 验证目的与通过条件 |
| --- | --- |
| `test_causal_gqa_decode_and_bidirectional` | 分别检查因果遮挡、共享键值头、单 token 解码和视觉双向注意力的结果。 |
| `test_packed_segments_are_isolated` | 多条变长序列拼接后，每条序列只使用自己的键和值，不受其他序列影响。 |
| `test_attention_argument_and_empty_input_contract` | 空查询仍返回正确的形状和设备；位置参数与具名参数写法结果一致，非法参数会报错。 |
| `test_erf_reference_and_dtype` | 用 erf 公式核对 GELU 的数值结果，并检查 FP32、FP16、BF16 和非连续输入的形状及数据类型。 |
| `test_none_semantics_are_not_tanh` | GELU 选择 none 模式时，应计算 erf 公式，而不是 tanh 近似公式。 |
| `test_instance_scope_and_idempotency` | 只替换指定模型的 GELU；重复安装不再次替换，模型参数数值也保持原样。 |
| `test_rejects_cpu_forward_and_unrecognized_model` | 把输入放在 CPU，或提供没有预期 GELU 模块的模型时，应报错。 |

### [bench/check_model.py](../../../../manipulation/hy-embodied-0.5-x/infer_with_torch/bench/check_model.py)

| 函数 | 验证目的与通过条件 |
| --- | --- |
| `test_model_loading` | 加载后检查所有模型参数都在 NPU 上、使用 BF16，并已替换预期的 29 个 GELU。 |
| `test_smoke_thinking_and_direct` | 用覆盖六类关系的十个固定输入检查两种回答模式、输出格式，以及生成数值是否包含 NaN 或 Inf。 |
| `test_single_and_serial_batch` | 对相同图片、问题和种子，比较单图命令与串行批量命令的生成结果。 |
| `test_observation_preserves_generated_output` | 相同输入最多生成 64 个 token，检查开启耗时和内存统计前后，文本、输出长度和结束状态是否一致。 |
| `test_missing_image_and_truncation` | 缺图时应记录具体错误；只允许生成一个 token 时上限应生效，截断的推理回答不能计为答对。 |
| `test_decoder_backend_equivalence` | 在两个独立进程中用同一输入最多生成 64 个 token，比较 graph 与 native 的文本、输出长度和结束状态。 |
| `test_twenty_samples_two_rounds` | 二十个固定输入连续处理两轮，两轮输出应一致，清理每题后的 NPU 已分配内存应稳定。 |
| `test_profile_ten_samples_five_times` | 显式开启 --profile 后，对十个固定输入各采集五次处理输入及前三输出 token 的算子和内存记录。 |

## 其他验证方法

### 比较两种解码方式的输出

下面的命令在两个独立进程中，分别使用 graph 和 native 处理同一输入，最多生成 64 个 token，比较文本、输出长度和结束状态：

```bash
python manipulation/hy-embodied-0.5-x/infer_with_torch/bench/check_model.py \
  --model-path /path/to/model \
  --dataset-path /path/to/dataset/embspatial_bench.json \
  --output-dir /path/to/checks/backend-comparison \
  --compare-backends \
  BackendNpuTests
```

### 检查连续推理后的结果和内存

下面的命令将二十个固定输入重复处理两轮，检查两轮输出是否一致，以及清理每次输入输出后 NPU 活跃内存是否稳定：

```bash
python manipulation/hy-embodied-0.5-x/infer_with_torch/bench/check_model.py \
  --model-path /path/to/model \
  --dataset-path /path/to/dataset/embspatial_bench.json \
  --output-dir /path/to/checks/stability \
  StabilityNpuTests.test_twenty_samples_two_rounds
```

### 记录 NPU 算子和内存使用

性能分析工具（profiler）记录 NPU 实际执行的算子和内存使用情况。下面的命令对十个固定样本各采集五次，覆盖模型一次处理输入序列的阶段（prefill）及前三个输出 token 的计算。

```bash
python manipulation/hy-embodied-0.5-x/infer_with_torch/bench/check_model.py \
  --model-path /path/to/model \
  --dataset-path /path/to/dataset/embspatial_bench.json \
  --output-dir /path/to/checks/profile \
  --profile \
  StabilityNpuTests.test_profile_ten_samples_five_times
```

`--profile` 开启采集，`--output-dir` 指定本次采集的保存目录，最后一个参数选择对应检查方法。该采集窗口只覆盖上述计算阶段。

## 常见错误处理

| 现象 | 处理方式 |
| --- | --- |
| `Image ... is not a readable file` 或 `cannot be decoded` | 按错误中的序号检查路径、读取权限和图片格式 |
| `Dependency versions differ` 或 `Required HY interfaces missing` | 按[样例依赖文件](../../../../manipulation/hy-embodied-0.5-x/infer_with_torch/requirements.txt)安装指定源码依赖和版本 |
| NPU 不可用或加载运行库失败 | 执行 `npu-smi info`，加载当前 CANN 的 set_env.sh，并核对 torch_npu 配套版本 |
| 模型文件或备份目录不可写 | 使用独立、可写的虚拟环境；错误信息给出目标文件路径 |
| `Decode backend is already ...` | 在新进程启动时选择目标后端 |
| `Resume configuration differs` | 对照列出的字段，使用首次配置继续原结果，或选择新输出路径 |
| JSONL 行解析失败 | 按错误行检查文件并保留原件；完整 JSON 行是续跑和重新统计分数的输入 |
| 标准答案与原始数据或标准答案文件不一致 | 核对数据版本和所用结果文件，保留原始输出供检查 |
