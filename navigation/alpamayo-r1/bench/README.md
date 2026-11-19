# bench 脚本

复算 [优化说明文档](../../../docs/navigation/alpamayo-r1/README.md) 中性能与精度结论的入口。
脚本需要在装好 `torch_npu` 的昇腾环境、且能 import 到上游 alpamayo 源码的目录下运行
（安装步骤见[样例 README](../README.md)）。以下命令都在样例根目录 `navigation/alpamayo-r1/` 下执行。

| 脚本 | 用途 | 对应文档 |
|---|---|---|
| `ablation.py` | 四档消融的端到端耗时与 minADE，`--graph` / `--prefetch` 组合 | 9.1 |
| `eval_minade.py` | 官方 chunk（默认 chunk id 3116）的 100 clips mean minADE | 9.2、9.3 |

常用命令：

```bash
# 四档耗时（默认关闭两个开关）
python bench/ablation.py --model-path model_ckpt --iters 3
python bench/ablation.py --model-path model_ckpt --iters 3 --graph
python bench/ablation.py --model-path model_ckpt --iters 3 --graph --prefetch

# 100 clips 精度
python bench/eval_minade.py --model-path model_ckpt --clips 100 --tag a2
python bench/eval_minade.py --model-path model_ckpt --clips 100 --graph --prefetch --tag a2-combo
```

`eval_minade.py` 默认所有 clip 用 t0 = 5.1 s；要按数据集标注取每个 clip 的起始时刻，
用 `--ood-parquet` 指定 `reasoning/ood_reasoning.parquet`。

算子级统计走 msprof，命令见优化说明文档 9.5 节。
