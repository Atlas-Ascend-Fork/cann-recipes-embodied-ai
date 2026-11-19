# 第三方来源

本样例只提交补丁、配置与脚本。模型权重、训练数据、LIBERO 资产和第三方源码树都不随仓库发布。下表是各部分固定的来源与许可依据。

| 内容 | 固定版本 | 许可依据 |
|---|---|---|
| LingBot-VA 源码及补丁上下文 | Robbyant/lingbot-va @ 7c6ffa9 | 上游 LICENSE.txt 为 Apache-2.0，保留原版权声明；本样例附带 Apache-2.0 许可文本 |
| Spatial 权重 | Wjjjh/lingbot-va-libero-spatial @ 933ab66 | 模型卡与 Hub API 声明 Apache-2.0，属社区发布 |
| Object 权重 | Wjjjh/lingbot-va-libero-object @ 489806f | 模型卡与 Hub API 声明 Apache-2.0，属社区发布 |
| Goal 权重 | Wjjjh/lingbot-va-libero-goal @ 42455fd | 模型卡与 Hub API 声明 Apache-2.0，属社区发布 |
| Long 权重及共享组件 | robbyant/lingbot-va-posttrain-libero-long @ 0e89d1e | Hub API 声明 Apache-2.0 |
| LIBERO 代码 | Lifelong-Robot-Learning/LIBERO @ 8f1084e | 代码为 MIT；上游 README 把 datasets 单列为 CC BY 4.0，两者适用范围不同 |
| BDDL | 实际安装的 bddl 1.0.1 | 安装包为 MIT，版权归 Stanford Vision and Learning Lab |
| robosuite | 实际安装的 robosuite 1.4.0 | 安装包为 MIT，保留其版权声明与 AUTHORS |

上游链接：

- [LingBot-VA @ 7c6ffa9](https://github.com/Robbyant/lingbot-va/tree/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb)
- [Spatial](https://huggingface.co/Wjjjh/lingbot-va-libero-spatial) / [Object](https://huggingface.co/Wjjjh/lingbot-va-libero-object) / [Goal](https://huggingface.co/Wjjjh/lingbot-va-libero-goal)
- [Long](https://huggingface.co/robbyant/lingbot-va-posttrain-libero-long)
- [LIBERO 许可说明](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01#license)

## 权重与配置的对应关系

三个社区模型的 `attn_mode` 从 `flex` 改为 `torch`，Long 配置保持原样。`prepare.py` 将 `configs/` 中的对应 Transformer 配置写入权重目录，再按 `checkpoints.json` 校验文件大小和 SHA256。它还生成 `configuration.json` 元数据，该文件不在锁定的 Hugging Face revision 中。

## 数据与依赖

LIBERO 的 datasets 按 CC BY 4.0 使用。mesh 与纹理按各自来源的许可使用，传递依赖和系统动态库跟随其上游许可。
