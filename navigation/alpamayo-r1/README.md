# Alpamayo-R1 智能驾驶VLA大模型昇腾A2/A3平台使用指南

[English Version](README_en.md)

本目录介绍在昇腾 Atlas A2 与 Atlas A3 上如何对Alpamayo-R1（下文简称 AR-1）模型进行推理，并附带精度验证方法、性能参考数据及输出示例。

## AR-1整体介绍
AR-1是英伟达提出的面向L4/L5级智能驾驶的视觉-语言-动作（VLA）大模型，旨在解决自动驾驶在长尾场景中因黑箱决策导致的脆弱性与不可解释性问题。该模型通过融合视觉感知与自然语言指令，构建结构化的场景理解，并引入显式的因果思维链（Chain-of-Causal-Thinking, CoC）机制，在推理过程中模拟人类驾驶员“观察—分析因果—预判后果—决策”的认知流程，从而生成更稳健、可验证的轨迹规划，同时输出自然语言形式的决策解释，提升系统透明度与人机协同效率。

## 环境准备

1. 安装CANN软件包。

   本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.5.0`及以上。本文给出的性能与精度数据在`CANN 9.1.0` + `torch 2.8.0` + `torch_npu 2.8.0`环境上采集，A2与A3使用同一套软件版本。

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Atlas-A2-cann-kernels_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=netconda&OS=Debian)进行安装。
      - `${version}`表示CANN包版本号，如8.5.0/9.1.0。
      - `${arch}`表示CPU架构，如aarch64、x86_64。

2. 安装 uv (如果没有uv)
    ``` bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ```
3. 克隆代码
    ```bash
    # 克隆cann-recipes-embodied-ai项目代码
    git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git

    # 克隆alpamayo原仓库代码
    git clone https://github.com/NVlabs/alpamayo.git
    cd alpamayo
    git reset --hard e0e2ac32ebd9c3ad8a314099aa5af5c3aaa28073
    cd ..

    # 应用适配昇腾A2/A3的patch
    patch -p0 < cann-recipes-embodied-ai/navigation/alpamayo-r1/patches/ar-1.patch

    # 应用图模式与预取优化patch（基于 ar-1.patch 之后的状态，顺序不可交换）
    patch -p0 < cann-recipes-embodied-ai/navigation/alpamayo-r1/patches/graph-prefetch.patch

    # 将原仓库alpamayo的代码复制到本项目目录下
    cp -rn alpamayo/* cann-recipes-embodied-ai/navigation/alpamayo-r1/
    cd cann-recipes-embodied-ai/navigation/alpamayo-r1
    ```
4. 创建虚拟环境
    ```bash
    uv venv ar1_venv --python python3.11
    source ar1_venv/bin/activate
    ```
5. 安装环境依赖
    ```bash
    # 安装alpamayo-r1依赖
    uv sync --active
    ```
6. 安装physical-ai-av（physical-ai-av依赖python312，需手动修改依赖后安装，再修改语法）
    ```bash
    #手动下载tar包：
    wget https://files.pythonhosted.org/packages/25/20/730fab2dc243e720fa107520c187c7a3a928f471c40d367d55980032daac/physical_ai_av-0.1.0.tar.gz
    #解压
    tar -zxvf physical_ai_av-0.1.0.tar.gz
    #应用patch
    patch -p0 < patches/physical_ai_av-0.1.0-python311.patch
    # 安装
    cd physical_ai_av-0.1.0
    uv pip install .
    cd ..
    ```

## 模型权重下载
```bash
    hf download nvidia/Alpamayo-R1-10B --local-dir model_ckpt
```

## 模型推理
```bash
    # 执行推理
    cd ./src/alpamayo_r1
    python test_inference.py
```
基于上述运行过程，可以得到Alpamayo-R1的单次推理结果（详细的优化过程介绍见 [Alpamayo-R1 优化说明文档](../../docs/navigation/alpamayo-r1/README.md)）。

- 推理结果：单次推理结果为10组驾驶轨迹预测序列和CoC思维链文本。
- 推理性能（单次端到端，含VLM自回归生成与扩散采样，各设备20次稳态中位）：

    |设备|推理配置|单次推理耗时|相对默认|
    |:---:|:---:|:---:|:---:|
    |Atlas 800T A2 (910B3)|默认|7.88 s|—|
    |Atlas 800T A2 (910B3)|图模式|**7.49 s**|−4.9%|
    |Atlas A3 (Ascend910_9382)|默认|6.79 s|—|
    |Atlas A3 (Ascend910_9382)|图模式|**6.53 s**|−3.9%|

- 推理精度（官方 `chunk_0000` = chunk id 3116，100 clips，10条预测轨迹，mean minADE，越低越好）：

    |设备|mean minADE|门槛|
    |:---:|:---:|:---:|
    |Atlas 800T A2 (910B3)|**0.4828**|≤0.5676|
    |Atlas A3 (Ascend910_9382)|**0.4954**|≤0.5676|

## 推理效果展示

下面用一组真实推理结果展示模型效果。参考输入为公开 clip `030c760c-ae38-49aa-9ad8-f5650a545d26` 在 t0 = 5.1 s 时刻的 4 路摄像头画面：

![模型参考输入：4路摄像头画面](https://raw.gitcode.com/xy200303/ar1-assets/blobs/073fd5f72f45c4920801ca9873ca16010f899855/ar1_input_cameras.png)

对应的推理输出如下图所示。浅蓝色为模型给出的 10 条预测轨迹，黑色为真值轨迹，橙色为其中与真值最接近的一条，minADE 为 0.2315 m。图下方的文本是该条轨迹对应的 CoC 思维链输出，含义是车辆应向左微调，以避开右侧的施工区。

![模型推理结果：10条预测轨迹与真值轨迹对比](https://raw.gitcode.com/xy200303/ar1-assets/blobs/1eb079468c6a5ef6b8ee5105eba72b90f2bc2b4c/ar1_output_trajectories.png)

下面这张动图展示了同一次推理中轨迹随时间步逐步生成的过程：

![推理过程动图：轨迹逐步生成](https://raw.gitcode.com/xy200303/ar1-assets/blobs/f9706f5f282c3198ecbf72d3a9b355e57d5fa63c/ar1_inference.gif)

## 图模式与权重预取

AR-1 的推理链路中，扩散去噪部分每一步的输入形状固定、步数固定，可以整段捕获成图后重复回放；VLM 解码部分的逐层权重读取是访存热点，可以在计算前异步预取到片上缓存。本节介绍的两个开关就是分别开启这两项优化，默认均关闭，不开启时推理行为与上一节完全一致。

开关既可设置环境变量，也可作为 `sample_trajectories_from_data_with_vlm_rollout` 的参数传入，两者任一为真即生效：

|开关|默认|作用|
|---|---|---|
|`AR1_ENABLE_GRAPH` / `enable_graph`|关闭|把扩散去噪的10步整段捕获，之后每次推理直接回放；捕获失败时自动退回未捕获路径。视觉塔位置编码缓存随该开关一起生效|
|`AR1_ENABLE_PREFETCH` / `enable_prefetch`|关闭|在VLM解码逐层计算前，异步预取本层MatMul权重到片上缓存|
|`AR1_STATIC_KV`|开启|让VLM缓存长度跨推理保持一致，从而使图模式每个进程只需捕获一次；与图模式开关无关，设为0回到按实际长度分配|
|`AR1_VISION_CACHE`|开启|缓存视觉塔的网格位置编码，仅在图模式下挂载；设为0可单独关闭|

支持矩阵（A2 与 A3 使用同一份 patch、同一套开关）：

|功能|Atlas 800T A2 (910B3)|Atlas A3 (Ascend910_9382)|
|---|---|---|
|单次推理（默认）|已实测|已实测|
|图模式（扩散 10 步捕获回放）|已实测，首次捕获约 0.4 s|已实测，首次捕获约 0.4 s|
|权重预取（CMO 接口）|已接入，端到端增益未量化|已接入，端到端增益未量化|
|静态 KV 与 vision 元数据缓存|默认开启|默认开启|
|NPU 原生 Cholesky|已实测|已实测|
|官方 chunk_0000 精度（100 clips，10 轨迹）|0.4828|0.4954|

```bash
    # 四档消融：默认关闭两个开关，--graph / --prefetch 分别打开
    python bench/ablation.py --model-path model_ckpt --iters 3 --graph --prefetch
    # 100 clips minADE，chunk_0000 为 chunk id 3116
    python bench/eval_minade.py --model-path model_ckpt --clips 100 --graph --prefetch --tag a2
```

以上命令都在本目录（`navigation/alpamayo-r1/`）下执行。脚本参数见 [`bench/README.md`](bench/README.md)。

适用设备与运行环境：Atlas 800T A2 (910B3)、Atlas 800I A2 与 Atlas A3 (Ascend910_9382)，均运行于 CANN 9.1.0 + torch_npu 2.8.0。A3 与 A2 使用同一份 patch、同一套开关，无需额外改动；两者的差异仅体现在耗时上（见上文性能表）。首次捕获约 0.4 s，属于每个进程一次性开销，稳态回放不再有额外编译。

其他限制：

- 图模式要求捕获区间内没有会自行派生侧流的融合算子，因此轨迹专家的注意力使用即时执行实现，数学上与融合算子等价；
- 权重预取在 VLM 解码阶段提交，是否生效以 `CmoWeightPrefetcher.stats` 为准，端到端增益未量化；
- 权重与数据集不随代码入库，获取方式见下文。

更细的边界说明见 [优化说明文档](../../docs/navigation/alpamayo-r1/README.md) 第 11 节。

## 在昇腾A2/A3上的精度验证步骤

### 基于minADE指标来验证昇腾的推理精度

在官方 `chunk_0000`（= `get_clip_chunk()` 返回的 chunk id 3116，共100个clips）上按1/3/5/10轨迹实测A2与A3两个平台，t0=5.1s，eager attention，CANN 9.1.0 + torch_npu 2.8.0：

|预测轨迹数|A2 910B3|A3（Ascend910_9382）|昇腾A2参考值|
|:---:|:---:|:---:|:---:|
|1|1.7244|1.6155|2.2026|
|3|0.9314|0.9371|0.9498|
|5|0.7556|0.7712|0.8514|
|**10**|**0.4828**|**0.4954**|0.5676（门槛 ≤）|

3轨迹时两个平台与参考值相差1–2%，5轨迹优于参考值，说明chunk与t0的取法与官方一致；10轨迹低于门槛0.5676。两组对照实验确认图模式与算子替换不改变模型输出：注意力实现用eager与FA两种方式结果完全一致，算子替换前后20个clip逐一比对结果相同。详见 [优化说明文档](../../docs/navigation/alpamayo-r1/README.md) 的第 9 节。

## Citation
```
@article{nvidia2025alpamayo,
      title={{Alpamayo-R1}: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail},
      author={NVIDIA and Yan Wang and Wenjie Luo and Junjie Bai and Yulong Cao and Tong Che and Ke Chen and Yuxiao Chen and Jenna Diamond and Yifan Ding and Wenhao Ding and Liang Feng and Greg Heinrich and Jack Huang and Peter Karkus and Boyi Li and Pinyi Li and Tsung-Yi Lin and Dongran Liu and Ming-Yu Liu and Langechuan Liu and Zhijian Liu and Jason Lu and Yunxiang Mao and Pavlo Molchanov and Lindsey Pavao and Zhenghao Peng and Mike Ranzinger and Ed Schmerling and Shida Shen and Yunfei Shi and Sarah Tariq and Ran Tian and Tilman Wekel and Xinshuo Weng and Tianjun Xiao and Eric Yang and Xiaodong Yang and Yurong You and Xiaohui Zeng and Wenyuan Zhang and Boris Ivanovic and Marco Pavone},
      year={2025},
      journal={arXiv preprint arXiv:2511.00088},
}
```

## 目录结构

本样例在仓库中的目录如下。`patches/` 放上游源码的最小改动，`src/` 放本例新增的运行时模块，
`bench/` 是复算优化说明文档中结论的脚本。

```text
navigation/alpamayo-r1/
├── README.md、README_en.md
├── patches/
│   ├── ar-1.patch                       # 昇腾基础适配
│   ├── graph-prefetch.patch             # 图模式、静态 KV、权重预取、NPU 原生 Cholesky
│   └── physical_ai_av-0.1.0-python311.patch
├── src/alpamayo_r1/                     # 本例新增的运行时模块，安装时拷入上游源码树
│   ├── flags.py                         # 环境变量开关集中读取
│   ├── graph_runner.py                  # 扩散步的捕获与回放，失败回退即时执行
│   ├── npu_ext.py                       # 上游文件的一行导入入口
│   ├── prefetch.py                      # 权重预取：CMO 与跨步触读两种实现
│   ├── static_kv.py                     # 桶化静态 KV，使图可跨请求复用
│   └── vision_cache.py                  # vision 塔位置编码缓存
└── bench/                               # 复算性能与精度结论的脚本
```

`cp -rn alpamayo/*` 会把上游源码合并进本目录的 `src/alpamayo_r1/`，本例新增的 6 个模块已经在同一目录下，不需要额外拷贝。安装完成后，上游的 `test_inference.py` 与 `bench/` 下的脚本都从该目录导入：

```text
alpamayo/src/alpamayo_r1/
├── flags.py、graph_runner.py、npu_ext.py、prefetch.py、static_kv.py、vision_cache.py
├── action_space/utils.py                # 轨迹平滑求解改为 NPU 原生 Cholesky
├── models/action_in_proj.py             # RmsNorm 换纯 torch 实现，便于入图
├── models/alpamayo_r1.py                # 接入静态 KV、位置编码缓存、预取与图模式
└── ...                                  # 上游原有文件
```
