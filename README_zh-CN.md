# DNN 训练 GPU 显存优化综述

> 系统梳理现代 DNN 训练系统如何减少、迁移、压缩、分布和复用 GPU 显存，并提供配套论文集合。

[English](README.md) | **中文**

![DNN 训练 GPU 显存优化技术总览](assets/gpu-memory-optimization-taxonomy.png)

## 项目简介

GPU 显存已经成为现代深度神经网络训练中的关键瓶颈。模型参数、优化器状态、梯度、激活值、通信缓冲区和临时张量需要共同使用有限的设备显存。当模型无法装入显存时，实际系统通常不会只依赖一种方法，而是组合使用数据卸载、激活重计算、数据压缩、分布式执行以及内存管理优化。

本仓库是 **DNN 训练 GPU 显存优化综述** 的配套资料库，主要提供三部分内容：

1. 从“**技术分类—核心挑战—解决方案类别—代表工作**”逐层展开的分类体系；
2. 与综述主题对应的代表性论文集合；
3. 用于比较显存节省、计算开销、通信开销、精度影响和系统复杂度的阅读路径。

上方图片是整篇综述的结构，第一列给出五类显存优化技术，第二列总结每类技术的核心挑战，第三列展示具体解决方案，第四列列出综述讨论的代表性工作。

## 五类显存优化技术

### 1. 数据卸载技术（§4）

数据卸载通过在 GPU、CPU 内存和 NVMe SSD 等存储层级之间迁移张量来扩展可用显存，其核心问题是数据传输可能阻塞计算、降低训练吞吐率。综述从四个方向讨论降低卸载开销的方法：

- **迁移延迟隐藏（§4.1.1）**：通过计算与传输重叠、提前预取等方式隐藏迁移延迟，代表工作包括 vDNN、SuperNeurons、Capuchin、Sentinel 和 MemFerry。
- **CPU-GPU 混合执行（§4.1.2）**：让 CPU 与 GPU 共同参与训练计算，代表工作包括 KARMA、ZeRO-Offload、ZeRO-Infinity 和 NVREC。
- **联合内存优化（§4.1.3）**：将换入换出与重计算、压缩、分配或调度结合，代表工作包括 Capuchin、HOME、STR、ATP、cDMA、CSWAP 和 FlashNeuron。
- **数据路径优化（§4.1.4）**：优化 GPU、主存与存储设备之间的数据层级和传输路径，代表工作包括 ZeRO-Infinity、MLP-Offload、FlashNeuron 和 SSDTrain。

该类别现收录 [19 份论文文件](papers/Data-Offloading/)。

### 2. 激活重计算优化（§5）

激活重计算（也称重新物化或激活检查点）在前向传播时丢弃部分中间结果，并在反向传播需要时重新计算，以额外计算换取显存空间。综述关注两个核心问题：如何控制重计算开销，以及如何高效搜索重计算策略。

- **自适应重计算（§5.1.1）**：根据显存压力、执行状态或流水线行为动态调整策略，代表工作包括 SuperNeurons、Capuchin、DTR、MegTaiChi、T-Control、AdaPipe 和 Hippie。
- **重计算开销隐藏（§5.1.2）**：利用流水线空闲时间或并行执行隐藏额外计算，代表工作包括 Obscura、Mario 和 Lynx。
- **基于启发式的方法（§5.2.1）**：使用实用启发式规则搜索低成本检查点策略，代表工作包括 Lynx、AdaPipe、Mario 和 T-Control。
- **基于 ILP 的全局方法（§5.2.2）**：将重新物化建模为全局优化问题，代表工作包括 Checkmate、Rockmate、Infinipipe、Memo 和 Adacc。

该类别现收录 [20 份论文文件](papers/Activation-Recomputation/)。

### 3. 数据压缩技术（§6）

数据压缩使用低精度、量化、稀疏化或低秩表示减少激活、梯度、优化器状态或模型数据的占用。有效的压缩方法需要在节省显存的同时控制编码解码成本，并保持模型收敛性和精度。

- **模型精度保持（§6.1）**
  - *自适应低精度压缩（§6.1.1）*：为不同数据选择合适的数值格式和精度，代表工作包括 FP8-LM、SNIP、BitNet、Q-Adam-mini 和 COAT。
  - *误差可控的低秩压缩（§6.1.2）*：在控制近似误差的同时降低数据秩，代表工作包括 GaLore、ELRT、DLRT、LoRITa 和 CompAct。
- **压缩开销降低（§6.2）**
  - *选择性压缩（§6.2.1）*：只压缩能够获得良好显存收益的张量，代表工作包括 ActNN、GACT、AC-GC、Q-Adam-mini、DSR 和 RigL。
  - *高效压缩算法（§6.2.2）*：降低元数据、格式转换和运行时开销，代表工作包括 ActNN、GACT、FP8-LM、BitNet、COAT、ANT、Gist、SparseTrain 和 SWAT。

该类别现收录 [31 份论文文件](papers/Data-Compression/)。

### 4. 分布式 GPU 显存优化（§7）

分布式训练既提供了扩展显存容量的机会，也引入了新的显存瓶颈。模型状态可以跨设备分片，但流水线调度可能延长激活生命周期，不均衡的划分还可能使单个设备成为显存瓶颈。

- **数据冗余消除（§7.1）**：细粒度切分参数、梯度和优化器状态，代表工作包括 ZeRO、FSDP、PipeMesh 和 Ulysses。
- **长生命周期激活显存最小化（§7.2）**：通过调度和集成式显存优化缩短激活生命周期，代表工作包括 PipeDream、DAPPLE、MegTaiChi、SlimPipe、GPipe、PipeMare、PipeOffload 和 SPPO。
- **跨设备显存负载均衡（§7.3）**：通过负载重分配、结合其他显存优化、算法设计或显存感知并行编排改善设备间不均衡，代表工作包括 vPipe、BPipe、MPress、Merak、AdaPipe、GShard、Switch Transformer、mCAP、SmartMoE 和 FasterMoE。

该类别现收录 [35 份论文文件](papers/Distributed-Memory-Optimization/)。

### 5. 其他优化（§8）

部分显存瓶颈并非只由模型状态或激活值引起。显存碎片会造成空闲空间无法使用，算子工作区和中间张量则可能产生很高的瞬时显存峰值。

- **GPU 显存碎片整理（§8.1）**
  - *静态分区（§8.1.1）*：划分内存区域或规划可复用内存池，代表工作包括 vDNN++、DyMem 和 Coop。
  - *预规划分配（§8.1.2）*：根据张量生命周期提前规划内存布局，代表工作包括 MegTaiChi、MODeL、MP-MoE 和 STAlloc。
  - *运行时碎片管理（§8.1.3）*：在线处理动态分配模式，代表工作包括 GMLake 和 T-Control。
- **临时缓冲区缩减（§8.2）**
  - *算子优化（§8.2.1）*：重新设计高显存占用算子，代表工作包括 MEC、KeOps、FlashAttention、FlashAttention-2、FlashAttention-3 和 BPT。
  - *计算图优化（§8.2.2）*：调整算子顺序、图执行方式或内存布局，代表工作包括 LayRub、MP-MoE、eXLA、MODeL、ROAM 和 Relax。

该类别现收录 [21 份论文文件](papers/Other-Optimizations/)。

## 论文集合概览

仓库目前共包含 **126 份 PDF 文件**。部分工作同时使用多种显存优化机制，因此会被交叉收录；这里统计的是文件数量，而不是去重后的论文数量。

| 分类 | 时间范围 | 文件数 | 目录 |
| --- | ---: | ---: | --- |
| 数据卸载 | 2016-2025 | 19 | [`papers/Data-Offloading`](papers/Data-Offloading/) |
| 激活重计算 | 2016-2026 | 20 | [`papers/Activation-Recomputation`](papers/Activation-Recomputation/) |
| 数据压缩 | 2018-2026 | 31 | [`papers/Data-Compression`](papers/Data-Compression/) |
| 分布式显存优化 | 2017-2025 | 35 | [`papers/Distributed-Memory-Optimization`](papers/Distributed-Memory-Optimization/) |
| 其他优化 | 2017-2026 | 21 | [`papers/Other-Optimizations`](papers/Other-Optimizations/) |

## 如何使用本仓库

1. 先通过分类图定位与你的训练任务相关的核心挑战。
2. 根据图中的章节编号阅读综文对应部分。
3. 进入相应论文目录，先阅读奠基工作，再追踪较新的系统和方法。
4. 从显存节省、计算开销、通信或 I/O 开销、精度影响和实现复杂度五个方面比较方案。
5. 当单一方法无法满足需求时，重点关注组合式显存优化方案。

## 参与贡献

欢迎通过 GitHub Issue 或 Pull Request 提交勘误和新论文建议。添加论文时，请提供年份、发表地点、标题、官方或开放访问链接、代码链接（如有），以及对主要贡献的简要说明。详细分类和记录格式见 [`papers/README.md`](papers/README.md)。

## 引用

如果本综述或仓库对你的研究有所帮助，欢迎引用我们的论文。论文正式发表后，我们将在此补充 BibTeX 信息。

## 版权说明

仓库原创内容遵循 [`LICENSE`](LICENSE) 中的许可协议。所收录论文的版权归作者或出版机构所有，请遵循每篇论文的再分发许可，并优先使用出版社页面、arXiv 页面或作者公开版本。

## 联系方式

本项目由浙江大学智能存储与计算系统实验室（ISCS Laboratory）维护。

联系邮箱：zjuchenping@zju.edu.cn
