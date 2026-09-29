# Survey of GPU Memory Optimization for DNN Training

> A structured survey and paper collection for understanding how modern DNN training systems reduce, move, compress, distribute, and reuse GPU memory.

[English](#english) | [中文](#中文)

![Overview of GPU memory optimization techniques for DNN training](assets/gpu-memory-optimization-taxonomy.png)

## English

### Overview

GPU memory has become one of the main constraints on training modern deep neural networks. Parameters, optimizer states, gradients, activations, communication buffers, and temporary tensors must all share limited device memory. When a model no longer fits, the solution is rarely a single technique: practical systems often combine offloading, recomputation, compression, distributed execution, and memory-management optimizations.

This repository accompanies our survey of **GPU memory optimization for DNN training**. It connects three resources:

1. a taxonomy that explains the field from **classification**, **challenge**, and **solution category** to **related work**;
2. a curated collection of representative papers;
3. a reading path for comparing memory savings against computation, communication, accuracy, and system complexity.

The figure above is the roadmap of the survey. Read it from left to right:

- **Classification** identifies the major family of techniques.
- **Challenge** describes the central problem within that family.
- **Solution Category** groups methods by the mechanism they use.
- **Related Works** provides representative systems and papers discussed in the survey.

### Taxonomy at a glance

#### 1. Data Offloading Techniques (§4)

Offloading extends effective GPU memory capacity by moving tensors between the GPU and external storage tiers such as CPU memory or NVMe SSDs. Its main challenge is that data movement can stall computation and reduce training throughput.

The survey organizes solutions into four directions:

- **Migration Latency Hiding (§4.1.1):** overlap transfers with computation or prefetch data before it is needed. Representative works include vDNN, SuperNeurons, Capuchin, Sentinel, and MemFerry.
- **Hybrid CPU-GPU Execution (§4.1.2):** use CPU and GPU resources jointly instead of treating the host only as passive storage. Representative works include KARMA, ZeRO-Offload, ZeRO-Infinity, and NVREC.
- **Joint Memory Optimizations (§4.1.3):** combine swapping with recomputation, compression, allocation, or scheduling. Representative works include Capuchin, HOME, STR, ATP, cDMA, CSWAP, and FlashNeuron.
- **Data Path Optimization (§4.1.4):** improve the hierarchy and transfer path among GPU memory, host memory, and storage. Representative works include ZeRO-Infinity, MLP-Offload, FlashNeuron, and SSDTrain.

Browse the [19 collected files](papers/Data-Offloading/) in this category.

#### 2. Activation Recomputation Optimization (§5)

Activation recomputation, also known as rematerialization or checkpointing, discards selected intermediate tensors during the forward pass and regenerates them during backpropagation. It saves memory at the cost of additional computation.

The key questions are how to control this overhead and how to choose an efficient recomputation plan:

- **Adaptive Recomputation (§5.1.1):** adapt the checkpointing or recomputation policy to memory pressure, execution state, or pipeline behavior. Representative works include SuperNeurons, Capuchin, DTR, MegTaiChi, T-Control, AdaPipe, and Hippie.
- **Recomputation Overhead Masking (§5.1.2):** hide recomputation in otherwise idle pipeline time or overlap it with other work. Representative works include Obscura, Mario, and Lynx.
- **Heuristic-Based Methods (§5.2.1):** use practical heuristics to find low-cost checkpointing strategies. Representative works include Lynx, AdaPipe, Mario, Kusumoto et al., and T-Control.
- **ILP-Based Global Methods (§5.2.2):** formulate rematerialization as a global optimization problem. Representative works include Checkmate, Rockmate, Infinipipe, Memo, and Adacc.

Browse the [20 collected files](papers/Activation-Recomputation/) in this category.

#### 3. Data Compression Techniques (§6)

Compression reduces the footprint of activations, gradients, optimizer states, or model data through lower precision, quantization, sparsity, or low-rank representations. A useful method must save enough memory to justify encoding and decoding costs while preserving convergence and accuracy.

The survey considers two major challenges:

- **Model Accuracy Preservation (§6.1)**
  - *Adaptive Low-Precision Compression (§6.1.1)* selects suitable numeric formats or precision levels. Representative works include FP8-LM, SNIP, BitNet, Q-Adam-mini, and COAT.
  - *Error-Controlled Low-Rank Compression (§6.1.2)* reduces rank while controlling approximation error. Representative works include GaLore, ELRT, DLRT, LoRITa, and CompAct.
- **Compression Overhead Reduction (§6.2)**
  - *Selective Compression (§6.2.1)* compresses only the tensors that provide a favorable memory-performance trade-off. Representative works include ActNN, GACT, AC-GC, Q-Adam-mini, DSR, and RigL.
  - *Efficient Compression Algorithms (§6.2.2)* reduce metadata, conversion, and runtime costs. Representative works include ActNN, GACT, FP8-LM, BitNet, COAT, ANT, BitTrain, Gist, SparseTrain, and SWAT.

Browse the [31 collected files](papers/Data-Compression/) in this category.

#### 4. Distributed GPU Memory Optimization (§7)

Distributed training introduces both opportunities and new memory bottlenecks. Model states can be sharded across devices, but pipeline schedules may prolong activation lifetimes and unbalanced partitions may leave one device as the memory bottleneck.

The survey groups the solutions by three challenges:

- **Data Redundancy Elimination (§7.1):** fine-grained sharding partitions parameters, gradients, and optimizer states. Representative works include ZeRO, FSDP, PipeMesh, and Ulysses.
- **Prolonged Activation Memory Minimization (§7.2):** scheduling and integrated memory optimizations shorten activation lifetimes. Representative works include PipeDream, DAPPLE, MegTaiChi, SlimPipe, GPipe, PipeMare, PipeOffload, and SPPO.
- **Memory Load Balance Across Devices (§7.3):** redistribute workloads, combine parallelism with memory optimization, improve algorithms, or orchestrate parallel strategies in a memory-aware way. Representative works include vPipe, BPipe, MPress, Merak, AdaPipe, GShard, Switch Transformer, mCAP, HIPPIE, SmartMoE, and FasterMoE.

Browse the [35 collected files](papers/Distributed-Memory-Optimization/) in this category.

#### 5. Other Optimizations (§8)

Some memory bottlenecks are caused neither by model states nor by activations alone. Fragmentation can make free memory unusable, while operator workspaces and intermediate tensors can create large temporary peaks.

- **GPU Memory Defragmentation (§8.1)**
  - *Static Partitioning (§8.1.1)* isolates memory regions or plans reusable pools. Representative works include vDNN++, DyMem, and Coop.
  - *Preplanned Allocation (§8.1.2)* plans tensor placement from known lifetimes. Representative works include MegTaiChi, MODeL, MP-MoE, and STAlloc.
  - *Runtime Fragmentation Management (§8.1.3)* handles changing allocation patterns online. Representative works include GMLake and T-Control.
- **Temporary Buffer Reduction (§8.2)**
  - *Operator Optimization (§8.2.1)* redesigns memory-intensive operators. Representative works include MEC, KeOps, FlashAttention, FlashAttention-2, FlashAttention-3, and BPT.
  - *Computation Graph Optimization (§8.2.2)* changes operator order, graph execution, or memory layout. Representative works include LayRub, MP-MoE, eXLA, MODeL, ROAM, and Relax.

Browse the [21 collected files](papers/Other-Optimizations/) in this category.

### Paper collection

The repository currently contains **126 PDF files**. Some works appear in more than one category because they combine multiple memory-optimization mechanisms; therefore, this number represents files rather than unique papers.

| Category | Coverage | Files | Directory |
| --- | ---: | ---: | --- |
| Data Offloading | 2016-2025 | 19 | [`papers/Data-Offloading`](papers/Data-Offloading/) |
| Activation Recomputation | 2016-2026 | 20 | [`papers/Activation-Recomputation`](papers/Activation-Recomputation/) |
| Data Compression | 2018-2026 | 31 | [`papers/Data-Compression`](papers/Data-Compression/) |
| Distributed Memory Optimization | 2017-2025 | 35 | [`papers/Distributed-Memory-Optimization`](papers/Distributed-Memory-Optimization/) |
| Other Optimizations | 2017-2026 | 21 | [`papers/Other-Optimizations`](papers/Other-Optimizations/) |

### How to use this repository

1. Start with the taxonomy figure to identify the challenge related to your workload.
2. Follow the section number to the corresponding discussion in the survey.
3. Open the category directory and read the foundational work before newer systems.
4. Compare methods along five dimensions: memory savings, computation overhead, communication or I/O overhead, accuracy impact, and implementation complexity.
5. Look for combined techniques when a single optimization is insufficient.

### Contributing

Corrections and new paper suggestions are welcome through GitHub issues or pull requests. When adding a paper, please include its year, venue, title, official or open-access URL, code link if available, and a short explanation of its primary contribution. See [`papers/README.md`](papers/README.md) for the detailed taxonomy and contribution format.

### Citation

If this survey or repository is useful to your work, please cite our paper. The BibTeX entry will be added after publication.

### Copyright notice

Repository-authored content is provided under the license in [`LICENSE`](LICENSE). Copyright of collected papers remains with their authors and publishers. Please use official publisher pages, arXiv records, or author-provided open-access versions and respect the redistribution terms of each paper.

Contact: zjuchenping@zju.edu.cn

---

## 中文

### 项目简介

GPU 显存已经成为现代深度神经网络训练中的关键瓶颈。模型参数、优化器状态、梯度、激活值、通信缓冲区和临时张量需要共同使用有限的设备显存。当模型无法装入显存时，实际系统通常不会只依赖一种方法，而是组合使用数据卸载、激活重计算、数据压缩、分布式执行以及内存管理优化。

本仓库是 **DNN 训练 GPU 显存优化综述** 的配套资料库，主要提供三部分内容：

1. 从“**技术分类—核心挑战—解决方案类别—代表工作**”逐层展开的分类体系；
2. 与综述主题对应的代表性论文集合；
3. 用于比较显存节省、计算开销、通信开销、精度影响和系统复杂度的阅读路径。

上方图片是整篇综述的结构，第一列给出五类显存优化技术，第二列总结每类技术的核心挑战，第三列展示具体解决方案，第四列列出综述讨论的代表性工作。

### 五类显存优化技术

#### 1. 数据卸载技术（§4）

数据卸载通过在 GPU、CPU 内存和 NVMe SSD 等存储层级之间迁移张量来扩展可用显存，其核心问题是数据传输可能阻塞计算、降低训练吞吐率。综述从四个方向讨论降低卸载开销的方法：

- **迁移延迟隐藏（§4.1.1）**：通过计算与传输重叠、提前预取等方式隐藏迁移延迟，代表工作包括 vDNN、SuperNeurons、Capuchin、Sentinel 和 MemFerry。
- **CPU-GPU 混合执行（§4.1.2）**：让 CPU 与 GPU 共同参与训练计算，代表工作包括 KARMA、ZeRO-Offload、ZeRO-Infinity 和 NVREC。
- **联合内存优化（§4.1.3）**：将换入换出与重计算、压缩、分配或调度结合，代表工作包括 Capuchin、HOME、STR、ATP、cDMA、CSWAP 和 FlashNeuron。
- **数据路径优化（§4.1.4）**：优化 GPU、主存与存储设备之间的数据层级和传输路径，代表工作包括 ZeRO-Infinity、MLP-Offload、FlashNeuron 和 SSDTrain。

该类别现收录 [19 份论文文件](papers/Data-Offloading/)。

#### 2. 激活重计算优化（§5）

激活重计算（也称重新物化或激活检查点）在前向传播时丢弃部分中间结果，并在反向传播需要时重新计算，以额外计算换取显存空间。综述关注两个核心问题：如何控制重计算开销，以及如何高效搜索重计算策略。

- **自适应重计算（§5.1.1）**：根据显存压力、执行状态或流水线行为动态调整策略，代表工作包括 SuperNeurons、Capuchin、DTR、MegTaiChi、T-Control、AdaPipe 和 Hippie。
- **重计算开销隐藏（§5.1.2）**：利用流水线空闲时间或并行执行隐藏额外计算，代表工作包括 Obscura、Mario 和 Lynx。
- **基于启发式的方法（§5.2.1）**：使用实用启发式规则搜索低成本检查点策略，代表工作包括 Lynx、AdaPipe、Mario 和 T-Control。
- **基于 ILP 的全局方法（§5.2.2）**：将重新物化建模为全局优化问题，代表工作包括 Checkmate、Rockmate、Infinipipe、Memo 和 Adacc。

该类别现收录 [20 份论文文件](papers/Activation-Recomputation/)。

#### 3. 数据压缩技术（§6）

数据压缩使用低精度、量化、稀疏化或低秩表示减少激活、梯度、优化器状态或模型数据的占用。有效的压缩方法需要在节省显存的同时控制编码解码成本，并保持模型收敛性和精度。

- **模型精度保持（§6.1）**
  - *自适应低精度压缩（§6.1.1）*：为不同数据选择合适的数值格式和精度，代表工作包括 FP8-LM、SNIP、BitNet、Q-Adam-mini 和 COAT。
  - *误差可控的低秩压缩（§6.1.2）*：在控制近似误差的同时降低数据秩，代表工作包括 GaLore、ELRT、DLRT、LoRITa 和 CompAct。
- **压缩开销降低（§6.2）**
  - *选择性压缩（§6.2.1）*：只压缩能够获得良好显存收益的张量，代表工作包括 ActNN、GACT、AC-GC、Q-Adam-mini、DSR 和 RigL。
  - *高效压缩算法（§6.2.2）*：降低元数据、格式转换和运行时开销，代表工作包括 ActNN、GACT、FP8-LM、BitNet、COAT、ANT、Gist、SparseTrain 和 SWAT。

该类别现收录 [31 份论文文件](papers/Data-Compression/)。

#### 4. 分布式 GPU 显存优化（§7）

分布式训练既提供了扩展显存容量的机会，也引入了新的显存瓶颈。模型状态可以跨设备分片，但流水线调度可能延长激活生命周期，不均衡的划分还可能使单个设备成为显存瓶颈。

- **数据冗余消除（§7.1）**：细粒度切分参数、梯度和优化器状态，代表工作包括 ZeRO、FSDP、PipeMesh 和 Ulysses。
- **长生命周期激活显存最小化（§7.2）**：通过调度和集成式显存优化缩短激活生命周期，代表工作包括 PipeDream、DAPPLE、MegTaiChi、SlimPipe、GPipe、PipeMare、PipeOffload 和 SPPO。
- **跨设备显存负载均衡（§7.3）**：通过负载重分配、结合其他显存优化、算法设计或显存感知并行编排改善设备间不均衡，代表工作包括 vPipe、BPipe、MPress、Merak、AdaPipe、GShard、Switch Transformer、mCAP、SmartMoE 和 FasterMoE。

该类别现收录 [35 份论文文件](papers/Distributed-Memory-Optimization/)。

#### 5. 其他优化（§8）

部分显存瓶颈并非只由模型状态或激活值引起。显存碎片会造成空闲空间无法使用，算子工作区和中间张量则可能产生很高的瞬时显存峰值。

- **GPU 显存碎片整理（§8.1）**
  - *静态分区（§8.1.1）*：划分内存区域或规划可复用内存池，代表工作包括 vDNN++、DyMem 和 Coop。
  - *预规划分配（§8.1.2）*：根据张量生命周期提前规划内存布局，代表工作包括 MegTaiChi、MODeL、MP-MoE 和 STAlloc。
  - *运行时碎片管理（§8.1.3）*：在线处理动态分配模式，代表工作包括 GMLake 和 T-Control。
- **临时缓冲区缩减（§8.2）**
  - *算子优化（§8.2.1）*：重新设计高显存占用算子，代表工作包括 MEC、KeOps、FlashAttention、FlashAttention-2、FlashAttention-3 和 BPT。
  - *计算图优化（§8.2.2）*：调整算子顺序、图执行方式或内存布局，代表工作包括 LayRub、MP-MoE、eXLA、MODeL、ROAM 和 Relax。

该类别现收录 [21 份论文文件](papers/Other-Optimizations/)。

### 论文集合概览

仓库目前共包含 **126 份 PDF 文件**。部分工作同时使用多种显存优化机制，因此会被交叉收录；这里统计的是文件数量，而不是去重后的论文数量。

| 分类 | 时间范围 | 文件数 | 目录 |
| --- | ---: | ---: | --- |
| 数据卸载 | 2016-2025 | 19 | [`papers/Data-Offloading`](papers/Data-Offloading/) |
| 激活重计算 | 2016-2026 | 20 | [`papers/Activation-Recomputation`](papers/Activation-Recomputation/) |
| 数据压缩 | 2018-2026 | 31 | [`papers/Data-Compression`](papers/Data-Compression/) |
| 分布式显存优化 | 2017-2025 | 35 | [`papers/Distributed-Memory-Optimization`](papers/Distributed-Memory-Optimization/) |
| 其他优化 | 2017-2026 | 21 | [`papers/Other-Optimizations`](papers/Other-Optimizations/) |

### 如何使用本仓库

1. 先通过分类图定位与你的训练任务相关的核心挑战。
2. 根据图中的章节编号阅读综文对应部分。
3. 进入相应论文目录，先阅读奠基工作，再追踪较新的系统和方法。
4. 从显存节省、计算开销、通信或 I/O 开销、精度影响和实现复杂度五个方面比较方案。
5. 当单一方法无法满足需求时，重点关注组合式显存优化方案。

### 参与贡献

欢迎通过 GitHub Issue 或 Pull Request 提交勘误和新论文建议。添加论文时，请提供年份、发表地点、标题、官方或开放访问链接、代码链接（如有），以及对主要贡献的简要说明。详细分类和记录格式见 [`papers/README.md`](papers/README.md)。

### 引用

如果本综述或仓库对你的研究有所帮助，欢迎引用我们的论文。论文正式发表后，我们将在此补充 BibTeX 信息。

### 版权说明

仓库原创内容遵循 [`LICENSE`](LICENSE) 中的许可协议。所收录论文的版权归作者或出版机构所有，请遵循每篇论文的再分发许可，并优先使用出版社页面、arXiv 页面或作者公开版本。

## Contact / 联系方式

Maintained by the Intelligent Storage and Computing Systems (ISCS) Laboratory, Zhejiang University.

本项目由浙江大学智能存储与计算系统实验室（ISCS Laboratory）维护。

联系：zjuchenping@zju.edu.cn
