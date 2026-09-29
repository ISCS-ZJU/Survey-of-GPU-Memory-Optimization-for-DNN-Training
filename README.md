# Survey of GPU Memory Optimization for DNN Training

> This survey systematically reviews representative research from the past decade on memory-efficient GPU utilization and optimization in DNN training frameworks. It also provides a curated collection of related papers for convenient reference, download, and study.

**English** | [中文](README_zh-CN.md)

![Overview of GPU memory optimization techniques for DNN training](assets/gpu-memory-optimization-taxonomy.png)

## Overview

GPU memory has become a critical resource bottleneck in modern deep neural network (DNN) training. Model parameters, optimizer states, gradients, activations, communication buffers, and temporary tensors must all share limited GPU memory. When a model or its training states exceed available capacity, training frameworks commonly employ memory optimization techniques such as recomputation, offloading, compression, and memory reuse. These techniques make different trade-offs among memory savings, computational overhead, data movement, and training performance, and their effectiveness varies across models, hardware platforms, and training scenarios.

This repository accompanies our **survey of GPU memory optimization for DNN training** and provides three main resources:

1. a hierarchical taxonomy covering **technique classification, key challenges, solution categories, and representative works**;
2. representative papers and related materials corresponding to the survey;
3. a comparison of the strengths, limitations, and applicable scenarios of different techniques in terms of memory savings, computational overhead, communication overhead, accuracy impact, and system complexity.

The figure above summarizes the organization of the survey. The first column presents five major classes of GPU memory optimization techniques, the second identifies their key challenges, the third groups the corresponding solution categories, and the fourth lists representative works discussed in the survey.

## Five Categories of GPU Memory Optimization

### 1. Data Offloading Techniques (§4)

Data offloading expands effective GPU memory capacity by moving tensors among storage tiers such as GPU memory, CPU memory, and NVMe SSDs. Its main challenge is that data transfers can stall computation and reduce training throughput. The survey discusses four directions for mitigating offloading overhead:

- **Migration Latency Hiding (§4.1.1):** overlap computation with data transfers and prefetch tensors before they are needed.
- **Hybrid CPU-GPU Execution (§4.1.2):** allow CPU and GPU resources to participate jointly in training computation.
- **Joint Memory Optimizations (§4.1.3):** combine swapping with recomputation, compression, memory allocation, or scheduling.
- **Data Path Optimization (§4.1.4):** optimize the memory hierarchy and transfer paths among GPU memory, host memory, and storage devices.

This category currently contains [19 paper files](papers/Data-Offloading/).

### 2. Activation Recomputation Optimization (§5)

Activation recomputation discards selected intermediate results during the forward pass and regenerates them when needed during backpropagation, trading additional computation for lower memory usage. The survey focuses on two central questions: how to control recomputation overhead and how to search efficiently for recomputation strategies.

- **Adaptive Recomputation (§5.1.1):** dynamically adjust the strategy according to memory pressure, execution state, or pipeline behavior.
- **Recomputation Overhead Masking (§5.1.2):** hide additional computation by exploiting pipeline idle time or concurrent execution.
- **Heuristic-Based Methods (§5.2.1):** use practical heuristics to search for low-cost checkpointing strategies.
- **ILP-Based Global Methods (§5.2.2):** formulate recomputation as a global optimization problem using integer linear programming.

This category currently contains [20 paper files](papers/Activation-Recomputation/).

### 3. Data Compression Techniques (§6)

Data compression reduces the memory footprint of activations, gradients, optimizer states, or model data through low precision, quantization, sparsification, or low-rank representations. An effective compression method must reduce memory consumption while controlling encoding and decoding costs and preserving model convergence and accuracy.

- **Model Accuracy Preservation (§6.1)**
  - *Adaptive Low-Precision Compression (§6.1.1):* select suitable numerical formats and precision levels for different data.
  - *Error-Controlled Low-Rank Compression (§6.1.2):* reduce data rank while controlling approximation error.
- **Compression Overhead Reduction (§6.2)**
  - *Selective Compression (§6.2.1):* compress only tensors that provide a favorable memory-saving benefit.
  - *Efficient Compression Algorithms (§6.2.2):* reduce metadata, format-conversion, and runtime overhead.

This category currently contains [31 paper files](papers/Data-Compression/).

### 4. Distributed GPU Memory Optimization (§7)

Distributed training provides an important way to overcome the memory capacity of a single GPU by sharding model states across multiple devices. At the same time, it introduces new memory-management challenges.

- **Data Redundancy Elimination (§7.1):** shard parameters, gradients, and optimizer states at a fine granularity.
- **Prolonged Activation Memory Minimization (§7.2):** shorten activation lifetimes through scheduling and integrated memory optimizations.
- **Memory Load Balance Across Devices (§7.3):** address imbalance through load redistribution, integration with other memory optimizations, algorithmic design, or memory-aware parallel orchestration.

This category currently contains [35 paper files](papers/Distributed-Memory-Optimization/).

### 5. Other Optimizations (§8)

Some memory bottlenecks are not caused solely by model states or activations. Memory fragmentation can leave free GPU memory unusable, while operator workspaces and intermediate tensors can create high transient memory peaks.

- **GPU Memory Defragmentation (§8.1)**
  - *Static Partitioning (§8.1.1):* divide GPU memory in advance and assign relatively fixed regions to different tensor types or tasks, reducing interference among different allocation demands.
  - *Preplanned Allocation (§8.1.2):* plan tensor placement and reuse ahead of execution, using known tensor sizes and lifetimes to determine memory addresses and reuse relationships and thereby reduce peak memory usage.
  - *Runtime Fragmentation Management (§8.1.3):* dynamically handle fragmentation generated during execution by merging, migrating, or reorganizing free memory that cannot be predicted in advance.
- **Temporary Buffer Reduction (§8.2)**
  - *Operator Optimization (§8.2.1):* redesign operators with high memory requirements.
  - *Computation Graph Optimization (§8.2.2):* adjust operator ordering, graph execution, or memory layout.

This category currently contains [21 paper files](papers/Other-Optimizations/).

## Paper Collection Overview

The repository currently contains **126 PDF files**. Some works combine multiple memory optimization mechanisms and are therefore included in more than one category.

| Category | Coverage | Files | Directory |
| --- | ---: | ---: | --- |
| Data Offloading | 2016-2025 | 19 | [`papers/Data-Offloading`](papers/Data-Offloading/) |
| Activation Recomputation | 2016-2026 | 20 | [`papers/Activation-Recomputation`](papers/Activation-Recomputation/) |
| Data Compression | 2018-2026 | 31 | [`papers/Data-Compression`](papers/Data-Compression/) |
| Distributed Memory Optimization | 2017-2025 | 35 | [`papers/Distributed-Memory-Optimization`](papers/Distributed-Memory-Optimization/) |
| Other Optimizations | 2017-2026 | 21 | [`papers/Other-Optimizations`](papers/Other-Optimizations/) |

## Contributing

Corrections and suggestions for additional papers are welcome through GitHub issues or pull requests. When adding a paper, please provide its publication year, venue, title, official or open-access link, code repository if available, and a brief description of its primary contribution. See [`papers/README.md`](papers/README.md) for the detailed taxonomy and contribution format.

## Citation

If this survey or repository is useful to your work, please cite our paper. The BibTeX entry will be added after publication.

## Copyright Notice

Repository-authored content is provided under the license in [`LICENSE`](LICENSE). Copyright of the collected papers remains with their authors or publishers. Please comply with the redistribution terms of each paper and prefer official publisher pages, arXiv records, or author-provided open-access versions.

## Contact

This project is maintained by the Intelligent Storage and Computing Systems (ISCS) Laboratory at Zhejiang University.

Email: zjuchenping@zju.edu.cn

## Acknowledgements

We thank the co-authors of this survey—Shuibing He, Nan Zhang, Ping Chen, Weixu Zong, Yi Zhang, and Siling Yang—for their contributions and support in writing the paper, curating the literature, and participating in research discussions.

We also thank the authors and research teams whose papers are cited in this survey for their important contributions to DNN training and GPU memory optimization. Their work forms the foundation of the systematic review and synthesis presented here.

Finally, we thank all readers and contributors who follow, use, and help improve this project. If you notice a missing paper, an inaccurate classification, or any other issue, please contact us by email or open a GitHub issue.
