# Paper Taxonomy / 论文分类

This directory mirrors the three-level taxonomy in the survey: **classification → challenge → solution category**. Numeric prefixes correspond to section numbers.

本目录与综述中的三级分类完全对应：**技术分类 → 核心挑战 → 解决方案类别**。数字前缀即为综文章节编号。

## §4 Data Offloading Techniques / 数据卸载技术

- **§4.1 Offloading Overhead Mitigation / 卸载开销缓解**
  - `04.1.1-migration-latency-hiding` — Migration Latency Hiding / 迁移延迟隐藏
  - `04.1.2-hybrid-cpu-gpu-execution` — Hybrid CPU–GPU Execution / CPU–GPU 混合执行
  - `04.1.3-joint-memory-optimizations` — Joint Memory Optimizations / 联合内存优化
  - `04.1.4-data-path-optimization` — Data Path Optimization / 数据路径优化

## §5 Activation Recomputation Optimization / 激活重计算优化

- **§5.1 Computational Overhead Control / 计算开销控制**
  - `05.1.1-adaptive-recomputation` — Adaptive Recomputation / 自适应重计算
  - `05.1.2-recomputation-overhead-masking` — Recomputation Overhead Masking / 重计算开销隐藏
- **§5.2 Efficient Strategy Search / 高效策略搜索**
  - `05.2.1-heuristic-based-methods` — Heuristic-Based Methods / 基于启发式的方法
  - `05.2.2-ilp-based-global-methods` — ILP-Based Global Methods / 基于整数线性规划的全局方法

## §6 Data Compression Techniques / 数据压缩技术

- **§6.1 Model Accuracy Preservation / 模型精度保持**
  - `06.1.1-adaptive-low-precision-compression` — Adaptive Low-Precision Compression / 自适应低精度压缩
  - `06.1.2-error-controlled-low-rank-compression` — Error-Controlled Low-Rank Compression / 误差可控的低秩压缩
- **§6.2 Compression Overhead Reduction / 压缩开销降低**
  - `06.2.1-selective-compression` — Selective Compression / 选择性压缩
  - `06.2.2-efficient-compression-algorithms` — Efficient Compression Algorithms / 高效压缩算法

## §7 Distributed GPU Memory Optimization / 分布式 GPU 显存优化

- **§7.1 Data Redundancy Elimination / 数据冗余消除**
  - `07.1.1-fine-grained-data-sharding` — Fine-Grained Data Sharding / 细粒度数据分片
- **§7.2 Prolonged Activation Memory Minimization / 长生命周期激活显存最小化**
  - `07.2.1-scheduling-strategy-optimization` — Scheduling Strategy Optimization / 调度策略优化
  - `07.2.2-integration-of-memory-optimizations` — Integration of Memory Optimizations / 显存优化方法集成
- **§7.3 Memory Load Balance Across Devices / 跨设备显存负载均衡**
  - `07.3.1-load-redistribution` — Load Redistribution / 负载重分配
  - `07.3.2-combination-with-memory-optimizations` — Combination with Memory Optimizations / 与显存优化技术结合
  - `07.3.3-algorithmic-optimization` — Algorithmic Optimization / 算法优化
  - `07.3.4-memory-aware-parallel-orchestration` — Memory-Aware Parallel Orchestration / 显存感知的并行编排

## §8 Other Optimizations / 其他优化

- **§8.1 GPU Memory Defragmentation / GPU 显存碎片整理**
  - `08.1.1-static-partitioning` — Static Partitioning / 静态分区
  - `08.1.2-preplanned-allocation` — Preplanned Allocation / 预规划分配
  - `08.1.3-runtime-fragmentation-management` — Runtime Fragmentation Management / 运行时碎片管理
- **§8.2 Temporary Buffer Reduction / 临时缓冲区缩减**
  - `08.2.1-operator-optimization` — Operator Optimization / 算子优化
  - `08.2.2-computation-graph-optimization` — Computation Graph Optimization / 计算图优化

## Recommended paper record / 推荐论文记录格式

Each paper should be recorded in the leaf directory for its primary contribution. A Markdown record may use:

```markdown
## Paper title

- **Authors:**
- **Venue / Year:**
- **Paper:** [Official or open-access link](https://example.com)
- **Code:** Optional
- **Survey reference:** Citation number and section
- **Keywords:**
- **Summary:** Problem, core idea, and main trade-offs.
```

每篇论文应放入与其主要贡献对应的最末级目录。涉及多个方向时，在记录中添加交叉引用即可，不建议重复保存文件。

## Contribution rules / 收录规则

- Verify metadata and links against an authoritative source.
- Prefer DOI, publisher, arXiv, or author-provided links.
- Keep summaries concise, factual, and neutral.
- Do not upload publisher PDFs without explicit redistribution permission.
- Name redistributable PDFs as `YYYY-FirstAuthor-ShortTitle.pdf`.

- 通过权威来源核对论文元数据和链接；
- 优先使用 DOI、出版社、arXiv 或作者公开版本链接；
- 论文简介应简洁、客观、中立；
- 未获得明确再分发许可时，不上传出版社版本 PDF；
- 可合法公开的 PDF 建议命名为 `YYYY-FirstAuthor-ShortTitle.pdf`。
