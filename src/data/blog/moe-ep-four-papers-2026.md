---
title: "LLM 推理系统论文精读：MegaScale-Infer 与 SpecPilot"
pubDatetime: 2026-11-16T23:30:00+08:00
modDatetime: 2026-11-26T20:40:00+08:00
description: "精读 MegaScale-Infer 的 Attention–FFN 分离架构与 SpecPilot 的自主投机解码优化：从资源解耦到算法发现、无损权重压缩与正确性验证。"
tags: ["llm-systems", "research-survey", "inference", "moe", "speculative-decoding", "agentic-optimization"]
featured: true
timezone: Asia/Shanghai
hideEditPost: true
---

## 1. MegaScale-Infer：为什么 Attention 与 FFN 必须放在同一组 GPU 上？

**论文**：[MegaScale-Infer: Efficient Mixture-of-Experts Model Serving with Disaggregated Expert Parallelism](https://doi.org/10.1145/3718958.3750506) · [SIGCOMM 2025 正式版 PDF](https://chaojin0310.github.io/files/papers/SIGCOMM25-MegaScale-Infer.pdf)  
**作者**：Ruidong Zhu、Ziheng Jiang、Chao Jin（共同一作）、Peng Wu、Cesar A. Stuardo 等。  
**机构**：北京大学计算机学院、字节跳动。**代码**：未确认原系统完整公开实现；可参考 [SGLang AF 分离讨论](https://github.com/sgl-project/sglang/issues/9401)。

### 1.1 研究背景与问题定义：为什么 MoE 的稀疏计算没有等比例降低推理成本？

MoE 的优势是每个 Token 只激活少数专家。然而，从模型层面减少 FLOPs，并不意味着 GPU 的执行时间会按相同比例降低。

在 Decode 阶段，Attention 需要读取各个请求独有的 KV Cache。上下文越长，并发请求越多，显存容量与访存带宽的压力就越大；FFN 则可以让不同 Token 共享同一份专家权重，但只有同一个专家收到足够多的 Token，才能摊薄读取权重的成本。

![图 1：MegaScale-Infer 原文 Figure 1](/assets/posts/moe-ep-four-papers-2026/megascale-infer/figure-01-attention-ffn-utilization.png)

**图 1｜Dense、MoE 与分离式 MoE 的 GPU 利用率（MegaScale-Infer 原文 Figure 1）** 左图中 Dense FFN 随 Batch 增长能够较快达到计算饱和；中图中 MoE 的 Token 被分散到不同专家，达到 Batch 上限时 FFN 仍可能没有充分利用 GPU；右图说明拆分资源后，可以扩大专家的有效 Batch，或为不同模块选择性价比更高的硬件。该图解释设计动机，并非具体模型的实测曲线。

关键矛盾是：**Attention 限制整个系统能够容纳的 Batch，而 FFN 恰恰需要更大的 Batch 才能高效执行。** 单纯增加传统完整模型副本，往往同时增加 Attention 与 Expert 权重副本，无法独立控制二者的比例。

MegaScale-Infer 的研究问题是：在不改变模型 Router、不减少专家计算的前提下，能否释放 Attention 对 Expert Batch 的限制，同时控制新增通信和单 Token 延迟？

### 1.2 核心算法：把模型层拆成两个独立资源池

MegaScale-Infer 提出 Disaggregated Expert Parallelism，即在每个 Transformer Layer 内分离 Attention 和 Expert FFN：

- Attention Pool：多个 Attention Replica 保存各自请求的 KV Cache；副本之间进行 DP，副本内部可使用 TP。
- Expert Pool：各 Expert Node 保存专家权重，组成 EP Group；单个专家节点也可以使用 TP。
- 每个 Attention Replica 完成 Attention 与路由后，把中间激活发送到对应 Expert Node；专家计算完成后将结果送回 Attention 侧聚合，进入下一层。

![图 2：MegaScale-Infer 原文 Figure 3](/assets/posts/moe-ep-four-papers-2026/megascale-infer/figure-03-disaggregated-runtime-architecture.png)

**图 2｜MegaScale-Infer 的运行时架构（MegaScale-Infer 原文 Figure 3）** 左边是一组 Attention 副本，各自保存 Attention 权重和 KV Cache；右边是按专家划分的 Expert 节点。中间双向箭头表示两组 GPU 之间的激活值和输出交换。Attention 副本数与 Expert 并行规模不必相等。

这种设计类似 PD 分离，但切分对象不同：

| | PD 分离 | AF 分离（MegaScale-Infer） |
|---|---|---|
| 切分维度 | Prefill / Decode 两个阶段 | 每一层的 Attention / FFN |
| 主要传输 | 阶段交接时的 KV Cache | 每一层的中间激活与专家输出 |
| 优化目的 | 减少长 Prefill 对 Decode 的干扰 | 增大专家有效 Batch，独立配置资源 |
| 新增开销 | 状态迁移与调度 | 高频通信及流水线等待 |

MegaScale-Infer 本身也将 Prefill 与 Decode 分在不同集群，主要研究 Decode 集群内部的 AF 分离。因此两种方法可以同时使用。

### 1.3 核心 Insight：让同一份专家权重服务更多 Token

假设一个模型有 $E$ 个专家，每个 Token 选择 $K$ 个专家，每个 Attention 副本在一个 Microbatch 中处理 $b_a$ 个 Token，总共有 $n_a$ 个 Attention 副本。若专家负载均匀，则：

$$
\overline{b_e}=\frac{n_a b_a K}{E}.
$$

$\overline{b_e}$ 为单个专家平均处理的 Token 数。这个公式是平均量，不保证每个专家的实际负载一致。

举例：$E=8$，$K=2$，单个 Attention 副本一次只处理 128 个 Token。

| Attention 副本数 | 专家池总请求 Token 数 | 每个专家平均 Token 数 |
|---:|---:|---:|
| 1 | 128 | 32 |
| 2 | 256 | 64 |
| 4 | 512 | 128 |
| 8 | 1024 | 256 |

多个 Attention 副本共享 Expert Pool，就能提高 Expert GEMM 的 M 维度，而不需要每个 Attention 副本单独保存所有专家权重。

从 Roofline 的角度理解，若一次 BF16 GEMM 的输入为 $b\times h$、权重为 $h\times d$，暂时只考虑权重读取，则算术强度近似为：

$$
I\approx\frac{2bhd}{2hd}=b\quad\mathrm{FLOP/byte}.
$$

当 $b$ 很小时，GEMM 即使 FLOPs 很少，也可能主要在等待权重读取；增大 $b$ 能让同一份权重参与更多计算，提高计算单元利用率。真实 GEMM 还受到激活、中间结果、量化、Tile Padding 等影响，因此不能把上式当作准确的执行时间模型。

这就是为什么 AF 分离不是简单地把算子放到另一张 GPU：**它改变了专家接收 Token 的规模，使原来受访存限制的计算有机会转向更高的计算利用率。**

### 1.4 算法如何对应问题：Ping-Pong 流水线、部署搜索与 M2N 通信

**（1）Ping-Pong Pipeline：解决两类 GPU 轮流空闲的问题。**

如果只有一个 Batch，Attention 执行时 Expert Pool 空闲，Expert 执行时 Attention Pool 又空闲。直接拆分会让利用率进一步下降。

作者将请求切分成多个 Microbatch，让不同 Microbatch 分别占用 Attention、A2E 通信、Expert 和 E2A 通信阶段。

![图 3：MegaScale-Infer 原文 Figure 4](/assets/posts/moe-ep-four-papers-2026/megascale-infer/figure-04-ping-pong-pipeline.png)

**图 3｜四个 Microbatch 在两个模型层之间的乒乓执行（MegaScale-Infer 原文 Figure 4）** 四行分别表示 Attention、发往 Expert 的通信、Expert 计算和返回 Attention 的通信。横轴是时间。同一个 Microbatch 必须等上一层 Expert 结果返回才能进入下一层，但另一组请求可以利用这段时间执行 Attention。这将原本的等待时间转变为其他请求的计算时间。

论文用 $T_a$、$T_e$、$T_c$ 分别表示单个 Microbatch 的 Attention、Expert 和单向通信时间，令：

$$
T_f=\max(T_a,T_e).
$$

在论文的简化流水线模型中，充分利用流水线需要满足：

$$
T_a\approx T_e,\qquad T_c<T_f,\qquad mT_f\geq 2(T_f+T_c).
$$

其中 $m$ 是 Microbatch 数量。前两条分别约束计算阶段平衡、通信能够被计算覆盖；第三条要求有足够多的独立工作填满管线。

例如，两侧计算各需要 100 μs，单向通信 30 μs，按上述条件至少需要 3 个 Microbatch。如果通信变成 80 μs，则至少需要 4 个。这个计算只解释理想流水线的设计门槛，不意味着真实系统必然达到完全重叠。

**（2）部署配置搜索：解决 Attention 和 Expert 速度不匹配的问题。**

即使 Microbatch 数量足够，如果 Attention 比 Expert 快很多，Expert Pool 仍会成为瓶颈；反之，Expert Pool 将长期等待输入。因此作者将资源配置也纳入优化。

搜索变量包括 Attention 与 Expert 的 TP、Attention 副本数、Microbatch 数及全局 Batch。论文基于算子 profiling、通信带宽和 KV Cache 容量建立预测模型，在时延 SLO 下搜索可容纳的 Batch，并选择单位硬件成本吞吐最高的方案。

用简化的形式表达其目标：

$$
\max\frac{B/T_{\mathrm{total}}}{C_{\mathrm{attention}}+C_{\mathrm{expert}}}
\quad\text{s.t.}\quad
T_{\mathrm{iter}}\leq \mathrm{SLO},\quad
M_{\mathrm{used}}\leq M_{\mathrm{available}}.
$$

这里的 $B$ 表示全局 Batch，$C$ 表示相应资源成本。关键在于同时选择资源比例和工作量，而非先固定机器数量、再独立调节 Batch。

**（3）M2N 通信：解决每层跨节点交换的固定开销。**

AF 分离把传统 EP 的同组 All-to-All 转化为两组设备之间的多对多交换。Attention 路由之后，Token 需要发给动态选中的专家；下一次跨层执行又需要取回结果。

作者采用预注册 GPU Buffer、RDMA Write with Immediate、完成队列轮询及 CUDA Stream 控制，减少中间数据拷贝、通用通信组建立和同步开销。其目标是在每次消息不大的情况下缩短通信延迟，使 Ping-Pong Pipeline 有机会满足 $T_c<T_f$。

三项设计的对应关系如下：

| 原有问题 | 对应方法 | 解决机制 |
|---|---|---|
| 专家 Batch 小、权重复用不足 | AF 分离 | 汇集多个 Attention 副本的 Token |
| 两侧计算互相等待 | Ping-Pong Pipeline | 用其他 Microbatch 填充等待时间 |
| 两侧速度与资源价格不匹配 | 部署搜索与异构 GPU | 根据时延约束调整资源比例 |
| 每层跨节点传输过重 | M2N 通信库 | 降低小消息固定开销与尾延迟 |

### 1.5 实验结果：拆分架构到底增加了吞吐，还是只是增加通信？

**实验设置。** 论文使用 Mixtral 8×22B（141B 参数）、DBRX（132B）与 Scaled-MoE（317B），比较 vLLM 和 TensorRT-LLM。在 8 台各含 8 张 80GB Ampere GPU 的同构集群及 H20/L40S 异构集群上评估；采用 BF16，生产负载的输入、输出长度中位数分别为 571、159 Token。Decode 的 TBT 目标为 150 ms。以下数据均来自正式版第 7 节。

**（1）同构部署：Decode 吞吐明显增长，单 Token 延迟没有同步下降。**

![图 4：MegaScale-Infer 原文 Figure 8](/assets/posts/moe-ep-four-papers-2026/megascale-infer/figure-08-homogeneous-serving-performance.png)

**图 4｜同构 GPU 下的 Decode 吞吐、TBT 与端到端吞吐（MegaScale-Infer 原文 Figure 8）** 左图比较单卡 Decode 吞吐，中图比较生成两个 Token 之间的平均时间，右图把 Prefill 也计入整体吞吐。拆分显著提高 Decode 阶段的吞吐，但中图并没有出现同样幅度的延迟下降，右图的整体收益也更小。

以 Scaled-MoE 为例，相对当时的 TensorRT-LLM 单卡 Decode 吞吐最高为 1.90×；相对该论文的 vLLM 基线为 7.11×（受到当时多节点基线实现限制）。把 Prefill 一并计入后，相对较强基线最高为 1.18×。因此不能将 Decode 的最高收益理解为整个请求加速 1.90×。

**（2）异构部署：根据模块需求选择 GPU，比一味增加高端 GPU 更划算。**

![图 5：MegaScale-Infer 原文 Figure 9](/assets/posts/moe-ep-four-papers-2026/megascale-infer/figure-09-heterogeneous-serving-performance.png)

**图 5｜H20 与 L40S 异构部署的单位成本吞吐（MegaScale-Infer 原文 Figure 9）** Attention 采用显存与带宽性价比较高的 H20，Expert 采用计算性价比较高的 L40S。左图显示 Decode 的单位硬件成本吞吐提高，中图显示延迟仍处于相近范围，右图显示端到端成本效率的改善。

| 异构实验指标 | 相对 H20 上 TensorRT-LLM 等基线的论文结果 |
|---|---:|
| Decode 单位硬件成本吞吐 | 最高 1.86×（TensorRT-LLM） |
| 包含 Prefill 的单位成本吞吐 | 最高 1.66× |
| Decode 单位功耗吞吐 | 最高 1.80× |
| 端到端单位功耗吞吐 | 最高 1.72× |

成本结果依赖论文采用的 GPU 价格与网络配置，不能直接套用其他采购价格或云服务价格。

**（3）通信库是否足够快：小消息和尾延迟是 AF 分离能否成立的前提。**

![图 6：MegaScale-Infer 原文 Figure 11](/assets/posts/moe-ep-four-papers-2026/megascale-infer/figure-11-m2n-communication-performance.png)

**图 6｜M2N 在不同消息大小下的性能（MegaScale-Infer 原文 Figure 11）** 从左到右分别是中位延迟、P99 延迟和吞吐，蓝线表示定制 M2N，橙线表示 NCCL。在消息较小时，降低固定开销的效果最明显；随消息变大，双方吞吐逐渐逼近带宽上限。在 256 KB 的典型消息下，作者报告中位延迟降低 68.2%、P99 降低 92.9%、吞吐提高 4.2×。这些是通信微基准结果，不是整个模型的加速比。

**（4）流水线是不是切得越细越好？**

![图 7：MegaScale-Infer 原文 Figure 14](/assets/posts/moe-ep-four-papers-2026/megascale-infer/figure-14-microbatch-pipeline-ablation.png)

**图 7｜不同 Microbatch 数量下的 Decode 吞吐（MegaScale-Infer 原文 Figure 14）** 从 1 个 Microbatch 增加至 2 个后，两类计算池可以同时处理不同请求；增加到 3 个时，又能覆盖更多通信时间。继续增加到 4 个的收益已经较小。该消融固定每个 Microbatch 的大小，并非固定总 Batch。

![图 8：MegaScale-Infer 原文 Figure 15](/assets/posts/moe-ep-four-papers-2026/megascale-infer/figure-15-attention-dp-scaling.png)

**图 8｜Attention DP 数量对 DBRX 推理性能的影响（MegaScale-Infer 原文 Figure 15）** 左图为单 Token 延迟，右图为单卡吞吐。当 Attention 副本较少时，Expert Pool 等待 Attention 输出；DP 增至 8 时吞吐达到峰值；继续增至 12 或 16，Expert 侧成为瓶颈，延迟上升、单卡吞吐反而下降。

**实验结论。** 该工作最重要的证据不是孤立的通信加速数字，而是 Figure 14、15 展示的资源平衡关系：AF 分离只有在两侧工作量、Microbatch 数和通信延迟相匹配时才具有优势。新架构需要一套新的部署规划逻辑才能充分发挥作用。

### 1.6 现有局限与值得进一步研究的问题

**作者讨论的局限与部署经验。** AF 分离的新增通信不能消除单个 Microbatch 的数据依赖，流水线主要改善整体吞吐，而非直接降低每个请求的延迟（§7.2）。同构实验在计算受限的 Prefill 上没有明显收益。论文也讨论了不同专家负载和消息粒度对执行方式的影响（§5、§6、§8）。

**潜在方向一：动态负载下的 AF Resource Rebalancing。** 原论文在给定负载参数后搜索部署配置，但服务中的上下文长度、活跃请求数与输出速度会动态变化。Attention 的访存时间与 Expert 的实际计算量并不会等比例变化。可以研究何时需要改变两侧分配的执行资源，以及如何控制重配置和 KV Cache 迁移开销。实验应先证明固定配置在真实非稳态负载下出现稳定的吞吐损失或 SLO 违约，再验证动态方案能否弥补。

**潜在方向二：异步流水线中的专家拖尾。** 当某个 Microbatch 的热点专家形成 Straggler 时，Attention Pool 可能没有按计划收到结果，原本能隐藏的通信窗口也随之失效。这里真正值得研究的是 Token-to-Replica 分配如何影响整个跨池流水线，而不只是某一张 GPU 的 GEMM 耗时。可以建立专家执行时间模型，比较单层 makespan 最优与全流水线吞吐最优是否存在系统性的差异。

---

## 2. SpecPilot：让 Agent 自主寻找投机解码的性能上限

**论文**：[SpecPilot: Autonomous Algorithm Discovery and Systems Optimization for Speculative Decoding](https://arxiv.org/abs/2611.02501) · [官方项目页](https://stargazerx0.github.io/specpilot-project-page/) · [开源代码](https://github.com/StargazerX0/SpecPilot)  
**作者**：Kunjun Li、Wenhao Chai、Yihao Liang（共同一作）、Yuezhou Hu、Shuyao Xu 等。  
**机构**：UC Berkeley、Princeton University、Stanford University、NVIDIA。2026 年 11 月预印本。

### 2.1 研究背景与问题定义：为什么投机解码已经很快，却仍有如此大的优化空间？

普通自回归解码（AR）每次执行 Target Model 生成一个 Token。对于低并发的大模型，GPU 大量时间花在从 HBM 读取权重上。投机解码先由较小的 Draft Model 提出若干候选，再由 Target 一次验证多个位置。如果候选正确，一次读取权重就能产生多个输出 Token。

DFlash2、DSpark 等并行 Drafter 已经能一次生成整个候选块。但部署条件不同，瓶颈也不同：单请求可能由权重流量限制，多请求则可能受 Attention、循环状态更新和计算量限制。手工为每一种模型、GPU、Drafter 和 Batch Size 搜索算法及 Kernel，成本很高。

![图 9：投机解码的三个性能与正确性问题](https://stargazerx0.github.io/specpilot-project-page/assets/figures/fig_motiv.svg)

**图 9｜SpecPilot 的研究动机（官方项目图）** 左图分解一次解码迭代的耗时，中图展示并发增加后原版投机解码的加速下降，右图说明普通任务准确率不足以识别某些错误的数值优化。

研究目标是：给定一个已可运行的推理引擎，让 Agent 根据实测瓶颈提出算法和系统修改，持续提升生成速度，同时约束输出正确性，而不是仅仅自动调整若干启动参数。

### 2.2 核心算法：以性能上限驱动的自主优化闭环

作者把每轮优化分解为性能分析、并行探索、集成、Kernel 优化与最终验证。输入是现有 SGLang 部署、模型、硬件及工作负载；输出是一组经过实验确认的代码补丁与配置。

![图 10：SpecPilot 的完整优化闭环](https://stargazerx0.github.io/specpilot-project-page/assets/figures/fig_agent.svg)

**图 10｜一轮 SpecPilot 如何运行（官方项目图）** 首先量化当前系统与理论上限的差距；再将不同候选方案分发给独立 GPU 上的子 Agent；通过测试的补丁被组合，性能较差的 Kernel 再由专门的 Agent 优化；最后用隔离的正确性及性能测试封存结果，供下一轮使用。

| 环节 | 具体职责 |
|---|---|
| Master Agent | 根据性能模型和已有经验选择研究方向 |
| Exploration Agent | 修改推理引擎，独立实现并测试候选机制 |
| Integration Agent | 将独立有效或互补的小优化组合 |
| Kernel Agent | 优先优化离计算/访存下限最远的 Kernel |
| 独立 Controller | 固定基准、统计检验、正确性验证与晋升决策 |
| Structured Memory | 存储可重放的代码差异、依赖、适用条件和实测证据 |

这里最重要的设计是 Controller 独立于负责代码修改的 Agent。候选补丁必须通过配对性能实验，并在未用于搜索的请求上确认收益，避免 Agent 通过改变测试方法获得虚假的加速。

### 2.3 核心 Insight：把投机解码优化拆成两个可以度量的目标

设一次验证涉及 $N$ 行，$\tau(N)$ 是平均每步接受的 Token 数，$T_{\mathrm{step}}(N)$ 是完整投机迭代时间，$T_{\mathrm{AR}}$ 是普通 AR 单步时间，则加速比为：

$$
S(N)=\frac{\tau(N)}{c(N)},\qquad
c(N)=\frac{T_{\mathrm{step}}(N)}{T_{\mathrm{AR}}}.
$$

因此优化有两个基本方向：增加一次迭代真正生成的 Token 数，或者降低这次迭代的实际成本。验证宽度本身不是目标，接受率也不能单独代表系统加速。

例如在作者的单请求实测中，验证相关 GEMM 从 1 行增加到 64 行，耗时只由 17.88 ms 增至 18.58 ms。这意味着低并发时，额外候选验证可能只增加很少的成本；但在高并发下，额外行需要更多 Attention、状态和矩阵计算，原来的规律不一定成立。

SpecPilot 用性能下限估计各 Kernel 的改进空间。简化的 Roofline 表达为：

$$
T_{\min}\approx
\max\left(\frac{F}{P_{\mathrm{eff}}},\frac{D}{B_{\mathrm{eff}}}\right).
$$

其中 $F$ 是计算量，$D$ 是数据传输量，$P_{\mathrm{eff}}$ 和 $B_{\mathrm{eff}}$ 是可实现的有效计算吞吐与带宽。将实测 Kernel 时间、候选树接受长度上限以及 Profile 之外的耗时分开评估，就能决定下一轮主要搜索算法还是底层实现。

### 2.4 算法如何对应问题：Agent 实际发现了什么？

**（1）Best-first Tree：同一次验证可以覆盖多条候选路径。**

原版并行 Drafter 通常只验证一条最优候选链，一旦某个位置不匹配，后续候选无法被直接接受。Best-first Tree 则按路径分数选择具有较高价值的候选节点，让 Target 在有限验证预算内覆盖多条可能路径。

作者的离线路由回放中，原版 8 位置链平均接受 4.25 个 Token；32 节点的 Best-first Tree 将同一候选集合提高到 5.38，扩大 Draft 长度后达到 5.85。

对于 Qwen3.8-27B 一类混合架构模型，Gated DeltaNet 还维护循环状态。不同树分支不能简单共享错误的状态快照，因此需要沿各自根路径重建状态，并使用 WY Verify Kernel 加速多路径验证。

**（2）W12：不损失 BF16 权重信息，也能减少权重流量。**

![图 11：W12 权重编码及寄存器内解码](https://stargazerx0.github.io/specpilot-project-page/assets/figures/fig3_w12.svg)

**图 11｜W12 的权重存储与计算（官方项目图）** 左图展示 BF16 的符号、指数和尾数如何变成三个 4-bit Plane；右图展示 GEMM 读取压缩数据后直接在寄存器内还原权重，再送入 Tensor Core，避免先将全部解压权重写回 HBM。

标准 BF16 使用 1 bit 符号、8 bit 指数和 7 bit 尾数。作者观察到，同一输出通道内许多权重的指数集中在较窄范围。W12 为通道选择覆盖最多权重的 16 个连续指数，将常见权重表示为符号、4-bit 指数偏移和完整尾数；超出范围的少数权重由 Escape List 精确恢复。

$$
1+4+7=12\ \mathrm{bits}.
$$

理想情况下相较 16-bit 存储可减少 25% 的权重流量。作者对八个 Dense 模型测得实际平均为 12.01–12.02 bits/weight；在 Qwen3.8-27B 已转换层中，权重占用由 44.67 GB 降至 33.58 GB。加载时逐 Tensor 检查还原权重与 Checkpoint 的 bitwise 一致性。

这是**无损编码而非普通有损量化**。编码无损不等于不同 GEMM 实现的浮点累加顺序完全一致，因此仍需进行输出正确性验证。

**（3）Two-level Verification（TLV）：让一次精确验证覆盖两轮 Draft。**

![图 12：候选树与两级验证](https://stargazerx0.github.io/specpilot-project-page/assets/figures/fig_tree_tlv.svg)

**图 12｜从单链验证到树验证和两级验证（官方项目图）** 左图展示候选树如何覆盖原本在贪心链中被舍弃的路径；右图展示 Q Pass 先提出一条可能的链，再从链尾生成新的候选子树，最终用一次精确验证检查全部候选。

W12 的部分编码可以构成低精度的 8-bit View，不需要另外常驻一份低精度模型。TLV 首先用较低成本的 Q Pass 预测可能的接受路径，随后 Drafter 从预测链尾继续生成。最终只有完整权重的 Target Pass 有权提交输出。

核心是将精确权重读取的开销摊销到更长的有效输出，而不是让近似模型决定最终 Token。如果 Q Pass 的预测不正确，仍由精确验证决定合法接受前缀。

**（4）Batched Serving：根据并发状态限制验证成本。**

当并发增加时，继续扩大树和增加一次 Q Pass 可能让计算超出原本可隐藏的时间。Agent 因此发现共享节点预算、按近期接受长度调整 Draft 深度、在合适的调度条件下混合新 Prompt 与 Verify 等机制。官方报告在 FP8 Batched Target 上某种 24 节点树使性能下降 14.9%，BF16 Batched 配置中的 TLV 下降 13%。

**（5）Token Forcing 与经验复用：保证优化没有走偏。**

![图 13：逐 Token 正确性检测](https://stargazerx0.github.io/specpilot-project-page/assets/figures/fig_forcing.svg)

**图 13｜Token Forcing 能识别任务准确率看不出的错误（官方项目图）** 柱状图比较正常优化与人为注入 Logit 偏差的系统，误差线给出置信区间，虚线表示作者采用的拒绝阈值。

将优化系统输出的前缀重新送入原版 Target，逐 Token 比较 scorer 的 argmax。原版系统本身受浮点舍入影响也存在少量不一致，因此作者设置相对原版的统计阈值，并单独检查无法解释的大 Margin 错误。测试中正常系统的 mismatch rate 与 stock 接近，而错误系统约达 2.0% 和 2.4%。

性能实验同样设置晋升门槛：配对收益的 95% 置信下界必须超过 1% 加两倍基线测量噪声。成功机制被保存成可重放、带依赖关系的代码差异，而不是仅保留自然语言建议。官方项目页报告积累了 140 条经验记录。

### 2.5 实验结果：自主优化能否持续超过人工实现？

**实验设置。** 论文与项目页报告使用 SGLang，在单张 H100 SXM5 或 RTX PRO 6000 Blackwell 上评估。任务包括 GSM8K、MATH500、HumanEval、MBPP、MT-Bench；采用 greedy 解码。Stock 表示针对当前环境调优后的原版 Drafter 与引擎，AR 表示关闭投机解码的参照。下表采用官方项目页公布的八组结果。

![图 14：单请求投机解码相对于 AR 的加速](https://stargazerx0.github.io/specpilot-project-page/assets/figures/fig1_speedup_wide.svg)

**图 14｜单请求条件下的整体加速（官方项目图）** 每组柱子比较相同部署的 Stock 和 SpecPilot 相对 AR 的加速比。四组单请求配置均超过 Stock 的两倍速度提升水平；这里的 10.87× 是相对 AR，并不是相对 Stock。

| GPU / 场景 | 模型与 Drafter | Stock / AR | SpecPilot / AR | 封存测试相对 Stock |
|---|---|---:|---:|---:|
| H100 单请求 | Qwen3.8-27B / DFlash2 | 3.95× | 8.86× | 2.13× |
| H100 单请求 | Qwen3.8-27B / DSpark | 3.74× | 9.41× | 2.38× |
| RTX PRO 6000 单请求 | Qwen3.8-27B / DFlash2 | 4.46× | 10.87× | 2.44× |
| RTX PRO 6000 单请求 | Muse-Glimmer-30B / DFlash2 | 4.69× | 10.63× | 2.21× |
| H100 Batching | Qwen3.8-27B FP8 / DFlash2 | 2.02× | 3.90× | 1.85× |
| H100 Batching | Qwen3.8-27B FP8 / DSpark | 1.94× | 3.93× | 1.94× |
| H100 Batching | Qwen3.5-4B / DFlash2 | 1.51× | 3.77× | 2.39× |
| H100 Batching | Gemma-4-12B / EAGLE-3 | 0.68× | 2.09× | 3.11× |

AR 比较使用开发请求上的交替实验，封存收益使用独立未见请求，不能直接将前两列相除当作最后一列的精确值。官方项目页还报告，相比加入 DDTree 的更强基线，H100 上有 1.84× 的提升。

![图 15：三轮自主优化的吞吐演进](https://stargazerx0.github.io/specpilot-project-page/assets/figures/fig_evolution.svg)

**图 15｜不同优化轮次的累积收益（官方项目图）** 横轴是优化进程，纵轴是单请求 Qwen3.8-27B 在 H100 上的生成吞吐。标记点对应被采纳的新机制；这张图体现的是多次探索与集成后获得的累积结果，而非一个补丁的独立贡献。

![图 16：W12 在普通 AR 解码中的收益](https://stargazerx0.github.io/specpilot-project-page/assets/figures/fig_w12.svg)

**图 16｜W12 的独立性能价值（官方项目图）** 三个子图分别比较普通 AR 解码速度、权重占用以及 W12 GEMM 相对 cuBLAS 的速度。项目报告四个 8B–30B 模型在无投机解码条件下，单请求提升 1.19–1.28×，32 请求提升 1.13–1.20×。Qwen3.8-27B 的投影 GEMM 在 1 行和 32 行时分别报告 1.19–1.47× 与 1.17–1.35× 的加速。

实验说明这种闭环既可以改进原有投机策略，也可以产生独立的底层机制；但是不同负载下被采纳的机制并不相同。不能把单请求最优配置直接作为高并发部署的默认配置。

### 2.6 现有局限与值得进一步研究的问题

**适用范围。** 当前公开核心结果以单 GPU、greedy 解码和有限模型/任务组合为主。Token Forcing 提供比任务准确率更严格的经验验证，但不是对任意输入、随机采样分布及所有精度实现的形式化等价证明。W12 的收益也依赖权重指数分布、硬件指令支持和访存瓶颈；多节点 EP 的通信与拖尾问题未被上述单卡实验充分覆盖。

**研究方向一：跨部署复用优化经验。** 将经验从模型名和 GPU 型号抽象为性能特征，例如算术强度、验证行数、接受长度、通信比例与状态更新成本。研究能否用少量在线测量判断某个机制可否迁移，降低重新搜索所需的 GPU 时间，并在未见模型上比较冷启动搜索与记忆迁移的收益。

**研究方向二：将自主搜索扩展至分布式推理。** 单卡的验证树预算可能在多卡 EP 上产生新的专家热点，AF 分离又会引入跨资源池流水线等待。可以联合考虑候选节点的接受概率、专家激活成本与通信关键路径，在端到端吞吐及延迟约束下搜索算法。先使用固定基线验证这些相互作用，再考虑自动代码生成。

**研究方向三：将正确性测试纳入系统优化基础设施。** 对连续批处理、混合精度、随机采样和循环状态模型分别定义测试契约；把输出差异、吞吐收益和搜索资源开销同时计入评价。这样才能判断 Agent 获得的改进是否适合长期维护，而不仅是某组 Benchmark 的最高分。
