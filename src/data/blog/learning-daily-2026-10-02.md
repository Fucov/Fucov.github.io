---
title: "学习日报 · 2026-10-02"
pubDatetime: 2026-10-02T23:59:00+08:00
description: " 今日共收录 10 项。最强信号是 MoE runtime 全面进入动态化：SGLang 开始无重启切换 P/D 角色，MegaFlux 把热专家复制推入 megakernel，MoE-CORE 与 RapidMoE 分别从驻留缓存和 bit-level residual 重写异构卸载。"
tags: ["learning-daily","llm-systems","moe","kv-cache","training","communication"]
timezone: Asia/Shanghai
hideEditPost: true
---

[日报目录](/learning/) · [2026-10-03 →](/posts/learning-daily-2026-10-03/)

> 同步自已有 Notion 日报的速通版。报告日期为研究覆盖日期，时区为北京时间；原有结论与数据保留，未重新检索。 [原始报告与详细附件](https://www.notion.so/3db4db4a892a8039b37bc0f4c359d331)

**结论：** 今日共收录 10 项。最强信号是 MoE runtime 全面进入动态化：SGLang 开始无重启切换 P/D 角色，MegaFlux 把热专家复制推入 megakernel，MoE-CORE 与 RapidMoE 分别从驻留缓存和 bit-level residual 重写异构卸载。

**数量**
- Inference & Serving：6
- Training & Post-training：4
- Frontier Lab Watch：3（NVIDIA、Huawei Ascend、ByteDance）
### Must Read
1. [SGLang v0.5.21](https://github.com/sgl-project/sglang/releases/tag/v0.5.21) — **96 / S+**。DeepSeek-V4.1 长 prompt 首 token 提升 22%，Kimi K3 的 P/D prefill throughput 提升 20.6%，并支持动态 P/D 角色切换。
2. [MegaFlux](https://arxiv.org/abs/2610.00671) — **95 / S+**。8×B200 上前后向几何平均加速 1.45× / 1.28×，峰值 2.14× / 2.64×；vLLM 端到端 prefill 提升 1.13–1.26×。
3. [Leto](https://arxiv.org/abs/2610.00687) — **94 / S**。6/72×A100 集群上恢复快 3.6–6.5×，productive training time 最高增加 13.7 个百分点。
### 关键结果
- [GPU-Initiated Communication](https://arxiv.org/abs/2610.01380)：最小 GPU 发起路径 issue 约 0.7μs、完成约 4.0μs；3,000 活跃连接可让 All-to-All message rate 下降 59%。
- [MoE-CORE](https://arxiv.org/abs/2610.01950)：在单卡 84GB Ascend NPU 约束下显著压低超大 MoE TPOT；但对照设置与输出长度仍需谨慎解读。
- [RapidMoE](https://arxiv.org/abs/2610.01265)：利用 expert/bit 双层非对称性，decode 最高 3.5×、prefill 最高 2.1×。
- [TACO](https://arxiv.org/abs/2610.02199)：optimizer state 从 27.7GB 降至 0.16GB（174×），峰值显存从 80.6GB 降至 27.5GB。
- [FastCI](https://arxiv.org/abs/2610.01967)：生产环境 CI latency 降低 77.5%，GPU 使用量降低 63.9%，modified-code coverage 增加 3.2%。
### 趋势
1. **Dynamic MoE**：优化单位从模型/层下沉到 expert、tile 和 bit。
2. **Stateful Runtime**：动态 P/D、working-state retention 与 cache provenance 正在成为主路径能力。
3. **Fabric Truth**：queue、doorbell、连接规模和 SM residency 比“GPU 发起”标签本身更决定通信性能。
**筛除说明：** 通用 HPC、区块链、联邦学习、机器人与普通 Agent 应用未进入主体；Leto 与 MegaFlux 因 10-02 首次进入 arXiv 公共列表而收录并标注原始提交时间。LMCache 仅发现同一提交的例行 nightly；vLLM、TensorRT-LLM、FlashInfer、Megatron、DeepSpeed、TorchTitan、verl 与 TRL 未确认当日新的重大正式版本。AIHOT 仅用于候选发现，不作为技术证据。

**来源入口：** [arXiv cs.DC](https://arxiv.org/list/cs.DC/recent) · [arXiv cs.LG](https://arxiv.org/list/cs.LG/recent) · [SGLang Release](https://github.com/sgl-project/sglang/releases/tag/v0.5.21)

---

[日报目录](/learning/) · [2026-10-03 →](/posts/learning-daily-2026-10-03/)
