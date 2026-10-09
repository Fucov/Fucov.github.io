---
title: "学习日报 · 2026-10-05"
pubDatetime: 2026-10-05T23:59:00+08:00
description: "今日共收录 8 项。推理侧把动态性下沉到 rail、expert placement、KV/state admission 与统一内存；训练侧开始围绕“解锁 learner 的 group 完成时间”和跨设施排队现实设计系统。"
tags: ["learning-daily","llm-systems","moe","kv-cache","training"]
timezone: Asia/Shanghai
hideEditPost: true
---

[← 2026-10-04 ](/posts/learning-daily-2026-10-04/) · [日报目录](/learning/) · [2026-10-06 →](/posts/learning-daily-2026-10-06/)

> 同步自已有 Notion 日报的速通版。报告日期为研究覆盖日期，时区为北京时间；原有结论与数据保留，未重新检索。 [原始报告与详细附件](https://www.notion.so/3db4db4a892a8039b37bc0f4c359d331)

**结论：**今日共收录 8 项。推理侧把动态性下沉到 rail、expert placement、KV/state admission 与统一内存；训练侧开始围绕“解锁 learner 的 group 完成时间”和跨设施排队现实设计系统。

- **三部分数量：**Inference & Serving 6；Training & Post-training 2；Frontier Lab Watch 0（Meta 参与 Coda，已在 Part I 计入，不重复计数）。
- **Must Read：**[vLLM v0.31.0](https://github.com/vllm-project/vllm/releases/tag/v0.31.0) · [RailWave](https://arxiv.org/abs/2610.03415) · [VenusRL](https://arxiv.org/abs/2610.03286)
- **关键结果：**RailWave 在 32×H800/H20 上相对 Native 的 P50 最高加速 5.84×/4.36×；VenusRL 端到端训练最高加速 4.24×、环境成本最高降低 89%；Coda 平均将 output-token / SLO-compliant throughput 提升 20.3% / 70.5%；AFORE 在 110B MoE 上将 throughput 提升 10.1–17.6%，P95 ITL 降低 7.1–9.5%。
- **趋势：**① state/KV 成本成为准入与恢复的一等输入；② MoE 优化从 router 下沉到 rail 与 AFD 流水窗口；③ Agentic RL 从平均 GPU 利用率转向 group-critical-path。
- **筛除说明：**D2K-Bench 属诊断 benchmark、无独立系统贡献；WakeKV 缺少可审计吞吐数字、机构与代码，证据不足。10-05 arXiv 条目均为当日首次公共列表、v1 于 10-02 UTC 提交；AIHOT 仅用于发现。
- **来源：**[arXiv cs.DC recent](https://arxiv.org/list/cs.DC/recent) · [vLLM Release](https://github.com/vllm-project/vllm/releases/tag/v0.31.0) · [RailWave Code](https://github.com/CyberSecurityErial/RailWave-EP)

---

[← 2026-10-04 ](/posts/learning-daily-2026-10-04/) · [日报目录](/learning/) · [2026-10-06 →](/posts/learning-daily-2026-10-06/)
