---
title: "研究综述｜Agentic Serving、MoE 与 Hybrid Attention（2026）"
pubDatetime: 2026-11-15T16:30:00+08:00
description: "聚焦近期 LLM 推理系统三条主线：Agent 生命周期与 KV 缓存、MoE Expert Parallelism、Hybrid Recurrent State；含 16 项论文与工程对照、社区路线图、研究空白及四周实验方案。"
tags: ["llm-systems", "research-survey", "inference", "moe", "kv-cache"]
featured: true
timezone: Asia/Shanghai
hideEditPost: true
---

这是一份可持续迭代的研究综述（**v0.1，2026-11-15**），面向 LLM 推理系统近期科研选题。

以 **技术重要性、跨模型长期复用价值、真实系统实验可行性** 为主线，整理三个研究方向：

- **A · Agentic Serving**：Agent 生命周期驱动的 KV 保留、恢复、预取及分布式调度。
- **B · MoE Expert Parallelism**：专家副本放置、动态负载均衡与通信计算重叠。
- **E · Hybrid Attention State**：GDN/KDA/Mamba 的状态组合、Replay、Checkpoint、压缩与 Decode 优化。

[**打开完整的交互式 HTML Survey →**](/research/llm-systems-survey-abe-2026.html)

完整报告包含核心论文链接、摘要、证据边界、已覆盖工作、潜在研究问题、实验矩阵和四周预研计划。**这是研究审阅稿；论文结果不等于已经在个人环境复现。**
