---
title: "研究综述｜MoE Expert Parallelism：从负载均衡到动态分布式执行"
pubDatetime: 2026-11-15T22:35:00+08:00
description: "深度梳理 2025–2026 年 15 项 MoE/EP 系统工作：作者、机构、研究脉络、核心方法、原文、开源代码、明确局限与未来选题。"
tags: ["llm-systems", "research-survey", "moe", "expert-parallel", "inference"]
featured: true
timezone: Asia/Shanghai
hideEditPost: true
---

2026 年 MoE Expert Parallelism 的研究正在从静态专家放置、按 Token 数均衡，走向动态复制、每 Step 的低开销求解、真实 GEMM 耗时建模和通信—计算协同。

这是 **v0.1 深度研究稿（2026-11-15）**，详细整理 15 项代表论文及工程实现，逐篇包括作者与署名机构、核心 Insight、系统方法、实验结果、已声明的局限以及可能的后续研究问题。

[**打开宽屏 HTML Survey →**](/research/moe-ep-systems-survey-2026.html)

本文附有作者/实验室研究线索、研究覆盖对照、三个候选选题、实验矩阵与 ACM Computing Surveys 2026 等参考综述。论文结果均注明原作者的条件，不表示已独立复现。
