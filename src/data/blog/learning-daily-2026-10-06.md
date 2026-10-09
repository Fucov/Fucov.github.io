---
title: "学习日报 · 2026-10-06"
pubDatetime: 2026-10-06T23:59:00+08:00
description: "今日值得重点追踪的是“更细粒度的共享与控制”：MOLT 将推理—微调内存让渡细化到单个 activation；SyclKittens 与 RESOLVE 分别补齐 kernel agent 的高性能接口与强验证层；端侧 3B LLM 完整训练已在手机上具备现实可行性。"
tags: ["learning-daily","llm-systems","training"]
timezone: Asia/Shanghai
hideEditPost: true
---

[← 2026-10-05 ](/posts/learning-daily-2026-10-05/) · [日报目录](/learning/) · [2026-10-07 →](/posts/learning-daily-2026-10-07/)

> 同步自已有 Notion 日报的速通版。报告日期为研究覆盖日期，时区为北京时间；原有结论与数据保留，未重新检索。 [原始报告与详细附件](https://www.notion.so/3db4db4a892a8039b37bc0f4c359d331)

**结论：**今日值得重点追踪的是“更细粒度的共享与控制”：MOLT 将推理—微调内存让渡细化到单个 activation；SyclKittens 与 RESOLVE 分别补齐 kernel agent 的高性能接口与强验证层；端侧 3B LLM 完整训练已在手机上具备现实可行性。
**数量：**PART I Inference & Serving 6；PART II Training & Post-training 3；PART III Frontier Lab Watch 2 个去重信号（Microsoft Research、Intel）。
**Must Read：**[MOLT](https://arxiv.org/abs/2610.05748)（96/S+） · [SyclKittens](https://arxiv.org/abs/2610.04277)（95/S+） · [Smartphone Fine-Tuning](https://arxiv.org/abs/2610.06325)（94/S）
**关键结果：**
- MOLT：24B–70B、H100/B200 上维持 ≥99.7% 推理 SLO，微调产出为 discard-based sharing 的 1.9–3.3×。
- SyclKittens：agent 生成 GEMM 从 oneDNN 的 57.5% 提升到 82.1%；完整套件约 96%，Llama-3.1-8B 推理单卡 1.59×、多卡最高 2.91×。
- 手机 3B 微调：修复 MLX backward kernel 后完整 adapter 训练 1.47× 加速，净能耗从 39.3 kJ 降至 26.2 kJ。
**趋势：**
1. 共置弹性从作业/step 级下沉到 activation 等可证明安全的最小对象。
2. Kernel agent 正形成“生成接口—性能搜索—形式/动态验证”的完整基础设施。
3. 端侧训练的决定性瓶颈是量化 backward、热稳定和 OS/runtime 支持，而非 adapter 参数量本身。
**筛除：**LearnSched 缺真实端到端部署与复现资产；Nexus 更偏通用 agent execution fabric；纯并行策略 characterization 缺可复现实物；普通应用、产品更新和例行 nightly 未进入主体。所有条目按 2026-10-06 首次公共列表收录，并保留其更早 v1 提交日期。
**来源：**[arXiv cs.DC](https://arxiv.org/list/cs.DC/recent) · [arXiv cs.LG](https://arxiv.org/list/cs.LG/recent) · [vLLM Releases](https://github.com/vllm-project/vllm/releases) · [SGLang Releases](https://github.com/sgl-project/sglang/releases) · [TensorRT-LLM Releases](https://github.com/NVIDIA/TensorRT-LLM/releases) · [FlashInfer Releases](https://github.com/flashinfer-ai/flashinfer/releases)

---

[← 2026-10-05 ](/posts/learning-daily-2026-10-05/) · [日报目录](/learning/) · [2026-10-07 →](/posts/learning-daily-2026-10-07/)
