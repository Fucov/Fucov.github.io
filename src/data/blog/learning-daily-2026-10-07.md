---
title: "学习日报 · 2026-10-07"
pubDatetime: 2026-10-07T23:59:00+08:00
description: " 今日主线是训练—rollout 权重数据平面、Agent KV 生命周期放置与低资源集合通信；共 8 项达到 70 分门槛。"
tags: ["learning-daily","llm-systems","kv-cache","training","communication"]
timezone: Asia/Shanghai
hideEditPost: true
---

[← 2026-10-06 ](/posts/learning-daily-2026-10-06/) · [日报目录](/learning/) · [2026-10-08 →](/posts/learning-daily-2026-10-08/)

> 同步自已有 Notion 日报的速通版。报告日期为研究覆盖日期，时区为北京时间；原有结论与数据保留，未重新检索。 [原始报告与详细附件](https://www.notion.so/3db4db4a892a8039b37bc0f4c359d331)

**结论：** 今日主线是训练—rollout 权重数据平面、Agent KV 生命周期放置与低资源集合通信；共 8 项达到 70 分门槛。
- **三部分数量：** PART I Inference & Serving 4；PART II Training & Post-training 4；PART III Frontier Lab Watch 2 个信号（NVIDIA、Intel，已与主体去重）。
- **Must Read：** [NeMo-DCR](https://arxiv.org/abs/2610.08430)（98/S+）；[NCCL M2N](https://arxiv.org/abs/2610.07516)（96/S+）；[Lachesis](https://arxiv.org/abs/2610.08378)（94/S）。
- **关键结果：** NeMo-DCR 在 30B–1T、3%/5% 变化率下较全 checkpoint 快 12–40×，1T/3% refit 从 87.5 分钟降至 150 秒；NCCL M2N 在 256 GPU DeepSeek-V3 NeMo-RL 中把权重同步从 5.78 秒降至 2.77 秒，step time 降 12.7%；Lachesis 将高带宽闪存寿命延长 1.19–3.13×；T-CCL 接入 vLLM 后端到端吞吐最高提升 1.31×。
- **趋势 1：** RL 权重同步正从框架辅助功能拆成布局编译、传输、应用和事务恢复数据平面。
- **趋势 2：** Agent harness 的 session、sub-agent 与 last-use 语义开始进入 KV cache 控制面。
- **趋势 3：** Collective 的优化目标从裸带宽转向必要字节、SM footprint 与真实 overlap。
- **筛除说明：** Mosaic 为高质量通用 GPU sharing 但 LLM 专属贡献不足；Persistent Memory 主要是测量/消融；FailBench 为 benchmark。vLLM、SGLang、TensorRT-LLM、FlashInfer、LMCache 等官方 release 入口无当日合格重大版本，例行构建不收录。AIHOT 只用于候选发现。
- **来源：** [arXiv cs.DC recent](https://arxiv.org/list/cs.DC/recent) · [NeMo-DCR code](https://github.com/NVIDIA-NeMo/RL/pull/2444) · [DySCo artifact](https://github.com/Large-scale-Sustainable-Computing-LSC/dysco-artifact) · [固定 Archive](https://app.notion.com/p/3db4db4a892a8039b37bc0f4c359d331)

---

[← 2026-10-06 ](/posts/learning-daily-2026-10-06/) · [日报目录](/learning/) · [2026-10-08 →](/posts/learning-daily-2026-10-08/)
