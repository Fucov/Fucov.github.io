---
title: "学习日报 · 2026-10-08"
pubDatetime: 2026-10-08T23:59:00+08:00
description: "今日主线高度集中于 MoE 通信与异构 serving runtime：一端开始利用 router/token 语义减少实际传输量，另一端把 AR、DiT、codec、world model 与机器人 session 纳入统一控制面。"
tags: ["learning-daily","llm-systems","moe","kv-cache","training","communication"]
timezone: Asia/Shanghai
hideEditPost: true
---

[← 2026-10-07 ](/posts/learning-daily-2026-10-07/) · [日报目录](/learning/)

> 同步自已有 Notion 日报的速通版。报告日期为研究覆盖日期，时区为北京时间；原有结论与数据保留，未重新检索。 [原始报告与详细附件](https://www.notion.so/3db4db4a892a8039b37bc0f4c359d331)

**结论：**今日主线高度集中于 MoE 通信与异构 serving runtime：一端开始利用 router/token 语义减少实际传输量，另一端把 AR、DiT、codec、world model 与机器人 session 纳入统一控制面。

- **三部分数量：**Inference & Serving 4 · Training & Post-training 3 · Frontier Lab Watch 0
- **Must Read：**[vLLM-Omni Technical Report](https://arxiv.org/abs/2610.09307)（97/S+） · [Expert Coupling](https://arxiv.org/abs/2610.09372)（95/S） · [CoMoE](https://arxiv.org/abs/2610.09424)（94/S）
- **关键结果：**vLLM-Omni 的 Qwen3-TTS fused profile 在 64 并发达 612.1 audio-s/s（默认 234.6）；Expert Coupling 将 All-to-All 时间降低 1.16–2.63×、step time 最多提升 1.41×；CoMoE 在 8×RTX 5090 上较 SGLang 吞吐最多提升 1.46×。
- **趋势 1：**MoE 通信优化从“更快 collective”转向同时读取模型路由相关性与底层 PCIe/host 拓扑。
- **趋势 2：**Serving 调度边界从单一 token decode loop 扩展到 stage、data plane 与长生命周期 session。
- **趋势 3：**内存优化转向结构感知：ResidualQuant 利用跨 loop KV 相似性，MemFerry 在 DHA compute 与搬运之间动态选择。
- **筛除说明：**Frontier Lab Watch 严格空报；Anthropic 当日为科研资助公告，OpenAI 为企业案例。LOCAA、普通 benchmark、应用型 agent 与例行 nightly 均排除；核心 serving/training 项目 release 入口未发现当日可独立入选的重大版本。
- **来源：**[arXiv cs.DC](https://arxiv.org/list/cs.DC/recent) · [arXiv cs.LG](https://arxiv.org/list/cs.LG/recent) · [arXiv cs.CL](https://arxiv.org/list/cs.CL/recent) · [vLLM releases](https://github.com/vllm-project/vllm/releases) · [SGLang releases](https://github.com/sgl-project/sglang/releases) · [TensorRT-LLM releases](https://github.com/NVIDIA/TensorRT-LLM/releases) · [FlashInfer releases](https://github.com/flashinfer-ai/flashinfer/releases)

---

[← 2026-10-07 ](/posts/learning-daily-2026-10-07/) · [日报目录](/learning/)
