# AI / Forward Deployed Engineer: Interview Guide

Tap a question to expand. Prepared by Rayudu Addagarla. Numbers are rules of thumb; always validate with load tests on your own stack.

## Your five extra questions

A. Selecting GPU capacity for concurrent users and latency targets

**1. Memory first.** Weights = params × bytes (70B at FP16 ≈ 140 GB; FP8 ≈ 70 GB; 4-bit ≈ 35 GB).

**2. KV cache budget.** Per token = 2 × layers × KV heads × head_dim × bytes. Concurrent sequences ≈ (usable GPU memory − weights) ÷ (KV per token × avg context). GQA/MQA and FP8 KV cache raise concurrency.

**3. Latency.** Prefill is compute-bound and sets TTFT. Decode is memory-bandwidth-bound: per-user tokens/s is roughly bandwidth ÷ bytes read per step, and batching amortizes weight reads at the cost of per-user speed.

**4. Size from SLOs.** Define p95 TTFT and inter-token latency, expected input/output lengths and peak QPS (Little's law: concurrency = QPS × latency). Benchmark one replica with vLLM/TensorRT-LLM/SGLang, find the max concurrency that still meets the SLO, then replicas = peak concurrency ÷ that number, plus 20–30% headroom. Use tensor parallelism when the model doesn't fit or decode is too slow, and prefix caching for shared prompts.

B. Which model, and reasoning vs non-reasoning

**Non-reasoning (fast, cheap):** classification, extraction, routing, summarization, chat, anything with a tight latency budget or clear instructions.

**Reasoning:** multi-step math/logic, hard debugging, planning, ambiguous or high-stakes analysis where extra thinking tokens measurably improve accuracy. Cost: more tokens, higher latency, and extra variance in time.

**Method:** build an eval set, test the smallest model first, move up only where it fails, route by task difficulty, and set a thinking budget rather than defaulting to max.

C. Increasing recall in RAG, and what if it still fails

**Recall levers:** hybrid search (BM25 + dense); better chunking (semantic, with overlap, parent-document retrieval); contextual chunk headers; query rewriting, multi-query and HyDE; larger top-k then cross-encoder rerank; domain-tuned embeddings; metadata filters; fix OCR/table parsing at ingestion.

**If it still fails:** measure retrieval (recall@k) separately from generation. Check whether the answer exists in the corpus at all. Then try agentic/iterative retrieval, fine-tuning the embedder or reranker, knowledge-graph or SQL routes for structured questions, long-context stuffing for small corpora, and finally abstain or hand off to a human with citations rather than guess.

D. Sliding window attention and positional ("time") embeddings

**Sliding window attention:** each token attends only to the previous W tokens, so cost is O(n·W) instead of O(n²) and the KV cache can be a fixed rolling buffer. Information still travels farther through stacked layers (receptive field ≈ layers × W). Used in Mistral 7B and Longformer; Gemma-style models interleave local and global layers.

**Positional encodings:** sinusoidal (original Transformer), learned absolute, relative bias (T5), RoPE (rotates Q/K by position; most modern LLMs), ALiBi (linear distance penalty, extrapolates well). Context extension: position interpolation, NTK scaling, YaRN.

**Timestep embeddings** (diffusion) are a different thing: sinusoidal embeddings of the noise step fed through an MLP to condition the network.

E. Anthropic model choice for the multi-agent coding workflow

| Agent | Model | Why |
| --- | --- | --- |
| A. Orchestrator | Opus | Long-running delegation, failure-mode judgment, deciding when to escalate to a human, and user-facing chat all need strong reasoning and reliability. Sonnet is a cost-saving alternative if tasks are simple. |
| B. Backend connectors | Sonnet | Strong tool/MCP use and precise schema-to-contract work at good speed and cost; errors here break the frontend handoff, so Haiku is risky. |
| C. Frontend UI | Haiku | Work is well-specified (spec + design system + contracts), high volume, and latency-sensitive. Validate with lint/schema checks; upgrade to Sonnet if design-system compliance drifts. |
| D. Architect | Fable | Intent-based judgment, architecture trade-offs and advising other agents are the hardest cognitive tasks, and they run rarely, so the top tier's cost is justified. Opus is the fallback. |

Principle: spend capability where errors compound (architecture, orchestration), economize where the task is bounded and verifiable. Confirm current model names and pricing at docs.claude.com.

## Inference

1\. Batch LLM requests on one GPU while users wait

Use continuous (in-flight) batching: new requests join the running batch at every decode step instead of waiting for a full batch. Add PagedAttention for KV memory, chunked prefill so long prompts don't stall decode, and a small queue-wait cap to bound latency.

2\. What is the KV cache, and why does it limit concurrency?

It stores each token's attention keys and values so they aren't recomputed per step. It grows linearly with tokens × layers × KV heads, so after weights, remaining GPU memory caps how many sequences fit. Mitigate with GQA, paged allocation, KV quantization, prefix sharing, and shorter contexts.

3\. Autoscale LLM inference on Kubernetes. Why is CPU the wrong signal?

GPU-bound serving leaves CPU low even when saturated. Scale on queue depth, in-flight requests, KV-cache utilization, or TTFT/latency SLO (via KEDA or HPA custom metrics). Account for slow cold starts (model load): keep warm pool, pre-pull images, scale up early and down slowly.

4\. Distillation vs quantization: what do you trade away?

Quantization shrinks precision of the same model: cheap, no training (PTQ) or little (QAT), small accuracy loss, mainly hurts at 4-bit and below or on outlier-heavy layers. Distillation trains a smaller student from a teacher: larger savings and a new architecture, but costs training compute/data and may lose long-tail capability. They combine.

5\. Diagnose high latency. Which metrics matter?

Split TTFT (queueing + prefill), time per output token (decode), and total. Check queue time, batch size, GPU utilization, KV-cache pressure/evictions, prompt and output lengths, network and tokenization overhead, and retrieval/tool time. Look at p95/p99, not averages.

## RAG

6\. Version documents so stale content never surfaces

Store doc ID, version, effective/expiry dates and status as metadata; on update, upsert the new chunks and tombstone the old ones; filter retrieval on "current" (or as-of date); use content hashes for change detection; keep an audit log and cite version in answers.

7\. Ingest large tables without losing structure

Don't flatten into prose. Parse to structured form (layout-aware parser), keep header context on every row-group chunk, store rows in a SQL/dataframe store, and let the agent query it (text-to-SQL) with a table summary indexed for retrieval. Embed row-wise with column names for lookup-style questions.

8\. Prompting vs RAG vs fine-tuning: how do you choose?

Start with prompting (behavior, format). Add RAG when knowledge is private, large or changing and needs citations. Fine-tune for consistent style/format, narrow tasks, latency/cost reduction, or skills prompts can't teach. Fine-tuning is poor for injecting fresh facts. Often combine all three.

## Agents and orchestration

9\. A claims-approval agent under a token budget

Triage with a cheap model and rules; auto-decide clear cases; retrieve only relevant policy clauses; summarize long documents once and cache; structured outputs; cap loop iterations; route ambiguous or high-value claims to a human with a compact evidence packet. Track cost per claim.

10\. A support agent with tools, memory, and human handoff

Short-term memory in the conversation state, long-term memory as retrieved customer facts; least-privilege tools with confirmations for writes; confidence/sentiment/policy triggers for handoff; transfer a summary and tool trace to the human so users don't repeat themselves.

11\. A summarizer agent in LangGraph

Graph with nodes: load/split, summarize chunks (map), combine (reduce), optional critique/refine loop with a conditional edge and iteration cap; typed state; checkpointer for persistence and resume; streaming output; evaluate for faithfulness.

12\. Orchestrate multiple agents: planner vs executors, shared state, failure recovery

Planner decomposes and assigns; executors have narrow tools. Shared state is a typed, versioned store with clear ownership. Recovery: per-step timeouts, retries with backoff, idempotent actions, checkpoints, replan on failure, circuit breakers, and human escalation. Trace everything.

13\. Connect an agent to enterprise tools over MCP: auth, RBAC, schema-validated calls

OAuth 2.x with per-user (delegated) tokens rather than shared keys; enforce RBAC server-side, not in the prompt; validate every call against JSON Schema; allow-list tools, rate-limit, log and audit; human approval for destructive actions; treat tool output as untrusted input.

14\. A voice agent: sub-second latency, interruptions, handoff

Streaming ASR → LLM → TTS pipeline (or speech-to-speech model) with overlap; small fast model; voice activity detection and barge-in that cancels TTS and generation; WebRTC transport; filler/acknowledgements; warm transfer with transcript summary to a human.

15\. An AI recruiter for sales hiring: screening, outreach, bias checks

Screen against job-related, structured criteria only; keep humans making final decisions; exclude protected attributes and proxies; run adverse-impact analysis (e.g., four-fifths rule) and periodic audits; disclose AI use; comply with local rules (e.g., NYC Local Law 144, EU AI Act high-risk); personalized but honest outreach.

## Guardrails and Responsible AI

16\. Guardrails: prompt injection, PII redaction, output validation, tool permissions

Layered defense: separate trusted instructions from untrusted content, no single filter is enough; injection classifiers; PII detection/redaction before the model and in logs; schema and policy validation of outputs; least-privilege tools, approvals for risky actions, sandboxing, and egress controls.

17\. Responsible AI for regulated decisions: bias testing, explainability, audit trails

Test outcomes across groups (disparate impact, equalized odds); document with model cards; provide reason codes tied to real decision factors (not post-hoc stories); human review and appeal; log inputs, model/prompt versions, outputs and approvals immutably; monitor drift. Map to applicable regulation.

## Evaluation and observability

18\. How do you know it actually works?

Golden dataset from real traffic with edge cases; task-specific metrics; automated plus human review; offline evals in CI, then online monitoring and A/B or shadow tests; track failure categories and fix by category.

19\. LLM-as-a-judge: failure modes and calibration

Biases: position, verbosity, self-preference, leniency. Mitigate with pairwise order swapping, rubrics, reference answers, and a different judge model. Calibrate against human labels (agreement, Cohen's kappa), re-check periodically.

20\. A model hallucinates or repeats itself on high-stakes answers. Fix it

Ground with retrieval and require citations; allow "I don't know"; verify claims against sources (checker pass); lower temperature; adjust repetition/frequency penalties or max tokens; use structured outputs; stronger model or reasoning where needed; human review for high-stakes cases.

21\. Micro vs Macro F1 on imbalanced data

Micro pools all TP/FP/FN, so frequent classes dominate (close to accuracy). Macro averages per-class F1 equally, exposing weak performance on rare classes. Use macro (or weighted) when rare classes matter; report per-class precision/recall too.

22\. Catch regressions when a prompt or model changes, and roll back safely

Version prompts and models like code; run the regression eval suite in CI with thresholds; shadow or canary release with automatic metric guards; feature flags and pinned model versions; keep the previous version deployable for instant rollback.

23\. Observability for agents: traces, cost per request, alerts

Trace every step (LLM calls, tool calls, retrieval) with OpenTelemetry-style spans; record tokens, cost, latency, model version, and outcome per request; dashboards by route; alerts on error rate, latency, cost spikes, loops, and eval-score drift; sample traces for review.

## Coding

24\. A rate limiter with per-user and global limits

Token bucket per user plus a global bucket; a request must pass both. Single process: dict of buckets with a lock. Distributed: Redis with an atomic Lua script (or sliding-window counters). Return 429 with Retry-After. Decide fail-open vs fail-closed if the store is down.

25\. Async retries with backoff, jitter, and timeouts

Wrap calls in `asyncio.timeout`; retry only transient errors (429, 5xx, timeouts); delay = random(0, min(cap, base × 2^attempt)) ("full jitter"); respect Retry-After; cap attempts and total deadline; require idempotency for retried writes.

26\. A RAG pipeline over a folder of docs

Load files → parse/clean → chunk (200–500 tokens with overlap) → embed → store in a vector DB (with metadata) → at query: embed, retrieve top-k (hybrid), rerank → prompt with context and citation instructions → generate. Add incremental re-indexing by file hash.

## System design and FDE

27\. Kafka into an AI pipeline: ordering, duplicates, consumer lag, replay

Ordering holds only within a partition, so key by entity. Kafka gives at-least-once by default, so make processing idempotent (dedupe by event ID) or use transactions. Scale consumers and watch lag; use retries and a dead-letter topic for poison messages; replay by resetting offsets, with deterministic or versioned results.

28\. RAG over 50M patient records under HIPAA, inside a customer's VPC

Deploy everything in the VPC (self-hosted or private-endpoint models, BAA with any vendor); encryption at rest/in transit; per-user access control enforced in retrieval via metadata filters; de-identification where possible; audit logging; minimum necessary access; sharded ANN index (e.g., HNSW/IVF-PQ) and incremental updates; no PHI in logs.

29\. Cut an LLM search from 1.5s to under 100ms

Profile first. Remove the LLM from the hot path: semantic/embedding cache, precomputed results, ANN retrieval, smaller embedding model, and a small reranker. Use the LLM only asynchronously or for hard queries; stream and parallelize steps; co-locate services. Honest answer: generated answers rarely hit 100ms, but retrieval does.

30\. Turn body-camera audio into a draft incident report that stays accurate and human-reviewed

Diarized ASR with timestamps → LLM drafts only from the transcript, each statement linked to its audio timestamp; flag low-confidence segments; never invent facts; officer reviews and edits side by side and signs off; log versions and AI involvement; strict access and retention controls.

31\. Unify fraud detection across three acquired systems. Scope the first 90 days

**Days 1–30:** inventory systems, data, rules, metrics, and owners; agree baseline KPIs. **31–60:** common event schema and data pipeline, read-only unified view, shadow-mode scoring. **61–90:** pilot unified model/rules on one segment with human review, compare to baseline, plan phased migration and retirement.

32\. Walk through a system you owned end to end: what broke, what you changed

Use STAR: context and scale, your role, the failure with root cause (be specific and quantified), the fix and trade-offs, measured result, and what you'd do differently. Pick a real system you can defend in depth.

33\. The deployment slipped three weeks. Tell the CTO, how would you react?

Tell them promptly and directly: what happened, impact, root cause, the new date with confidence level, options (cut scope, add resources, phase release), and what you're changing to prevent recurrence. Own it; bring a plan, not excuses.

Accuracy note: these answers are from my general knowledge and were not checked against live sources. Verify model names, limits and regulations before relying on them.
