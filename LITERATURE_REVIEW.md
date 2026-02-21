# ShieldLM Literature Review: Prompt Injection Attack & Defense Research (2023–2026)

*Compiled by Dimiter Milushev — February 2026*
*For: ShieldLM project — dataset curation, model training, and technical report*

---

## Executive Summary

The field of prompt injection defense has undergone a paradigm shift between 2024 and 2025. What began as chatbot-level content moderation has evolved into a multi-layered systems security discipline, driven by the explosion of agentic AI. This review covers **20+ papers** across four defense paradigms: detection-based classifiers, alignment-based model training, system-level architectural defenses, and dynamic benchmarking. The key takeaway for ShieldLM: **detection-based approaches remain essential** (they are the fastest and most deployable), but the bar has risen dramatically — false positive rates below 0.1% are now the standard for production viability.

---

## 1. Foundational Work (2023–2024)

### 1.1 InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents

**Zhan, Q. et al. (2024). arXiv:2403.02691. UIUC.**

The paper that defined the indirect prompt injection threat model for agentic systems. InjecAgent creates 1,054 test cases across 17 user tools and 62 attacker tools, covering two attack categories: direct harm (financial manipulation, IoT exploitation, file deletion) and data exfiltration (stealing payment methods, medical records, addresses and emailing them to the attacker).

**Key findings:**
- ReAct-prompted GPT-4 follows injected instructions 24% of the time (base), rising to 47% with a "hacking prompt" enhancement.
- Llama2-70B is the most vulnerable at 87%+ attack success rate.
- Fine-tuned GPT-4 drops to 6.6% ASR — a 3–6× improvement over prompting alone.
- Even fine-tuned models forward exfiltrated data to the attacker 100% of the time once they extract it.
- Claude-2 is the most resilient prompted agent: the hacking prompt *decreased* its ASR from 11.4% to 3.4% (triggered safety awareness).
- "Content freedom" (free-text fields like reviews and bios) is the primary vulnerability vector.

**Relevance to ShieldLM:** Core dataset. The tool response templates with embedded injections are the gold standard for indirect injection training data. Our pipeline extracts both the attacker instruction (payload) and the full tool response (context).

---

### 1.2 JailbreakBench: An Open Robustness Benchmark for Jailbreaking LLMs

**Chao, P. et al. (2024). NeurIPS 2024 Datasets and Benchmarks Track. arXiv:2404.01318.**

Introduces JBB-Behaviors: 100 harmful + 100 paired benign behaviors across 10 OpenAI policy categories. The paired structure (harmful/benign on the same topic) is methodologically important — it forces classifiers to learn intent rather than topic.

**Key findings:**
- Six jailbreak classifiers compared via rigorous human evaluation.
- Llama-3-Instruct-70B is an effective automated judge with proper prompting.
- Existing attacks like PAIR achieve high ASRs on Vicuna but lower on safety-tuned models.
- Provides a repository of adversarial prompts ("jailbreak artifacts") for reproducible research.

**Relevance to ShieldLM:** Evaluation dataset. The paired harmful/benign structure is critical for measuring false positive rates on semantically similar but benign inputs.

---

### 1.3 Formalizing and Benchmarking Prompt Injection Attacks and Defenses

**Liu, Y. et al. (2024). USENIX Security 2024.**

The first systematic formalization of prompt injection attacks. Introduces a framework that separates the attack into injection placement, injection content, and injection strategy. Proposes the SEP benchmark with 9.1K samples.

**Relevance to ShieldLM:** Provides the formal threat model our taxonomy builds on. The SEP benchmark is a potential additional evaluation corpus.

---

### 1.4 StruQ: Defending Against Prompt Injection with Structured Queries

**Chen, S. et al. (2024). USENIX Security 2025. arXiv:2402.06363.**

Proposes structured queries that create explicit syntactic boundaries between instructions and data. By reformatting inputs into structured formats, the LLM can distinguish trusted instructions from potentially poisoned data.

**Relevance to ShieldLM:** Key baseline. StruQ's training data construction (pairing injected inputs with both secure and insecure outputs) directly influenced our dataset design for the alignment approach.

---

### 1.5 AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses

**Debenedetti, E. et al. (2024). NeurIPS 2024. ETH Zurich.**

A dynamic (not static) evaluation environment with 97 realistic tasks and 629 adversarial scenarios across 4 domains (workspace, banking, travel, e-commerce). Tests both utility under attack and attack success rates.

**Key findings:**
- GPT-4o achieves 69% benign utility but drops to 45% under attack.
- "Important message" canonical injection achieves 53.1% ASR.
- Prompt sandwiching improves utility-under-attack (65.7%) but leaves ASR high (30.8%).
- Tool filtering suppresses ASR to 7.5% but reduces utility to 53.3%.

**Relevance to ShieldLM:** Primary evaluation benchmark for agentic indirect injection. Our classifier should be benchmarked as a defense within AgentDojo's framework.

---

## 2. Detection-Based Defenses (2025)

### 2.1 PromptShield: Deployable Detection for Prompt Injection Attacks

**Jacob, D., Alzahrani, H., Hu, Z., Alomair, B., & Wagner, D. (2025). ACM CODASPY 2025. arXiv:2501.15145.**

**This is the paper most directly relevant to ShieldLM.**

PromptShield introduces a critical insight: prompt injection detectors must handle two fundamentally different data types — *conversational data* (chatbot inputs, where injections are self-defeating) and *application-structured data* (tool inputs, where injections are the real threat). Prior detectors conflated these, leading to unusably high false positive rates.

**Key contributions:**
- **Deployment-aware evaluation**: Evaluates at low FPR operating points (0.1%, 1%) rather than AUC, which is misleading in practice.
- **PromptGuard (Meta's detector) achieves only 9.4% TPR at 0.1% FPR** on their benchmark — essentially unusable in production.
- **PromptShield detector achieves 65.3% TPR at 0.1% FPR** and 94.8% TPR at 1% FPR.
- Larger models and careful data curation are the two biggest levers for detector performance.
- Training data composition matters enormously: including conversational data as negatives is essential.

**Critical insight for ShieldLM:** The FPR operating point is everything. A detector with 99% accuracy but 5% FPR will block 1 in 20 legitimate requests — unacceptable at scale. We must evaluate at 0.1% FPR, not just report AUC or accuracy.

---

### 2.2 InstructDetector: Defending Against Indirect Prompt Injection by Instruction Detection

**Wen, T. et al. (2025). arXiv:2505.06311.**

A novel detection approach that leverages LLM behavioral state changes (hidden states and gradients from intermediate layers) to identify when external data contains embedded instructions.

**Key contributions:**
- Achieves **99.60% detection accuracy** in-domain and **96.90% out-of-domain**.
- Reduces ASR to **0.03%** on the BIPIA benchmark.
- Demonstrates that hidden states and gradients from intermediate layers provide highly discriminative features for distinguishing instructions from data.
- Works as an external screening step: scans external data *before* the LLM sees it.

**Relevance to ShieldLM:** Demonstrates that white-box features (hidden states) dramatically outperform text-only classifiers for indirect injection. Our DeBERTa classifier operates in a different (black-box) regime, but this sets the upper bound for what's achievable with model internals.

---

### 2.3 LlamaFirewall: An Open Source Guardrail System for Building Secure AI Agents

**Chennabasappa, S. et al. (2025). Meta. arXiv:2505.03574.**

Meta's production-grade open-source security framework with three layered guardrails:

1. **PromptGuard 2** (86M and 22M parameter variants) — a fine-tuned BERT-style classifier for real-time jailbreak and prompt injection detection. The 22M variant reduces latency by 75% with minimal performance loss.
2. **AlignmentCheck** — a chain-of-thought auditor that inspects agent reasoning for goal hijacking and indirect injection. The first open-source guardrail to audit CoT in real-time.
3. **CodeShield** — static analysis for insecure code generation (96% precision, 79% recall).

**Key results on AgentDojo:**
- PromptGuard 2 alone: ASR drops from 17.6% → 7.5%.
- AlignmentCheck alone: ASR drops to 2.9% (but higher compute cost).
- Combined: ASR drops to **1.75%** (90% reduction), with utility at 42.7%.

**Relevance to ShieldLM:** PromptGuard 2 is a direct competitor/baseline. The 86M and 22M model sizes set the reference points for our DeBERTa classifier. The AlignmentCheck approach (CoT auditing) represents a complementary defense strategy we should evaluate against.

---

## 3. Alignment-Based Defenses (2025)

### 3.1 SecAlign: Defending Against Prompt Injection with Preference Optimization

**Chen, S. et al. (2025). ACM CCS 2025. arXiv:2410.05451.**

Uses DPO (Direct Preference Optimization) to fine-tune LLMs to prefer following the legitimate instruction over the injected one. The preference dataset is constructed automatically from any instruction-tuning dataset — no manual injection crafting needed.

**Key contributions:**
- Reduces GCG-based attack ASR from 96% (Sandwich defense) and 56% (StruQ) to **2%**.
- First method to achieve consistent **0% ASR for optimization-free attacks**.
- Works by making the LLM fundamentally harder to optimize against, not just filtering inputs.
- Preference dataset construction is simple string concatenation — no expensive red-teaming.

**Relevance to ShieldLM:** SecAlign's training data construction method (pairing secure/insecure outputs for injected inputs) could be adapted for our Mistral/Llama fine-tuning approach. The key insight is that DPO generalizes better than SFT for this task.

---

### 3.2 Meta SecAlign: A Secure Foundation LLM Against Prompt Injection Attacks

**Chen, S., Zharmagambetov, A., Wagner, D., & Guo, C. (2025). Meta. arXiv:2507.02735.**

The first fully open-source LLM with built-in model-level prompt injection defense that achieves commercial-grade performance. Builds on SecAlign with several improvements.

**Key contributions:**
- Releases **Meta-SecAlign-8B** and **Meta-SecAlign-70B** — open weights and complete training recipe.
- Trained only on a generic instruction-tuning dataset, yet security transfers to unseen downstream tasks including tool-calling and web navigation.
- Evaluated on **9 utility benchmarks + 7 security benchmarks** — the most comprehensive evaluation to date.
- Meta-SecAlign-70B is more secure than several flagship proprietary models.
- Introduces a new `input` role in the chat template to create explicit data boundaries.

**Key results:**
- On InjecAgent: significant reduction in ASR for both base and enhanced attacks.
- On AgentDojo & WASP: security transfers even though training used only generic instruction data.

**Relevance to ShieldLM:** Direct competitor and potential base model. We could fine-tune Meta-SecAlign-8B with LoRA for our detector, combining their alignment-based defense with our detection-based approach. Also validates that InjecAgent is used as a primary evaluation benchmark by top labs.

---

## 4. System-Level Defenses (2025)

### 4.1 CaMeL: Defeating Prompt Injections by Design

**Debenedetti, E., Shumailov, I., Fan, T., Hayes, J., Carlini, N., Fabian, D., Kern, C., Shi, C., Terzis, A., & Tramèr, F. (2025). Google / Google DeepMind / ETH Zurich. arXiv:2503.18813.**

The most architecturally ambitious defense to date. CaMeL creates a protective system layer *around* the LLM, inspired by traditional software security: Control Flow Integrity, Access Control, and Information Flow Control.

**How it works:**
1. A **privileged LLM** extracts a program (control flow + data flow) from the trusted user query.
2. A **quarantined LLM** processes untrusted data but returns only symbolic variables ($VAR1, $VAR2), never directly influencing control flow.
3. **Capability-based access control** prevents data exfiltration by enforcing security policies when tools are called.

**Key results:**
- Solves **77% of tasks with provable security** in AgentDojo (vs. 84% with no defense).
- Security is *provable*, not probabilistic — if the control/data flow separation is maintained, injection is impossible by construction.
- The first defense to make formal security guarantees against prompt injection.

**Relevance to ShieldLM:** CaMeL represents the architectural future of prompt injection defense. Our detection-based approach is complementary — CaMeL still needs a classifier to flag when untrusted data might contain instructions (the quarantined LLM step). ShieldLM could serve as the fast pre-filter in a CaMeL-style architecture.

---

### 4.2 Design Patterns for Securing LLM Agents against Prompt Injections

**Beurer-Kellner, L. et al. (2025). IBM / Invariant Labs / ETH Zurich / Google / Microsoft. arXiv (June 2025).**

Synthesizes the field into **six design patterns** for defending against prompt injection:

1. **Input sanitization** — filter injections before the LLM sees them (PromptShield, ShieldLM).
2. **Prompt augmentation** — add defensive instructions to the system prompt.
3. **Alignment training** — fine-tune the LLM to resist injections (SecAlign).
4. **Output monitoring** — check outputs for signs of compromise.
5. **Privilege separation** — dual-LLM architecture (CaMeL).
6. **Execution sandboxing** — limit what the agent can do.

**Key insight:** *"As long as both agents and their defenses rely on the current class of language models, we believe it is unlikely that general, reliable defenses for prompt injection will be achieved."* Defense-in-depth is not optional — it's structurally necessary.

**Relevance to ShieldLM:** Positions our classifier as Pattern #1 (input sanitization) — the fastest and most deployable layer. The paper validates that this remains essential even in a world with CaMeL and SecAlign.

---

### 4.3 PromptArmor: Simple yet Effective Prompt Injection Defenses

**arXiv:2507.15219. (July 2025).**

Demonstrates that an off-the-shelf LLM can be strategically prompted to detect and remove injected prompts — achieving both FPR and FNR below 1% on AgentDojo with GPT-4o, GPT-4.1, or o4-mini as the guardrail.

**Key finding:** *"Even if the guardrail LLM itself remains vulnerable to prompt injection — attacks achieve 55% ASR against GPT-4.1 when no defense is deployed — it can still be strategically prompted to accurately detect and remove injected prompts."*

This challenges the assumption that detection requires fine-tuned specialist models.

**Relevance to ShieldLM:** A strong baseline to beat. If a prompted GPT-4o achieves <1% FPR and FNR, our fine-tuned DeBERTa must match or exceed this while being 100× faster and cheaper. The value proposition shifts to latency and cost rather than raw accuracy.

---

## 5. New Benchmarks & Evaluation (2025)

### 5.1 WASP: Benchmarking Web Agent Security Against Prompt Injection Attacks

**Evtimov, I., Zharmagambetov, A., Grattafiori, A., Guo, C., & Chaudhuri, K. (2025). Meta. arXiv:2504.18575. ICML 2025.**

The first realistic benchmark for web-navigation agent security. Unlike InjecAgent's simulated API calls, WASP tests actual web agents (OpenAI Operator, Claude Computer Use, WebArena scaffolding) navigating real websites with adversarial content.

**Key contributions:**
- **Realistic threat model**: attacker controls only user-generated content areas (reviews, comments, bios), not the whole page.
- End-to-end evaluation: measures both whether the agent is *hijacked* and whether it *completes* the malicious task.
- Even top models (GPT-4o, Claude 3.5 Sonnet, o1) are deceived by simple human-written injections.
- Task-related injections are far more effective than task-agnostic ones.

**Relevance to ShieldLM:** Represents the next generation of evaluation beyond InjecAgent. We should benchmark our classifier on WASP's injection payloads to test generalization to web-native attacks.

---

### 5.2 AgentDyn: A Dynamic Open-Ended Benchmark for Agent Security

**arXiv:2602.03117. (February 2026).**

Questions whether existing benchmarks (InjecAgent, AgentDojo) truly reflect real-world risks. Argues that static benchmarks create a false sense of security — defenses that achieve near-zero ASR on AgentDojo may fail in dynamic, open-ended scenarios.

**Relevance to ShieldLM:** Validates our adversarial robustness testing strategy. We should not just report AgentDojo numbers — we need paraphrase attacks, multilingual evasion, and encoding tricks to test true generalization.

---

### 5.3 Securing AI Agents Against Prompt Injection Attacks

**arXiv:2511.15759. (November 2025).**

Introduces an 847-case benchmark across 5 attack categories (direct injection, context manipulation, instruction override, data exfiltration, cross-context contamination) and evaluates a multi-layered defense combining content filtering, hierarchical guardrails, and response verification.

**Key result:** Combined defense reduces ASR from 73.2% → **8.7%** while maintaining 94.3% of baseline task performance.

**Relevance to ShieldLM:** Their taxonomy of 5 attack categories partially overlaps ours. The cross-context contamination category is novel and should be added to our evaluation.

---

## 6. Emerging Directions (Late 2025 – Early 2026)

### 6.1 Multi-Agent Defense Pipelines

**arXiv:2509.14285. (December 2025).**

Uses specialized LLM agents in coordinated pipelines (chain-of-agents or coordinator-based) to detect and neutralize prompt injection in real-time. Introduces the HPI_ATTACK_DATASET with 400 attack instances across 8 categories.

---

### 6.2 MCP Protocol Vulnerabilities

**ScienceDirect (2025). "From prompt injections to protocol exploits."**

A comprehensive survey of 150+ publications identifying 30+ attack techniques against LLM-powered agent ecosystems. Specifically flags vulnerabilities in the Model Context Protocol (MCP), agentic web interfaces, and memory-centric LLM risks.

**Relevance to ShieldLM:** MCP-specific attacks are a gap in current datasets. As MCP adoption grows, we need injection examples targeting MCP tool calls.

---

### 6.3 Visual Prompt Injection

**Meta CyberSecEval 3 (2024). facebook/cyberseceval3-visual-prompt-injection on HuggingFace.**

1,000 test cases for multimodal prompt injection through images. Injection techniques include text embedded in images, CAPTCHA-style payloads, and adversarial perturbations.

**Relevance to ShieldLM:** Out of scope for v1 (text-only), but important for v2 as multimodal agents become standard.

---

## 7. Competitive Landscape: Where ShieldLM Fits

| Defense | Type | Speed | Open Source | Covers Indirect PI | Production-Ready |
|---------|------|-------|-------------|-------------------|-----------------|
| **ProtectAI DeBERTa v2** | Detection (classifier) | ~10ms | Yes (Apache 2.0) | No | Yes |
| **PromptGuard 2 (Meta)** | Detection (classifier) | ~5ms (22M) | Yes | Limited | Yes |
| **PromptShield** | Detection (classifier) | ~15ms | Benchmark only | Yes | Yes |
| **LlamaFirewall** | System (multi-layer) | ~100ms+ | Yes | Yes (AlignmentCheck) | Yes |
| **CaMeL** | System (architectural) | ~2× base | Yes | Yes (by design) | Experimental |
| **SecAlign / Meta SecAlign** | Alignment (model-level) | Same as base | Yes (8B, 70B) | Yes | Experimental |
| **PromptArmor** | Detection (LLM-as-guard) | ~500ms+ | No | Yes | Experimental |
| **ShieldLM (ours)** | Detection (classifier) | Target: <10ms | Yes | **Yes** | Planned |

### ShieldLM's Differentiation

1. **Unified taxonomy**: First classifier trained on direct injection + indirect injection + jailbreak with a hierarchical label scheme.
2. **Indirect injection focus**: Most classifiers (ProtectAI, PromptGuard) are weak on tool-response-embedded injections. We train explicitly on InjecAgent data.
3. **Deployment-aware evaluation**: Following PromptShield's methodology, we evaluate at 0.1% and 1% FPR, not just AUC.
4. **Multilingual**: Trained on injection data in 7+ languages (leveraging the yanismiraoui dataset).
5. **Adversarial robustness**: Tested against paraphrase attacks, encoding tricks, and cross-lingual evasion.

---

## 8. Key Open Problems (Opportunities for ShieldLM)

1. **False positive rate at scale.** PromptShield showed that even state-of-the-art detectors struggle below 0.1% FPR. Every improvement here has massive production impact.

2. **Indirect injection in realistic contexts.** InjecAgent uses synthetic tool responses. Real-world injections will be embedded in actual emails, web pages, and documents with natural language noise. No dataset captures this yet.

3. **Adaptive attacks against detectors.** No study has systematically evaluated how well prompt injection *detectors* (as opposed to LLM defenses) withstand adaptive adversaries who know the detector exists. This is the adversarial ML problem applied to PI detection.

4. **MCP and tool-protocol-specific injections.** As MCP becomes standard, injection vectors specific to MCP's tool-calling structure will emerge. No benchmark exists yet.

5. **Cost-performance tradeoff.** PromptArmor shows a prompted GPT-4o achieves <1% FPR/FNR. A fine-tuned DeBERTa at 10ms per request is 100× cheaper and faster. Quantifying this tradeoff is valuable for production deployments.

---

## Full Reference List

### Foundational (2023–2024)

- Zhan, Q. et al. (2024). *InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents.* arXiv:2403.02691.
- Chao, P. et al. (2024). *JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models.* NeurIPS 2024. arXiv:2404.01318.
- Liu, Y. et al. (2024). *Formalizing and Benchmarking Prompt Injection Attacks and Defenses.* USENIX Security 2024.
- Chen, S. et al. (2024). *StruQ: Defending Against Prompt Injection with Structured Queries.* USENIX Security 2025. arXiv:2402.06363.
- Debenedetti, E. et al. (2024). *AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents.* NeurIPS 2024.
- Sharma, R. et al. (2024). *SPML: A DSL for Defending Language Models Against Prompt Attacks.* arXiv:2402.11755.
- Greshake, K. et al. (2023). *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.* ACM AISec.
- Zou, A. et al. (2023). *Universal and Transferable Adversarial Attacks on Aligned Language Models.* arXiv:2307.15043.
- OWASP (2025). *LLM01:2025 Prompt Injection.* OWASP Top 10 for LLM Applications.

### Detection-Based Defenses (2025)

- Jacob, D. et al. (2025). *PromptShield: Deployable Detection for Prompt Injection Attacks.* ACM CODASPY 2025. arXiv:2501.15145.
- Wen, T. et al. (2025). *Defending against Indirect Prompt Injection by Instruction Detection (InstructDetector).* arXiv:2505.06311.
- Chennabasappa, S. et al. (2025). *LlamaFirewall: An open source guardrail system for building secure AI agents.* Meta. arXiv:2505.03574.
- ProtectAI (2024). *deberta-v3-base-prompt-injection-v2.* HuggingFace.

### Alignment-Based Defenses (2025)

- Chen, S. et al. (2025). *SecAlign: Defending Against Prompt Injection with Preference Optimization.* ACM CCS 2025. arXiv:2410.05451.
- Chen, S. et al. (2025). *Meta SecAlign: A Secure Foundation LLM Against Prompt Injection Attacks.* Meta. arXiv:2507.02735.
- Wallace, E. et al. (2024). *Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions.* OpenAI.

### System-Level Defenses (2025)

- Debenedetti, E. et al. (2025). *Defeating Prompt Injections by Design (CaMeL).* Google / ETH Zurich. arXiv:2503.18813.
- Beurer-Kellner, L. et al. (2025). *Design Patterns for Securing LLM Agents against Prompt Injections.* IBM / ETH Zurich / Google / Microsoft.
- PromptArmor (2025). *Simple yet Effective Prompt Injection Defenses.* arXiv:2507.15219.
- Zhu et al. (2025). *MELON: Indirect Prompt Injection Defense via Masked Re-Execution and Tool Comparison.* arXiv:2502.05174.
- Shi et al. (2025). *Progent: Programmable Privilege Control for LLM Agents.*

### Benchmarks & Evaluation (2025)

- Evtimov, I. et al. (2025). *WASP: Benchmarking Web Agent Security Against Prompt Injection Attacks.* Meta. ICML 2025. arXiv:2504.18575.
- arXiv:2602.03117 (2026). *AgentDyn: A Dynamic Open-Ended Benchmark for Evaluating Prompt Injection Attacks of Real-World Agent Security Systems.*
- arXiv:2511.15759 (2025). *Securing AI Agents Against Prompt Injection Attacks: A Comprehensive Benchmark and Defense Framework.*
- arXiv:2509.14285 (2025). *A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks.*

### Surveys & Taxonomies

- ScienceDirect (2025). *From prompt injections to protocol exploits: Threats in LLM-powered AI agents workflows.*
- arXiv:2512.16307 (2025). *Beyond the Benchmark: Innovative Defenses Against Prompt Injection Attacks.*
- SafetyPrompts.com — Paul Röttger's living catalogue of open datasets for LLM safety.
- Simon Willison (2025). *Design Patterns for Securing LLM Agents against Prompt Injections.* Blog review, June 2025.

---

## Appendix: Datasets Referenced in 2025 Papers

| Dataset | Source | Size | Used By |
|---------|--------|------|---------|
| InjecAgent | UIUC | 1,054 | Meta SecAlign, WASP, AgentDyn |
| AgentDojo | ETH Zurich | 97 tasks / 629 scenarios | CaMeL, LlamaFirewall, Meta SecAlign, PromptArmor |
| WASP | Meta | End-to-end web scenarios | Meta SecAlign |
| JBB-Behaviors | JailbreakBench | 200 | Multiple |
| SEP | Liu et al. | 9.1K | SecAlign |
| BIPIA | Yi et al. | ~300 | InstructDetector |
| PromptShield Benchmark | Jacob et al. | Multi-source curated | PromptShield |
| HPI_ATTACK_DATASET | Multi-Agent Defense | 400 | Multi-agent pipeline |
| CyberSecEval 3 (visual) | Meta | 1,000 | Visual PI research |
| AdvBench | Zou et al. | 520 | JailbreakBench, HarmBench |
| deepset/prompt-injections | deepset | 662 | ProtectAI, many classifiers |
| Harelix/Mixed-Techniques | Harelix | 1,174 | ProtectAI v2 |
| SPML Chatbot PI | Sharma et al. | ~7,000 | SPML |

---

*Last updated: February 21, 2026*
