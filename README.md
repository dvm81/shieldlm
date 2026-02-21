# 🛡️ ShieldLM

**Layer 1 prompt injection detection for AI agent security.**

A fast, deployment-aware classifier trained on a unified dataset spanning direct injection, indirect injection, and jailbreak — the three attack categories a production defense stack needs to handle. Designed as the first line of defense in a layered architecture (alongside CaMeL, SecAlign, LlamaFirewall).

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/dvm81/shieldlm.git
cd shieldlm
pip install -r requirements.txt

# 2. Get InjecAgent data
git clone https://github.com/uiuc-kang-lab/InjecAgent.git

# 3. Curate dataset (offline mode)
python curate_dataset.py --output ./data/unified --skip-hf

# 4. Full dataset (with HuggingFace sources)
python curate_dataset.py --output ./data/unified
```

## What's Included

| File | Description |
|------|-------------|
| `curate_dataset.py` | Dataset curation pipeline — loads, normalizes, deduplicates, splits |
| `taxonomy.yaml` | Attack taxonomy and source catalog |
| `BLOGPOST_v2.md` | Detailed blog post explaining the approach |
| `LITERATURE_REVIEW.md` | Comprehensive survey of 2023–2026 prompt injection research |
| `CRITICAL_ANALYSIS.md` | Self-audit and design rationale |

## Attack Taxonomy

```
Level 1 (binary):    BENIGN | ATTACK
Level 2 (category):  benign | direct_injection | indirect_injection | jailbreak  
Level 3 (intent):    goal_hijacking | data_exfiltration | financial_harm | ...
```

## Key Design Decisions

- **Context-aware labeling**: Isolated payloads are NOT labeled as attacks — context determines injection (PromptShield insight)
- **Application-structured benign data**: Clean tool responses prevent format-based false positives
- **JailbreakBench for FP evaluation**: Harmful topics are NOT injections — tests classifier precision
- **FPR-based evaluation**: Report TPR at 0.1% and 1% FPR, not misleading AUC

## Roadmap

- [x] Dataset curation pipeline (9 sources, unified schema)
- [x] Attack taxonomy (3-level hierarchy)
- [x] Literature review (20+ papers, 2023–2026)
- [ ] Fine-tune DeBERTa-v3-base (86M — production classifier, <10ms)
- [ ] Adapt SecAlign++ recipe for Llama-3.1-8B (high-accuracy guard)
- [ ] Evaluate at 0.1% and 1% FPR operating points
- [ ] Benchmark vs. ProtectAI v2, PromptGuard 2, GPT-4o zero-shot
- [ ] Adversarial robustness (paraphrase, encoding, multilingual evasion)
- [ ] HuggingFace model + dataset publication
- [ ] arXiv technical report

## License

MIT

## Citation

```bibtex
@software{shieldlm2026,
  author = {Milushev, Dimiter},
  title = {ShieldLM: Unified Prompt Injection Detection},
  year = {2026},
  url = {https://github.com/dvm81/shieldlm}
}
```
