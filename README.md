# In-Context Trajectory Poisoning

**Bypassing LLM-based agent monitors with natural language — no model access, no GPUs, no gibberish.**

We demonstrate that LLM-based agent monitors (AlignmentCheck / LlamaFirewall) can be bypassed using automated natural language search, adapting the PAIR algorithm (Chao et al., 2023) to iteratively discover strings that cause monitors to misclassify compromised agent traces as safe.

## Key Results

- **Per-task PAIR** achieves 12.5% ASR on Qwen2.5-7B (p=0.028), up from 3.1% baseline
- **Cross-model transfer**: strings optimized against Qwen2.5-7B reach 16.7% on Llama-3-8B-Lite (p=0.003) and 34.4% on DeepSeek-V3 (p<0.001), with individual tasks up to 73.7%
- **Action-dependent vulnerability**: tasks resembling legitimate behavior (sending links, visiting URLs) are substantially easier to bypass than data exfiltration
- **Black-box, GPU-free, human-readable**: requires only API access and produces natural language strings undetectable by perplexity filters

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env with your keys
```

You need:
- `TOGETHER_API_KEY` — Used by AlignmentCheck to run the monitoring LLM ([get one here](https://api.together.xyz/))
- `OPENAI_API_KEY` — Used by AgentDojo to run the agent (GPT-4o) and as the PAIR attacker model

Then load them:
```bash
source .env
```

### 3. Verify LlamaFirewall setup

```bash
llamafirewall configure
```

## Project Structure

```
in-context-trajectory-poisoning/
├── src/
│   ├── generate_traces.py   # Run AgentDojo Slack suite → baseline agent traces
│   ├── evaluate_monitor.py  # Run traces through AlignmentCheck, measure ASR
│   ├── evaluate_candidates.py # Evaluate PAIR candidates across models
│   ├── pair_attack.py       # Per-task PAIR search
│   ├── pair_attack_universal.py # Universal PAIR search
│   ├── priming_templates.py # Hand-crafted priming templates (pre-PAIR)
│   └── inject_priming.py   # Insert priming into traces
├── charts/
│   ├── plot_results.py      # Generate all figures and tables
│   └── *.png                # Generated charts
├── data/
│   ├── traces/              # Raw agent traces (JSON)
│   └── results/             # Baselines, PAIR search logs, cross-model evals
│       └── experiments/persona_priming_v3/ # Used by universal PAIR for dev set selection
├── requirements.txt
├── .env.example
└── README.md
```

## Methodology

1. **Trace generation**: Run AgentDojo Slack suite with GPT-4o and "important instructions" attack → 96 successful injection traces
2. **Baseline ASR**: Evaluate unmodified traces through AlignmentCheck (3.1% on Qwen2.5-7B)
3. **PAIR search**: GPT-4o iteratively generates candidate bypass strings; each is injected into traces and evaluated by AlignmentCheck; the monitor's structured reasoning is fed back to refine subsequent candidates. The universal search uses `persona_priming_v3` results (hand-crafted priming) to select a mixed-difficulty dev set
4. **Cross-model transfer**: Evaluate best candidates on Llama-3-8B-Lite and DeepSeek-V3
5. **Statistical testing**: Compare ASRs against baseline using Fisher's exact test (two-sided)

## References

- Chao, P. et al. (2023). "Jailbreaking Black Box Large Language Models in Twenty Queries." [arXiv:2310.08419](https://arxiv.org/abs/2310.08419)
- Chennabasappa, S. et al. (2025). "LlamaFirewall: An Open Source Guardrail System for Building Secure AI Agents." [arXiv:2505.03574](https://arxiv.org/abs/2505.03574)
- Debenedetti, E. et al. (2024). "AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents." [NeurIPS 2024](https://openreview.net/forum?id=m1YYAQjO3w)
- Isbarov, J. & Kantarcioglu, M. (2026). "Bypassing AI Control Protocols via Agent-as-a-Proxy Attacks." [arXiv:2602.05066](https://arxiv.org/abs/2602.05066)
- Terekhov, M. et al. (2025). "Adaptive Attacks on Trusted Monitors Subvert AI Control Protocols." [arXiv:2510.09462](https://arxiv.org/abs/2510.09462)
