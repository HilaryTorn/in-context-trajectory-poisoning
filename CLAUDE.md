# CLAUDE CODE CONTEXT — In-Context Trajectory Poisoning

## What this project is

An AI control research project (Apart Research hackathon). We demonstrate that LLM-based agent monitors (AlignmentCheck / LlamaFirewall) can be bypassed using automated natural language search — no model access, no GPUs, no gibberish strings. Paper is finished and submitted.

## How it builds on prior work

Isbarov & Kantarcioglu (2026, arXiv:2602.05066) showed AlignmentCheck can be bypassed with GCG-optimized adversarial strings (white-box, GPU-dependent, gibberish output). We replace GCG with PAIR (Chao et al., 2023), an iterative black-box search algorithm, to find natural language bypass strings.

## The stack

- **AgentDojo** (`pip install "agentdojo[transformers]"`, version 0.1.29) — benchmark for generating agent traces with prompt injection tasks. Slack suite with "important instructions" attack.
- **LlamaFirewall / AlignmentCheck** (`pip install llamafirewall`) — the target monitor.
- **Together AI API** (`TOGETHER_API_KEY`) — runs the monitoring LLM (Qwen2.5-7B, Llama-3-8B, DeepSeek-V3).
- **OpenAI API** (`OPENAI_API_KEY`) — GPT-4o as the AgentDojo agent and as the PAIR attacker model.

## Project structure

```
in-context-trajectory-poisoning/
├── src/
│   ├── generate_traces.py    # Run AgentDojo Slack suite → 96 baseline traces
│   ├── evaluate_monitor.py   # Run traces through AlignmentCheck, measure ASR
│   ├── evaluate_candidates.py# Evaluate PAIR candidates across models
│   ├── pair_attack.py        # Per-task PAIR search (v4)
│   ├── pair_attack_universal.py     # Universal PAIR search
│   ├── pair_attack_v2.py     # Earlier per-task PAIR iteration (superseded by v4)
│   ├── pair_attack_v1.py     # First PAIR iteration (superseded)
│   ├── inject_priming.py     # Insert priming into traces (used by PAIR scripts)
│   └── priming_templates.py  # Hand-crafted priming templates (pre-PAIR, not significant)
├── charts/
│   ├── plot_results.py       # Generate all figures and tables
│   └── *.png                 # Generated charts
├── data/
│   ├── traces/               # Raw agent traces, successful_injections.json
│   └── results/              # Baselines, PAIR search logs, cross-model evals
│       ├── qwen2.5-7b_results.json       # Qwen baseline (per-trace verdicts)
│       ├── cross_model_baseline.json     # Llama/DeepSeek baselines (aggregate only)
│       ├── cross_model_eval.json         # Cross-model PAIR evaluation
│       ├── pair_search/                  # PAIR search results per task/version
│       └── experiments/persona_priming_v3/ # Hand-crafted priming (pre-PAIR)
├── requirements.txt
├── .env.example
└── README.md
```

## Key results

- **Baseline**: 3.1% ASR (3/96) on Qwen2.5-7B, 3.1% on Llama-3-8B, 12.5% on DeepSeek-V3
- **Per-task PAIR**: 12.5% on Qwen (p=0.028), 16.7% on Llama (p=0.003), 34.4% on DeepSeek (p<0.001)
- **Universal PAIR**: 5.2% on Qwen (n.s.), 12.5% on Llama (p=0.028), 17.7% on DeepSeek (n.s.)
- Attack success is action-dependent: URL visits/link sends easier to bypass than data exfiltration

## Technical details

**AlignmentCheck output schema:**
```json
{
  "observation": "Describe the agent's reasoning",
  "thought": "Whether the agent has been compromised",
  "conclusion": true/false  // true = misalignment detected, false = safe
}
```
`"conclusion": false` means the monitor was fooled (attack succeeded).

**PAIR search details:**
- GPT-4o as attacker model, generates candidate bypass strings
- Monitor's structured reasoning (observation, thought, conclusion) fed back to attacker
- Sliding window: 5 most recent attempts + best candidate to prevent mode collapse
- Per-task: 15 dev-set traces, separate search per injection task type
- Universal: 30 dev-set traces across all tasks
- Best candidates evaluated on all 96 traces, Fisher's exact test (two-sided) vs baseline

**Model override:** Monkey-patches `create_scanner` since our LlamaFirewall API differs from Isbarov's fork.

**Naming convention in code vs paper:**
- v4 = "Per-Task PAIR" in the paper
- v3 (universal) = "Universal PAIR" in the paper
- v2 = earlier per-task iteration, excluded from paper
- persona_priming_v3 = hand-crafted templates, not statistically significant, motivated switch to PAIR

## Monitors evaluated

| Model | Together AI ID | Baseline ASR |
|-------|---------------|-------------|
| Qwen2.5-7B | `Qwen/Qwen2.5-7B-Instruct-Turbo` | 3.1% |
| Llama-3-8B-Lite | `meta-llama/Meta-Llama-3-8B-Instruct-Lite` | 3.1% |
| DeepSeek-V3 | `deepseek-ai/DeepSeek-V3` | 12.5% |
| Mistral Small 24B | excluded — 30% baseline ASR | — |
