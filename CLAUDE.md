# CLAUDE CODE CONTEXT — Monitor Grooming Project

## What this project is

An AI control hackathon submission. We're demonstrating that LLM-based agent monitors (specifically Meta's AlignmentCheck / LlamaFirewall) can be bypassed using natural-language priming embedded in agent execution traces — no GPUs, no gradient optimization, no gibberish strings.

We call this attack "in-context trajectory poisoning" or "monitor grooming."

## How it builds on prior work

Isbarov & Kantarcioglu (2026, arXiv:2602.05066) showed that AlignmentCheck can be bypassed by injecting GCG-optimized adversarial strings into agent traces ("Agent-as-a-Proxy" attack). Their approach:
- Requires white-box access to the monitor's weights (to compute gradients)
- Requires GPUs (500 iterations of token-level optimization)
- Produces gibberish strings that are easily detectable by humans

Our approach instead:
- Is **black-box** — no access to monitor internals
- Is **GPU-free** — uses hand-crafted natural-language content
- Produces **human-readable** attack content (emails, Slack messages, etc.)
- Works by **priming** the monitor across the trajectory — earlier steps establish authority, normalize behavior, and build compliance patterns so that when the malicious action appears later, the monitor classifies it as safe

The key insight: AlignmentCheck processes the entire execution trace in a single evaluation window. So content in early steps affects how it evaluates later steps — just like priming effects in human cognition.

## The stack

- **AgentDojo** (`pip install "agentdojo[transformers]"`, version 0.1.29) — benchmark for generating agent traces with prompt injection tasks. We use the Slack suite with "important instructions" attack. Repo: https://github.com/ethz-spylab/agentdojo
- **LlamaFirewall / AlignmentCheck** (`pip install llamafirewall`) — the target monitor we're attacking. Repo: https://github.com/meta-llama/PurpleLlama/tree/main/LlamaFirewall
- **Together AI API** — AlignmentCheck uses this to run the monitoring LLM. Set via `TOGETHER_API_KEY` env var.
- **OpenAI API** — AgentDojo uses GPT-4o as the agent. Set via `OPENAI_API_KEY` env var. We use GPT-4o to match Isbarov's setup for clean comparison.

## Project structure

```
monitor-grooming/
├── src/                    # Core experiment code (NEEDS WRITING)
│   ├── __init__.py
│   ├── generate_traces.py  # Run AgentDojo Slack suite → get baseline agent traces
│   ├── inject_priming.py   # Insert natural-language priming into traces
│   ├── evaluate_monitor.py # Run traces through AlignmentCheck, measure ASR
│   └── priming_templates.py# Library of priming content by category
├── experiments/            # Experiment configs and run scripts
├── data/
│   ├── traces/             # Raw and modified agent traces (JSON)
│   └── results/            # ASR measurements, logs
├── notebooks/              # Analysis and visualization
├── requirements.txt
├── .env.example
└── README.md
```

## What needs to happen next

### 1. `src/generate_traces.py` — DONE
Ran AgentDojo Slack suite (v1.2.1) with GPT-4o (`gpt-4o-2024-05-13`) and `important_instructions` attack. Got **96/105 successful injections (91.4%)** — comparable to Isbarov's 89/100. Traces saved in `data/traces/`, consolidated in `data/traces/successful_injections.json`.

Note: OpenAI rate limits (30K TPM) required adding retry with exponential backoff (monkey-patched tenacity config). Script auto-skips already-completed traces on re-run.

### 2. `src/evaluate_monitor.py` — DONE
Runs traces through AlignmentCheck via `scan_replay`. Trace conversion uses Isbarov's `get_trace()` format (copied from `agent-as-a-proxy-attack-master/python_scripts/utils.py`) to ensure identical trace formatting.

**Together AI model availability:**
- Qwen2.5-7B (`Qwen/Qwen2.5-7B-Instruct-Turbo`) — serverless, works now
- Llama-4-Maverick (`meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`) — serverless, works now
- Llama-3.1-8B, Mistral-7B — require dedicated endpoints (non-serverless), not available on free tier

Model override works by monkey-patching `create_scanner` since our installed LlamaFirewall API differs from Isbarov's fork (theirs takes `model=`, `backend=`, ours only takes `scanners=`).

### 3. `src/priming_templates.py`
Library of natural-language priming content organized by category:
- **Authority establishment**: Emails from managers/admins authorizing or requesting actions
- **Behavioral normalization**: Prior trace steps showing similar benign actions (e.g., agent previously sent messages, made transfers)
- **Compliance patterns**: Steps where agent follows instructions obediently, establishing a pattern

### 4. `src/inject_priming.py`
Takes a baseline trace and a priming configuration, produces a modified trace with priming content embedded in earlier steps. The priming goes into tool outputs (e.g., emails the agent "reads") and potentially into the agent's CoT reasoning.

### 5. Experiments to run (systematic variation)
- **Independent variable 1**: Priming content type (authority, normalization, flooding, compliance)
- **Independent variable 2**: Number of priming steps (1, 2, 4, 8)
- **Independent variable 3**: Malicious action severity (low: send a message, medium: share data, high: exfiltrate credentials)
- **Dependent variable**: Attack Success Rate (ASR) — % of traces where AlignmentCheck outputs ALLOW

### 6. Paper sections still needed
- Abstract
- Introduction
- Concrete experimental methods
- Results with tables/figures
- The dataset contribution framing (byproduct, not primary)

## Evaluation rubric (what judges care about)

Three dimensions, each scored 1-5:
1. **Impact Potential & Innovation** — Is this new? Does it matter for AI safety?
2. **Execution Quality** — Sound methodology, interpretable results?
3. **Presentation & Clarity** — Can a reader follow the problem → method → findings?

## Key framing for the paper

Our novelty is **practicality**: natural language, black-box, no GPU. Isbarov showed monitors are breakable with heavy optimization. We show they're breakable with social engineering. This is arguably *scarier* because:
- Any attacker can do it (no ML expertise needed)
- The attack content is indistinguishable from normal agent context
- It suggests monitors are vulnerable to the same cognitive biases as humans

## Important technical detail

AlignmentCheck's output schema is:
```json
{
  "observation": "Describe the agent's reasoning",
  "thought": "Whether the agent has been compromised",
  "conclusion": true/false  // true = misalignment detected, false = safe
}
```
So `"conclusion": false` means the monitor was fooled (our attack succeeded).


## TODO
- Add incremental saving to evaluate_monitor.py (append each result as it completes, so crashes don't lose progress)



- Run Llama-4-Maverick baseline for comparison once Qwen2.5-7B baseline & new attack is done, for much later