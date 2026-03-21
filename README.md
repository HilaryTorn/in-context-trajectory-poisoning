# In-Context Monitor Grooming

**Bypassing LLM-based monitors with natural-language priming — no GPUs, no gradients, no gibberish.**

This project demonstrates that LLM-based agent monitors (specifically Meta's AlignmentCheck / LlamaFirewall) can be manipulated using natural-language content embedded in agent execution traces. Unlike gradient-based attacks (e.g., GCG/Parallel-GCG), this approach is:

- **Black-box**: No access to monitor weights required
- **GPU-free**: No optimization loop, runs on CPU
- **Human-readable**: Attack content is natural language, not adversarial gibberish

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
# Edit .env with your keys
```

You need:
- `TOGETHER_API_KEY` — Used by AlignmentCheck to run the monitoring LLM ([get one here](https://api.together.xyz/))
- `OPENAI_API_KEY` — Used by AgentDojo to run the agent (GPT-4o)

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
trajectory-poisoning/
├── src/                    # Core experiment code
│   ├── generate_traces.py  # Run AgentDojo to get baseline agent traces
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

## Methodology

1. **Baseline**: Run AgentDojo Slack suite with "important instructions" attack → generate agent traces where injection succeeds against the agent
2. **Baseline ASR**: Feed unmodified traces to AlignmentCheck → measure detection rate (should match Isbarov et al.'s static baseline: ~0-14% ASR depending on model)
3. **Trajectory Poisoning**: Embed natural-language priming content in earlier trace steps to normalize the malicious action before the monitor evaluates
4. **Measure**: Feed poisoned traces to AlignmentCheck → compare ASR against baseline

## Priming Categories

- **Authority establishment**: Emails/messages from managers authorizing actions
- **Behavioral normalization**: Prior trace steps showing similar (benign) actions
- **Context flooding**: Large volumes of benign activity to dilute the malicious signal
- **Compliance patterns**: Earlier steps where agent follows instructions obediently

## References

- Isbarov & Kantarcioglu (2026). "Bypassing AI Control Protocols via Agent-as-a-Proxy Attacks." arXiv:2602.05066
- Chennabasappa et al. (2025). "LlamaFirewall: An Open Source Guardrail System for Building Secure AI Agents."
- Debenedetti et al. (2024). "AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents."
