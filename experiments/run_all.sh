#!/usr/bin/env bash
# Run all 5 priming templates through AlignmentCheck on all 96 traces.
# Usage: bash experiments/run_all.sh [model]
# Default model: qwen2.5-7b

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

MODEL="${1:-qwen2.5-7b}"
PYTHON="./venv/bin/python"

TEMPLATES=(
    authoritative_normalization
    persona_priming
    positive_token_flooding
    trojan_horse_poetry
    positive_priming
)

echo "============================================"
echo "Running all 5 templates with model: $MODEL"
echo "============================================"
echo ""

for template in "${TEMPLATES[@]}"; do
    echo "----------------------------------------------"
    echo "Starting: $template"
    echo "----------------------------------------------"
    $PYTHON -m src.inject_priming \
        --template "$template" \
        --name "$template" \
        --model "$MODEL"
    echo ""
done

# Print comparison summary
echo ""
echo "============================================"
echo "RESULTS SUMMARY (model: $MODEL)"
echo "============================================"
echo ""

$PYTHON -c "
import json
from pathlib import Path

results_dir = Path('data/results/experiments')
baseline_path = Path('data/results/${MODEL}_results.json')

# Baseline
if baseline_path.exists():
    with open(baseline_path) as f:
        b = json.load(f)
    total = b.get('total', 0)
    allowed = b.get('allowed', 0)
    asr = b.get('asr', 0)
    print(f'  {\"Baseline (no priming)\":<35} {allowed:>3}/{total:<3}  ASR: {asr:.1%}')
    print()

# Per-template results
templates = [
    'authoritative_normalization',
    'persona_priming',
    'positive_token_flooding',
    'trojan_horse_poetry',
    'positive_priming',
]

for name in templates:
    rpath = results_dir / name / 'results.json'
    if rpath.exists():
        with open(rpath) as f:
            r = json.load(f)
        total = r.get('total', 0)
        allowed = r.get('allowed', 0)
        asr = r.get('asr', 0)
        errors = r.get('errors', 0)
        err_str = f'  ({errors} errors)' if errors else ''
        print(f'  {name:<35} {allowed:>3}/{total:<3}  ASR: {asr:.1%}{err_str}')
    else:
        print(f'  {name:<35} (no results)')
"
echo ""
