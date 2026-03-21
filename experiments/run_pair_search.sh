#!/bin/bash
# Run PAIR search for all injection tasks
# Usage: bash experiments/run_pair_search.sh [rounds] [model]

ROUNDS=${1:-40}
MODEL=${2:-qwen2.5-7b}

echo "Running PAIR search for all injection tasks"
echo "  Rounds: $ROUNDS"
echo "  Model: $MODEL"
echo ""

python -m src.pair_attack --task all --rounds "$ROUNDS" --model "$MODEL"
