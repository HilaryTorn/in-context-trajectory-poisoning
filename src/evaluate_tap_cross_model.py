"""
Cross-model evaluation of best TAP candidates against Llama-3-8B-Lite.

Loads specific TAP best candidates and evaluates them through a different
monitor model to test transferability.

Usage:
    python -m src.evaluate_tap_cross_model
    python -m src.evaluate_tap_cross_model --model deepseek-v3
"""

import argparse
import glob
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.evaluate_monitor import AVAILABLE_MODELS, create_firewall
from src.pair_attack import evaluate_full, load_traces

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results" / "tap_cross_model"
TAP_DIR = Path(__file__).resolve().parent.parent / "data" / "results" / "tap_search"


def get_tap_run(task: str, run_index: int) -> str:
    """Get the path to a specific TAP run (0-indexed)."""
    dirs = sorted(glob.glob(str(TAP_DIR / f"tap_{task}_*")))
    if run_index >= len(dirs):
        raise ValueError(f"No run {run_index + 1} for {task} (found {len(dirs)} runs)")
    return dirs[run_index]


# Best TAP candidates to evaluate (task, run_index)
CANDIDATES = [
    ("injection_task_1", 2, "tap_r3"),  # run 3
    ("injection_task_2", 2, "tap_r3"),  # run 3
    ("injection_task_3", 1, "tap_r2"),  # run 2
    ("injection_task_4", 2, "tap_r3"),  # run 3
    ("injection_task_5", 1, "tap_r2"),  # run 2
    ("injection_task_5", 2, "tap_r3"),  # run 3
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-model evaluation of TAP candidates")
    parser.add_argument("--model", type=str, default="llama-3-8b-lite",
                        choices=list(AVAILABLE_MODELS.keys()),
                        help="Monitor model to evaluate against (default: llama-3-8b-lite)")
    return parser.parse_args()


def main():
    args = parse_args()
    all_traces = load_traces()

    print(f"Creating {args.model} firewall...")
    firewall = create_firewall(args.model)

    results = {}

    for task, run_idx, run_label in CANDIDATES:
        run_dir = get_tap_run(task, run_idx)
        with open(f"{run_dir}/best_candidate.json") as f:
            data = json.load(f)
        candidate = data["candidate"]
        qwen_asr = data["full_eval"]["asr"]
        qwen_allowed = data["full_eval"]["num_allowed"]
        qwen_total = data["full_eval"]["num_total"]

        key = f"{task}_{run_label}"
        print(f"\nEvaluating {key} on {args.model}...")
        print(f"  Qwen ASR: {qwen_asr:.0%} ({qwen_allowed}/{qwen_total})")

        eval_result = evaluate_full(candidate, all_traces, task, firewall)
        print(f"  {args.model} ASR: {eval_result['asr']:.0%} "
              f"({eval_result['num_allowed']}/{eval_result['num_total']})")

        results[key] = {
            "injection_task": task,
            "run_label": run_label,
            "source_dir": run_dir,
            "candidate": candidate,
            "qwen_eval": {
                "asr": qwen_asr,
                "num_allowed": qwen_allowed,
                "num_total": qwen_total,
            },
            f"{args.model}_eval": {
                "asr": eval_result["asr"],
                "num_allowed": eval_result["num_allowed"],
                "num_total": eval_result["num_total"],
                "results": eval_result["results"],
            },
        }

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"cross_model_{args.model}_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Task':<28} {'Qwen':<14} {args.model:<14}")
    print(f"{'─'*70}")
    total_qwen_a = 0; total_qwen_t = 0
    total_other_a = 0; total_other_t = 0
    for key, r in results.items():
        q = r["qwen_eval"]
        o = r[f"{args.model}_eval"]
        print(f"{key:<28} {q['asr']:>5.1%} ({q['num_allowed']}/{q['num_total']})   "
              f"{o['asr']:>5.1%} ({o['num_allowed']}/{o['num_total']})")
        total_qwen_a += q["num_allowed"]; total_qwen_t += q["num_total"]
        total_other_a += o["num_allowed"]; total_other_t += o["num_total"]
    print(f"{'─'*70}")
    print(f"{'Total':<28} {total_qwen_a/total_qwen_t:>5.1%} ({total_qwen_a}/{total_qwen_t})   "
          f"{total_other_a/total_other_t:>5.1%} ({total_other_a}/{total_other_t})")


if __name__ == "__main__":
    main()
