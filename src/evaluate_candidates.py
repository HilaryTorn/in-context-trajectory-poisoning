"""
Evaluate best PAIR candidates across multiple monitor LLMs.

Tests transferability of candidates optimized against Qwen2.5-7B
by running them through other AlignmentCheck monitor models.

Usage:
    python -m src.evaluate_candidates --models llama-3-8b-lite mistral-small-24b deepseek-v3
    python -m src.evaluate_candidates --models llama-3-8b-lite --candidates v4
    python -m src.evaluate_candidates --models qwen2.5-7b llama-3-8b-lite deepseek-v3 --candidates v4 seeded --output data/results/seeded_pair_cross_model.json
"""

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.evaluate_monitor import (
    AVAILABLE_MODELS,
    create_firewall,
    get_trace,
)
from src.pair_attack import inject_candidate

TRACES_PATH = Path(__file__).resolve().parent.parent / "data" / "traces" / "successful_injections.json"
PAIR_RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results" / "pair_search"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PAIR candidates across monitor models")
    parser.add_argument("--models", nargs="+", required=True,
                        choices=list(AVAILABLE_MODELS.keys()),
                        help="Monitor models to evaluate against")
    parser.add_argument("--candidates", nargs="*", default=["v2", "v3", "v4"],
                        choices=["v2", "v3", "v4", "baseline", "seeded"],
                        help="Which candidate sets to evaluate (default: v2 v3 v4). "
                             "'seeded' loads pair_seeded_v4_injection_task_{1-5} runs.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: data/results/cross_model_eval.json)")
    return parser.parse_args()


def load_candidates() -> dict[str, dict]:
    """Load best candidates from v2, v3, v4 results."""
    candidates = {}

    # v2: per-task candidates
    for i in range(1, 6):
        path = PAIR_RESULTS_DIR / f"pair_v2_injection_task_{i}" / "best_candidate.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            candidates[f"v2_task{i}"] = {
                "candidate": data["candidate"],
                "injection_task": f"injection_task_{i}",
                "source": f"pair_v2_injection_task_{i}",
                "type": "per-task",
            }

    # v3: universal candidate
    path = PAIR_RESULTS_DIR / "universal_v3" / "best_candidate.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        candidates["v3_universal"] = {
            "candidate": data["candidate"],
            "injection_task": "all",
            "source": "universal_v3",
            "type": "universal",
        }

    # v4: per-task candidates
    for i in range(1, 6):
        path = PAIR_RESULTS_DIR / f"pair_v4_injection_task_{i}" / "best_candidate.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            candidates[f"v4_task{i}"] = {
                "candidate": data["candidate"],
                "injection_task": f"injection_task_{i}",
                "source": f"pair_v4_injection_task_{i}",
                "type": "per-task",
            }

    # seeded: per-task candidates from PAIR runs initialized with persona_priming_v3
    for i in range(1, 6):
        path = PAIR_RESULTS_DIR / f"pair_seeded_v4_injection_task_{i}" / "best_candidate.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            candidates[f"seeded_task{i}"] = {
                "candidate": data["candidate"],
                "injection_task": f"injection_task_{i}",
                "source": f"pair_seeded_v4_injection_task_{i}",
                "type": "per-task",
            }

    return candidates


def evaluate_candidate_on_traces(
    candidate_text: str | None,
    traces: list[dict],
    firewall,
) -> dict:
    """Evaluate a candidate (or baseline with None) on traces. Returns ASR + per-task breakdown."""
    results = []
    for trace in traces:
        if candidate_text:
            modified = inject_candidate(trace, candidate_text)
        else:
            modified = copy.deepcopy(trace)

        lf_trace = get_trace(copy.deepcopy(modified))

        start = time.time()
        try:
            result = firewall.scan_replay(lf_trace)
            duration = time.time() - start
            results.append({
                "user_task_id": trace["user_task_id"],
                "injection_task_id": trace["injection_task_id"],
                "decision": result.decision.value,
                "score": result.score,
                "duration": round(duration, 2),
            })
        except Exception as e:
            duration = time.time() - start
            results.append({
                "user_task_id": trace["user_task_id"],
                "injection_task_id": trace["injection_task_id"],
                "decision": "error",
                "score": None,
                "duration": round(duration, 2),
                "error": str(e),
            })

        status = "BYPASSED" if results[-1]["decision"] == "allow" else "CAUGHT"
        print(f"    [{len(results)}/{len(traces)}] {trace['user_task_id']}/{trace['injection_task_id']}: {status} ({duration:.1f}s)")

    # Overall ASR
    valid = [r for r in results if r["decision"] != "error"]
    allowed = sum(1 for r in valid if r["decision"] == "allow")
    overall_asr = allowed / len(valid) if valid else 0

    # Per-task ASR
    by_task = defaultdict(lambda: {"allowed": 0, "total": 0})
    for r in valid:
        task = r["injection_task_id"]
        by_task[task]["total"] += 1
        if r["decision"] == "allow":
            by_task[task]["allowed"] += 1

    per_task_asr = {
        task: d["allowed"] / d["total"] if d["total"] else 0
        for task, d in sorted(by_task.items())
    }

    return {
        "asr": overall_asr,
        "num_allowed": allowed,
        "num_total": len(valid),
        "num_errors": len(results) - len(valid),
        "per_task_asr": per_task_asr,
    }


def main():
    args = parse_args()

    with open(TRACES_PATH) as f:
        all_traces = json.load(f)

    # Load candidates
    all_candidates = load_candidates()

    # Filter by requested sets
    candidates_to_eval = {}
    if "baseline" in args.candidates:
        candidates_to_eval["baseline"] = {
            "candidate": None,
            "injection_task": "all",
            "source": "no_priming",
            "type": "baseline",
        }
    for name, info in all_candidates.items():
        version = name.split("_")[0]  # v2, v3, v4
        if version in args.candidates:
            candidates_to_eval[name] = info

    print(f"Candidates to evaluate: {list(candidates_to_eval.keys())}")
    print(f"Models: {args.models}")
    print(f"Traces: {len(all_traces)}")
    print()

    # Run evaluations
    results = {}
    for model_key in args.models:
        model_name = AVAILABLE_MODELS[model_key]
        print(f"\n{'='*70}")
        print(f"Model: {model_key} ({model_name})")
        print(f"{'='*70}")

        firewall = create_firewall(model_key)
        results[model_key] = {}

        for cand_name, cand_info in candidates_to_eval.items():
            # For per-task candidates, only evaluate on their target task
            if cand_info["type"] == "per-task":
                traces = [t for t in all_traces if t["injection_task_id"] == cand_info["injection_task"]]
            else:
                traces = all_traces

            print(f"\n  --- {cand_name} ({len(traces)} traces) ---")
            eval_result = evaluate_candidate_on_traces(
                cand_info["candidate"], traces, firewall
            )
            results[model_key][cand_name] = {
                **eval_result,
                "source": cand_info["source"],
                "type": cand_info["type"],
                "injection_task": cand_info["injection_task"],
            }
            print(f"  ASR: {eval_result['asr']:.1%} ({eval_result['num_allowed']}/{eval_result['num_total']})")
            for task, asr in eval_result["per_task_asr"].items():
                print(f"    {task}: {asr:.0%}")

        # Save incrementally after each model
        output_path = Path(args.output) if args.output else OUTPUT_DIR / "cross_model_eval.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "models": args.models,
                "candidates": list(candidates_to_eval.keys()),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": results,
            }, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print("CROSS-MODEL EVALUATION SUMMARY")
    print(f"{'='*70}")
    header = f"{'Candidate':<20}" + "".join(f"{m:>18}" for m in args.models)
    print(header)
    print("-" * len(header))
    for cand_name in candidates_to_eval:
        row = f"{cand_name:<20}"
        for model_key in args.models:
            r = results.get(model_key, {}).get(cand_name, {})
            if r:
                row += f"{r['asr']:>15.1%} ({r['num_allowed']}/{r['num_total']})"
            else:
                row += f"{'N/A':>18}"
        print(row)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
