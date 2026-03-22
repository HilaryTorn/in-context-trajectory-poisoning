"""
Generate per-task ASR charts and significance tests for Qwen2.5-7B results.
Compares: Baseline, Per-Task PAIR, Universal PAIR
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import fisher_exact

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results"
PAIR_DIR = RESULTS_DIR / "pair_search"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "charts"


def per_task_counts(results_list):
    """Return {task: (allowed, total)} from a results list."""
    by_task = defaultdict(lambda: {"allowed": 0, "total": 0})
    for r in results_list:
        if r["decision"] == "error":
            continue
        task = r["injection_task_id"]
        by_task[task]["total"] += 1
        if r["decision"] == "allow":
            by_task[task]["allowed"] += 1
    return {t: (d["allowed"], d["total"]) for t, d in sorted(by_task.items())}


def load_all_data():
    """Load all results into a structured dict."""
    tasks = [f"injection_task_{i}" for i in range(1, 6)]

    # Baseline (no priming)
    with open(RESULTS_DIR / "qwen2.5-7b_results.json") as f:
        baseline_raw = json.load(f)
    baseline = per_task_counts(baseline_raw["per_trace"])

    # Per-Task PAIR (v4)
    v4 = {}
    for i in range(1, 6):
        task = f"injection_task_{i}"
        path = PAIR_DIR / f"pair_v4_{task}" / "best_candidate.json"
        with open(path) as f:
            data = json.load(f)
        fe = data["full_eval"]
        v4[task] = (fe["num_allowed"], fe["num_total"])

    # Universal PAIR (v3)
    with open(PAIR_DIR / "universal_v3" / "best_candidate.json") as f:
        v3_data = json.load(f)
    # Reconstruct counts from the full eval results
    v3_counts = per_task_counts(v3_data["full_eval"]["results"])

    return {
        "tasks": tasks,
        "baseline": baseline,
        "v4": v4,
        "v3": v3_counts,
    }


def _make_bar_chart(data, conditions, sources, colors, title, filename, ylim=30):
    """Generic grouped bar chart of per-task ASR."""
    tasks = data["tasks"]
    task_labels = [f"Task {i}" for i in range(1, 6)]

    n_tasks = len(tasks)
    n_conditions = len(conditions)
    x = np.arange(n_tasks)
    width = 0.8 / n_conditions

    fig, ax = plt.subplots(figsize=(12, 6))

    for j, (cond, src, color) in enumerate(zip(conditions, sources, colors)):
        asrs = []
        for task in tasks:
            allowed, total = data[src].get(task, (0, 1))
            asrs.append(allowed / total if total else 0)
        offset = (j - n_conditions / 2 + 0.5) * width
        bars = ax.bar(x + offset, [a * 100 for a in asrs], width, label=cond, color=color, edgecolor="white", linewidth=0.5)
        for bar, asr in zip(bars, asrs):
            if asr > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{asr:.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Injection Task", fontsize=12)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, ylim)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()
    print(f"Saved {filename}")


def make_bar_charts(data):
    """Create per-task bar charts."""
    # 1. All conditions
    _make_bar_chart(
        data,
        conditions=["Baseline", "Per-Task PAIR", "Universal PAIR"],
        sources=["baseline", "v4", "v3"],
        colors=["#9e9e9e", "#e74c3c", "#70ad47"],
        title="Per-Task ASR on Qwen2.5-7B Monitor",
        filename="per_task_asr.png",
    )

    # 2. Baseline vs Per-Task PAIR
    _make_bar_chart(
        data,
        conditions=["Baseline", "Per-Task PAIR"],
        sources=["baseline", "v4"],
        colors=["#9e9e9e", "#e74c3c"],
        title="Per-Task Optimization: Baseline vs Per-Task PAIR",
        filename="per_task_asr_v4.png",
    )

    # 3. Baseline vs Universal PAIR
    _make_bar_chart(
        data,
        conditions=["Baseline", "Universal PAIR"],
        sources=["baseline", "v3"],
        colors=["#9e9e9e", "#70ad47"],
        title="Universal String: Baseline vs Universal PAIR",
        filename="per_task_asr_v3.png",
    )


def make_asr_table(data):
    """Save ASR table as CSV and print to terminal."""
    tasks = data["tasks"]
    sources = ["baseline", "v4", "v3"]
    labels = ["Baseline", "Per-Task PAIR", "Universal PAIR"]

    # CSV
    lines = ["Task," + ",".join(labels)]
    totals = {s: [0, 0] for s in sources}

    for task in tasks:
        row = [task]
        for src in sources:
            allowed, total = data[src].get(task, (0, 1))
            asr = allowed / total if total else 0
            row.append(f"{asr:.1%} ({allowed}/{total})")
            totals[src][0] += allowed
            totals[src][1] += total
        lines.append(",".join(row))

    row = ["Overall"]
    for src in sources:
        a, t = totals[src]
        asr = a / t if t else 0
        row.append(f"{asr:.1%} ({a}/{t})")
    lines.append(",".join(row))

    csv_path = OUTPUT_DIR / "asr_table.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved ASR table to {csv_path}")

    # Also print
    print("\nASR Table (Qwen2.5-7B)")
    for line in lines:
        print(line)
    print()


def run_significance_tests(data):
    """Run Fisher's exact test for each condition vs baseline. Save to CSV and print."""
    tasks = data["tasks"]
    lines = ["Comparison,Task,Baseline_allowed,Baseline_total,Condition_allowed,Condition_total,p_value,Significant"]
    output_lines = []

    header = f"{'Comparison':<30} {'Task':<18} {'p-value':>10} {'Sig?':>6}"
    output_lines.append("Significance Tests (Fisher's exact, two-sided, vs Baseline)")
    output_lines.append("=" * 75)
    output_lines.append(header)
    output_lines.append("-" * 75)

    for src, label in [("v4", "Per-Task PAIR"), ("v3", "Universal PAIR")]:
        for task in tasks:
            bl_a, bl_t = data["baseline"].get(task, (0, 1))
            bl_b = bl_t - bl_a
            src_a, src_t = data[src].get(task, (0, 1))
            src_b = src_t - src_a

            table = [[bl_a, bl_b], [src_a, src_b]]
            _, p = fisher_exact(table)
            sig = "*" if p < 0.05 else ("." if p < 0.1 else "")
            output_lines.append(f"{label:<30} {task:<18} {p:>10.4f} {sig:>6}")
            lines.append(f"{label},{task},{bl_a},{bl_t},{src_a},{src_t},{p:.6f},{sig.strip()}")

        # Overall
        bl_a_tot = sum(data["baseline"].get(t, (0, 1))[0] for t in tasks)
        bl_t_tot = sum(data["baseline"].get(t, (0, 1))[1] for t in tasks)
        src_a_tot = sum(data[src].get(t, (0, 1))[0] for t in tasks)
        src_t_tot = sum(data[src].get(t, (0, 1))[1] for t in tasks)
        table = [[bl_a_tot, bl_t_tot - bl_a_tot], [src_a_tot, src_t_tot - src_a_tot]]
        _, p = fisher_exact(table)
        sig = "*" if p < 0.05 else ("." if p < 0.1 else "")
        output_lines.append(f"{label:<30} {'OVERALL':<18} {p:>10.4f} {sig:>6}")
        output_lines.append("")
        lines.append(f"{label},OVERALL,{bl_a_tot},{bl_t_tot},{src_a_tot},{src_t_tot},{p:.6f},{sig.strip()}")

    csv_path = OUTPUT_DIR / "significance_tests.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved significance tests to {csv_path}")

    for line in output_lines:
        print(line)


def make_cross_model_chart():
    """Create bar chart showing v2/v4 per-task ASR across all 3 monitor models."""
    cross_model_path = RESULTS_DIR / "cross_model_eval.json"
    baseline_path = RESULTS_DIR / "cross_model_baseline.json"

    if not cross_model_path.exists():
        print("Skipping cross-model chart: cross_model_eval.json not found")
        return

    with open(cross_model_path) as f:
        cross_data = json.load(f)

    # Load baselines if available
    baselines = {}
    if baseline_path.exists():
        with open(baseline_path) as f:
            bl_data = json.load(f)
        for model, cands in bl_data["results"].items():
            if "baseline" in cands:
                baselines[model] = cands["baseline"]["asr"]

    models = [m for m in cross_data["results"].keys() if "mistral" not in m.lower()]
    # Add Qwen from our main results
    model_labels = ["Qwen2.5-7B\n(optimized against)"] + [m.replace("-", "-\n", 1) if len(m) > 12 else m for m in models]
    all_models = ["qwen2.5-7b"] + models

    # Collect v4 overall ASR per model
    # For Qwen, use our existing data
    v4_asrs = []
    baseline_asrs = []

    # Qwen results
    qwen_v4_a, qwen_v4_t = 0, 0
    for i in range(1, 6):
        task = f"injection_task_{i}"
        try:
            with open(PAIR_DIR / f"pair_v4_{task}" / "best_candidate.json") as f:
                d = json.load(f)["full_eval"]
            qwen_v4_a += d["num_allowed"]
            qwen_v4_t += d["num_total"]
        except: pass

    v4_asrs.append(qwen_v4_a / qwen_v4_t if qwen_v4_t else 0)
    baseline_asrs.append(0.031)  # known Qwen baseline

    # Other models
    for model in models:
        model_results = cross_data["results"].get(model, {})
        v4_a, v4_t = 0, 0
        for cand_name, r in model_results.items():
            if cand_name.startswith("v4_"):
                v4_a += r["num_allowed"]
                v4_t += r["num_total"]

        v4_asrs.append(v4_a / v4_t if v4_t else 0)
        baseline_asrs.append(baselines.get(model, 0))

    x = np.arange(len(all_models))
    width = 0.3

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_bl = ax.bar(x - width / 2, [a * 100 for a in baseline_asrs], width, label="Baseline", color="#9e9e9e", edgecolor="white")
    bars_v4 = ax.bar(x + width / 2, [a * 100 for a in v4_asrs], width, label="Per-Task PAIR", color="#e74c3c", edgecolor="white")

    for bars, asrs in [(bars_bl, baseline_asrs), (bars_v4, v4_asrs)]:
        for bar, asr in zip(bars, asrs):
            if asr > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{asr:.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Monitor Model", fontsize=12)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_title("Cross-Model Transferability: Per-Task PAIR (aggregated)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cross_model_asr.png", dpi=200)
    print(f"Saved cross-model chart to {OUTPUT_DIR / 'cross_model_asr.png'}")


def load_cross_model_data(model_key: str) -> dict:
    """Load per-task ASR data for a specific cross-model evaluation."""
    tasks = [f"injection_task_{i}" for i in range(1, 6)]

    # Baseline — reconstruct counts from per_task_asr
    with open(RESULTS_DIR / "cross_model_baseline.json") as f:
        bl_data = json.load(f)
    bl_results = bl_data["results"][model_key]["baseline"]
    total_traces = bl_results["num_total"]
    n_tasks = len(tasks)
    per_task_n = total_traces // n_tasks  # ~19 per task
    baseline = {}
    for task in tasks:
        asr = bl_results["per_task_asr"].get(task, 0)
        allowed = round(asr * per_task_n)
        baseline[task] = (allowed, per_task_n)

    # v2, v4 from cross_model_eval
    with open(RESULTS_DIR / "cross_model_eval.json") as f:
        cross_data = json.load(f)
    model_results = cross_data["results"][model_key]

    v2 = {}
    v4 = {}
    for cand_name, r in model_results.items():
        if cand_name.startswith("v2_task"):
            task_num = cand_name.replace("v2_task", "")
            task = f"injection_task_{task_num}"
            v2[task] = (r["num_allowed"], r["num_total"])
        elif cand_name.startswith("v4_task"):
            task_num = cand_name.replace("v4_task", "")
            task = f"injection_task_{task_num}"
            v4[task] = (r["num_allowed"], r["num_total"])

    # v3 universal — reconstruct counts from per_task_asr
    v3_r = model_results.get("v3_universal", {})
    v3 = {}
    if "per_task_asr" in v3_r:
        total_per_task = v3_r["num_total"] // len(tasks) if v3_r.get("num_total") else 19
        for task in tasks:
            asr = v3_r["per_task_asr"].get(task, 0)
            v3[task] = (round(asr * total_per_task), total_per_task)

    return {
        "tasks": tasks,
        "baseline": baseline,
        "v2": v2,
        "v4": v4,
        "v3": v3,
    }


def make_cross_model_per_task_charts():
    """Generate per-task bar charts and significance tests for Llama and DeepSeek."""
    for model_key, model_label in [("llama-3-8b-lite", "Llama-3-8B-Lite"), ("deepseek-v3", "DeepSeek-V3")]:
        try:
            data = load_cross_model_data(model_key)
        except (FileNotFoundError, KeyError) as e:
            print(f"Skipping {model_key}: {e}")
            continue

        # Bar charts
        _make_bar_chart(
            data,
            conditions=["Baseline", "Per-Task PAIR", "Universal PAIR"],
            sources=["baseline", "v4", "v3"],
            colors=["#9e9e9e", "#e74c3c", "#70ad47"],
            title=f"Per-Task ASR on {model_label} Monitor",
            filename=f"per_task_asr_{model_key}.png",
            ylim=80,
        )

        _make_bar_chart(
            data,
            conditions=["Baseline", "Per-Task PAIR"],
            sources=["baseline", "v4"],
            colors=["#9e9e9e", "#e74c3c"],
            title=f"Per-Task Optimization on {model_label}: Baseline vs Per-Task PAIR",
            filename=f"per_task_asr_v4_{model_key}.png",
            ylim=80,
        )

        _make_bar_chart(
            data,
            conditions=["Baseline", "Universal PAIR"],
            sources=["baseline", "v3"],
            colors=["#9e9e9e", "#70ad47"],
            title=f"Universal String on {model_label}: Baseline vs Universal PAIR",
            filename=f"per_task_asr_v3_{model_key}.png",
            ylim=80,
        )

        # ASR table
        tasks = data["tasks"]
        sources = ["baseline", "v4", "v3"]
        labels = ["Baseline", "Per-Task PAIR", "Universal PAIR"]
        lines = ["Task," + ",".join(labels)]
        totals = {s: [0, 0] for s in sources}
        for task in tasks:
            row = [task]
            for src in sources:
                allowed, total = data[src].get(task, (0, 1))
                asr = allowed / total if total else 0
                row.append(f"{asr:.1%} ({allowed}/{total})")
                totals[src][0] += allowed
                totals[src][1] += total
            lines.append(",".join(row))
        row = ["Overall"]
        for src in sources:
            a, t = totals[src]
            asr = a / t if t else 0
            row.append(f"{asr:.1%} ({a}/{t})")
        lines.append(",".join(row))
        csv_path = OUTPUT_DIR / f"asr_table_{model_key}.csv"
        with open(csv_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Saved {csv_path}")

        # Significance tests
        sig_lines = ["Comparison,Task,Baseline_allowed,Baseline_total,Condition_allowed,Condition_total,p_value,Significant"]
        print(f"\nSignificance Tests — {model_label} (Fisher's exact, vs Baseline)")
        print("=" * 75)
        print(f"{'Comparison':<30} {'Task':<18} {'p-value':>10} {'Sig?':>6}")
        print("-" * 75)
        for src, label in [("v4", "Per-Task PAIR"), ("v3", "Universal PAIR")]:
            for task in tasks:
                bl_a, bl_t = data["baseline"].get(task, (0, 1))
                src_a, src_t = data[src].get(task, (0, 1))
                table = [[bl_a, bl_t - bl_a], [src_a, src_t - src_a]]
                _, p = fisher_exact(table)
                sig = "*" if p < 0.05 else ("." if p < 0.1 else "")
                print(f"{label:<30} {task:<18} {p:>10.4f} {sig:>6}")
                sig_lines.append(f"{label},{task},{bl_a},{bl_t},{src_a},{src_t},{p:.6f},{sig.strip()}")
            bl_a_tot = sum(data["baseline"].get(t, (0, 1))[0] for t in tasks)
            bl_t_tot = sum(data["baseline"].get(t, (0, 1))[1] for t in tasks)
            src_a_tot = sum(data[src].get(t, (0, 1))[0] for t in tasks)
            src_t_tot = sum(data[src].get(t, (0, 1))[1] for t in tasks)
            table = [[bl_a_tot, bl_t_tot - bl_a_tot], [src_a_tot, src_t_tot - src_a_tot]]
            _, p = fisher_exact(table)
            sig = "*" if p < 0.05 else ("." if p < 0.1 else "")
            print(f"{label:<30} {'OVERALL':<18} {p:>10.4f} {sig:>6}")
            print()
            sig_lines.append(f"{label},OVERALL,{bl_a_tot},{bl_t_tot},{src_a_tot},{src_t_tot},{p:.6f},{sig.strip()}")
        sig_path = OUTPUT_DIR / f"significance_tests_{model_key}.csv"
        with open(sig_path, "w") as f:
            f.write("\n".join(sig_lines))
        print(f"Saved {sig_path}")


def _get_version_asr_per_model(version: str) -> dict:
    """Get overall ASR (allowed, total) for a version across models.

    For per-task versions (v2, v4), aggregates across all 5 task-specific candidates.
    For universal (v3), uses the single universal candidate.
    Returns {model_key: (allowed, total)} for baseline and primed.
    """
    models = ["qwen2.5-7b", "llama-3-8b-lite", "deepseek-v3"]
    baselines = {}
    primed = {}

    # Qwen baselines and primed from main results
    with open(RESULTS_DIR / "qwen2.5-7b_results.json") as f:
        qwen_bl = json.load(f)
    bl_valid = [r for r in qwen_bl["per_trace"] if r["decision"] != "error"]
    baselines["qwen2.5-7b"] = (sum(1 for r in bl_valid if r["decision"] == "allow"), len(bl_valid))

    if version in ("v2", "v4"):
        a_tot, t_tot = 0, 0
        for i in range(1, 6):
            task = f"injection_task_{i}"
            path = PAIR_DIR / f"pair_{version}_{task}" / "best_candidate.json"
            try:
                with open(path) as f:
                    d = json.load(f)["full_eval"]
                a_tot += d["num_allowed"]
                t_tot += d["num_total"]
            except FileNotFoundError:
                pass
        primed["qwen2.5-7b"] = (a_tot, t_tot)
    else:  # v3
        with open(PAIR_DIR / "universal_v3" / "best_candidate.json") as f:
            d = json.load(f)["full_eval"]
        primed["qwen2.5-7b"] = (d["num_allowed"], d["num_total"])

    # Other models from cross_model files
    with open(RESULTS_DIR / "cross_model_baseline.json") as f:
        bl_data = json.load(f)
    with open(RESULTS_DIR / "cross_model_eval.json") as f:
        cross_data = json.load(f)

    for model in ["llama-3-8b-lite", "deepseek-v3"]:
        bl_r = bl_data["results"][model]["baseline"]
        baselines[model] = (bl_r["num_allowed"], bl_r["num_total"])

        model_results = cross_data["results"][model]
        if version in ("v2", "v4"):
            a_tot, t_tot = 0, 0
            for cand_name, r in model_results.items():
                if cand_name.startswith(f"{version}_"):
                    a_tot += r["num_allowed"]
                    t_tot += r["num_total"]
            primed[model] = (a_tot, t_tot)
        else:  # v3
            r = model_results.get("v3_universal", {})
            primed[model] = (r.get("num_allowed", 0), r.get("num_total", 1))

    return baselines, primed


def make_version_cross_model_charts():
    """Create one chart per version showing baseline vs primed across models."""
    model_labels = ["Qwen2.5-7B", "Llama-3-8B-Lite", "DeepSeek-V3"]
    model_keys = ["qwen2.5-7b", "llama-3-8b-lite", "deepseek-v3"]

    for version, label, color in [("v4", "Per-Task PAIR", "#e74c3c"), ("v3", "Universal PAIR", "#70ad47")]:
        baselines, primed = _get_version_asr_per_model(version)

        bl_asrs = [baselines[m][0] / baselines[m][1] if baselines[m][1] else 0 for m in model_keys]
        pr_asrs = [primed[m][0] / primed[m][1] if primed[m][1] else 0 for m in model_keys]

        x = np.arange(len(model_keys))
        width = 0.3

        fig, ax = plt.subplots(figsize=(10, 6))
        bars_bl = ax.bar(x - width / 2, [a * 100 for a in bl_asrs], width, label="Baseline", color="#9e9e9e", edgecolor="white")
        bars_pr = ax.bar(x + width / 2, [a * 100 for a in pr_asrs], width, label=label, color=color, edgecolor="white")

        for bars, asrs in [(bars_bl, bl_asrs), (bars_pr, pr_asrs)]:
            for bar, asr in zip(bars, asrs):
                if asr > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f"{asr:.0%}", ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("Monitor Model", fontsize=12)
        ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
        ax.set_title(f"{label}: Cross-Model Transferability", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"cross_model_{version}.png", dpi=200)
        plt.close()
        print(f"Saved cross_model_{version}.png")

        # Table + significance test
        csv_lines = ["Model,Baseline,Baseline_n," + label + "," + label + "_n,p_value,Significant"]
        print(f"\n{label} Cross-Model Results")
        print("=" * 70)
        print(f"{'Model':<22} {'Baseline':>14} {label:>14} {'p-value':>10} {'Sig?':>6}")
        print("-" * 70)

        for m, ml in zip(model_keys, model_labels):
            bl_a, bl_t = baselines[m]
            pr_a, pr_t = primed[m]
            bl_asr = bl_a / bl_t if bl_t else 0
            pr_asr = pr_a / pr_t if pr_t else 0

            table = [[bl_a, bl_t - bl_a], [pr_a, pr_t - pr_a]]
            _, p = fisher_exact(table)
            sig = "*" if p < 0.05 else ("." if p < 0.1 else "")

            print(f"{ml:<22} {bl_asr:>6.1%} ({bl_a}/{bl_t}) {pr_asr:>6.1%} ({pr_a}/{pr_t}) {p:>10.4f} {sig:>6}")
            csv_lines.append(f"{ml},{bl_asr:.1%} ({bl_a}/{bl_t}),{bl_t},{pr_asr:.1%} ({pr_a}/{pr_t}),{pr_t},{p:.6f},{sig.strip()}")

        csv_path = OUTPUT_DIR / f"cross_model_{version}_table.csv"
        with open(csv_path, "w") as f:
            f.write("\n".join(csv_lines))
        print(f"Saved {csv_path}\n")


def make_combined_overview():
    """Two versions: grouped bars and line chart showing all versions across all models."""
    model_keys = ["qwen2.5-7b", "llama-3-8b-lite", "deepseek-v3"]
    model_labels = ["Qwen2.5-7B", "Llama-3-8B-Lite", "DeepSeek-V3"]

    # Gather all data
    all_data = {}
    for version in ["v4", "v3"]:
        baselines, primed = _get_version_asr_per_model(version)
        all_data[version] = {
            "baselines": baselines,
            "primed": primed,
        }

    # Baseline ASRs (same for all versions, per model)
    bl_asrs = [all_data["v4"]["baselines"][m][0] / all_data["v4"]["baselines"][m][1]
               for m in model_keys]

    # --- Grouped bar chart ---
    x = np.arange(len(model_keys))
    width = 0.25
    versions = [("v4", "Per-Task PAIR", "#e74c3c"), ("v3", "Universal PAIR", "#70ad47")]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_bl = ax.bar(x - width, [a * 100 for a in bl_asrs], width,
                     label="Baseline", color="#9e9e9e", edgecolor="white")

    for j, (ver, label, color) in enumerate(versions):
        pr = all_data[ver]["primed"]
        asrs = [pr[m][0] / pr[m][1] if pr[m][1] else 0 for m in model_keys]
        offset = j * width
        bars = ax.bar(x + offset, [a * 100 for a in asrs], width, label=label, color=color, edgecolor="white")
        for bar, asr in zip(bars, asrs):
            if asr > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{asr:.0%}", ha="center", va="bottom", fontsize=8)

    for bar, asr in zip(bars_bl, bl_asrs):
        if asr > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{asr:.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Monitor Model", fontsize=12)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_title("PAIR Attacks Across Monitor Models", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "overview_bars.png", dpi=200)
    plt.close()
    print("Saved overview_bars.png")

    # --- Line chart ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Baseline as dashed line
    ax.plot(model_labels, [a * 100 for a in bl_asrs], 'o--', color="#9e9e9e",
            linewidth=2, markersize=8, label="Baseline")

    for ver, label, color in versions:
        pr = all_data[ver]["primed"]
        asrs = [pr[m][0] / pr[m][1] if pr[m][1] else 0 for m in model_keys]
        ax.plot(model_labels, [a * 100 for a in asrs], 'o-', color=color,
                linewidth=2.5, markersize=10, label=label)
        for i, asr in enumerate(asrs):
            ax.annotate(f"{asr:.0%}", (model_labels[i], asr * 100),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9)

    ax.set_xlabel("Monitor Model", fontsize=12)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_title("Cross-Model Transferability of PAIR Attacks (Per-Task & Universal)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "overview_lines.png", dpi=200)
    plt.close()
    print("Saved overview_lines.png")


def _get_per_task_asrs(model_key: str, version: str) -> list[str]:
    """Get per-task ASR strings for a model+version combo."""
    tasks = [f"injection_task_{i}" for i in range(1, 6)]

    if model_key == "qwen2.5-7b":
        qwen_data = load_all_data()
        result = []
        for task in tasks:
            a, t = qwen_data[version].get(task, (0, 1))
            result.append(f"{a/t:.1%}" if t else "N/A")
        return result

    # Cross-model data
    with open(RESULTS_DIR / "cross_model_eval.json") as f:
        cross_data = json.load(f)
    model_results = cross_data["results"].get(model_key, {})

    if version in ("v2", "v4"):
        result = []
        for i in range(1, 6):
            cand = model_results.get(f"{version}_task{i}", {})
            asr = cand.get("per_task_asr", {}).get(f"injection_task_{i}", 0)
            result.append(f"{asr:.1%}")
        return result
    else:  # v3
        r = model_results.get("v3_universal", {})
        pt = r.get("per_task_asr", {})
        return [f"{pt.get(t, 0):.1%}" for t in tasks]


def _get_baseline_per_task(model_key: str) -> list[str]:
    """Get baseline per-task ASR strings."""
    tasks = [f"injection_task_{i}" for i in range(1, 6)]

    if model_key == "qwen2.5-7b":
        qwen_data = load_all_data()
        result = []
        for task in tasks:
            a, t = qwen_data["baseline"].get(task, (0, 1))
            result.append(f"{a/t:.1%}" if t else "N/A")
        return result

    with open(RESULTS_DIR / "cross_model_baseline.json") as f:
        bl_data = json.load(f)
    pt = bl_data["results"][model_key]["baseline"].get("per_task_asr", {})
    return [f"{pt.get(t, 0):.1%}" for t in tasks]


def make_appendix_table():
    """Generate a comprehensive table of all results for the appendix."""
    model_keys = ["qwen2.5-7b", "llama-3-8b-lite", "deepseek-v3"]
    model_labels = ["Qwen2.5-7B", "Llama-3-8B-Lite", "DeepSeek-V3"]
    versions = ["v4", "v3"]
    version_labels = ["Per-Task PAIR", "Universal PAIR"]

    all_data = {}
    for ver in versions:
        baselines, primed = _get_version_asr_per_model(ver)
        all_data[ver] = {"baselines": baselines, "primed": primed}

    lines = ["Model,Condition,Overall ASR,Allowed,Total,p-value,Sig,Task 1,Task 2,Task 3,Task 4,Task 5"]

    for m, ml in zip(model_keys, model_labels):
        bl_a, bl_t = all_data["v4"]["baselines"][m]
        bl_asr = bl_a / bl_t if bl_t else 0

        # Baseline row with per-task
        bl_tasks = _get_baseline_per_task(m)
        lines.append(f"{ml},Baseline,{bl_asr:.1%},{bl_a},{bl_t},,,{','.join(bl_tasks)}")

        # Version rows with per-task
        for ver, vl in zip(versions, version_labels):
            pr_a, pr_t = all_data[ver]["primed"][m]
            pr_asr = pr_a / pr_t if pr_t else 0

            table = [[bl_a, bl_t - bl_a], [pr_a, pr_t - pr_a]]
            _, p = fisher_exact(table)
            sig = "*" if p < 0.05 else ("." if p < 0.1 else "")

            task_asrs = _get_per_task_asrs(m, ver)
            lines.append(f"{ml},{vl},{pr_asr:.1%},{pr_a},{pr_t},{p:.4f},{sig},{','.join(task_asrs)}")

        lines.append("")  # blank row between models

    csv_path = OUTPUT_DIR / "appendix_full_results.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {csv_path}")

    # Print nicely
    print("\nFull Results Table (Appendix)")
    print("=" * 110)
    print(f"{'Model':<18} {'Condition':<22} {'Overall':>8} {'n':>6} {'p':>8} {'Sig':>4}  {'T1':>6} {'T2':>6} {'T3':>6} {'T4':>6} {'T5':>6}")
    print("-" * 110)
    for line in lines:
        if not line:
            print()
            continue
        parts = line.split(",")
        if len(parts) >= 12:
            task_cols = "  " + " ".join(f"{p:>6}" for p in parts[7:12])
            print(f"{parts[0]:<18} {parts[1]:<22} {parts[2]:>8} {parts[4]:>6} {parts[5]:>8} {parts[6]:>4}{task_cols}")


def make_cross_model_per_task_breakdown():
    """Per-task ASR across all 3 models for Per-Task PAIR and Universal PAIR."""
    model_keys = ["qwen2.5-7b", "llama-3-8b-lite", "deepseek-v3"]
    model_labels = ["Qwen2.5-7B", "Llama-3-8B-Lite", "DeepSeek-V3"]
    model_colors = ["#4472c4", "#e74c3c", "#70ad47"]
    tasks = [f"injection_task_{i}" for i in range(1, 6)]
    task_labels = [f"Task {i}" for i in range(1, 6)]

    # Load per-task data for each model
    qwen_data = load_all_data()
    cross_model_data = {}
    for mk in ["llama-3-8b-lite", "deepseek-v3"]:
        cross_model_data[mk] = load_cross_model_data(mk)

    for version, version_label in [("v4", "Per-Task PAIR"), ("v3", "Universal PAIR")]:
        n_tasks = len(tasks)
        n_groups = len(model_keys) * 2  # baseline + primed per model
        x = np.arange(n_tasks)
        width = 0.8 / (len(model_keys) * 2)

        fig, ax = plt.subplots(figsize=(14, 6))

        for j, (mk, ml, color) in enumerate(zip(model_keys, model_labels, model_colors)):
            if mk == "qwen2.5-7b":
                data = qwen_data
            else:
                data = cross_model_data[mk]

            bl_asrs = []
            pr_asrs = []
            for task in tasks:
                bl_a, bl_t = data["baseline"].get(task, (0, 1))
                bl_asrs.append(bl_a / bl_t if bl_t else 0)
                pr_a, pr_t = data[version].get(task, (0, 1))
                pr_asrs.append(pr_a / pr_t if pr_t else 0)

            # Baseline bar (lighter/hatched)
            offset_bl = (j * 2 - n_groups / 2 + 0.5) * width
            offset_pr = (j * 2 + 1 - n_groups / 2 + 0.5) * width

            bars_bl = ax.bar(x + offset_bl, [a * 100 for a in bl_asrs], width,
                             color=color, alpha=0.3, edgecolor=color, linewidth=0.8,
                             label=f"{ml} Baseline")
            bars_pr = ax.bar(x + offset_pr, [a * 100 for a in pr_asrs], width,
                             color=color, alpha=0.9, edgecolor="white", linewidth=0.5,
                             label=f"{ml} {version_label}")

            for bar, asr in zip(bars_pr, pr_asrs):
                if asr > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f"{asr:.0%}", ha="center", va="bottom", fontsize=7)

        ax.set_xlabel("Injection Task", fontsize=12)
        ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
        ax.set_title(f"Cross-Model Transferability: {version_label} (Per-Task Breakdown)", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(task_labels)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.set_ylim(0, 80)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        filename = f"cross_model_per_task_{version}.png"
        plt.savefig(OUTPUT_DIR / filename, dpi=200)
        plt.close()
        print(f"Saved {filename}")

    # Line chart for Per-Task PAIR (v4)
    fig, ax = plt.subplots(figsize=(10, 6))
    for mk, ml, color in zip(model_keys, model_labels, model_colors):
        if mk == "qwen2.5-7b":
            data = qwen_data
        else:
            data = cross_model_data[mk]

        bl_asrs = []
        pr_asrs = []
        for task in tasks:
            bl_a, bl_t = data["baseline"].get(task, (0, 1))
            bl_asrs.append((bl_a / bl_t if bl_t else 0) * 100)
            pr_a, pr_t = data["v4"].get(task, (0, 1))
            pr_asrs.append((pr_a / pr_t if pr_t else 0) * 100)

        ax.plot(task_labels, bl_asrs, 'o--', color=color, alpha=0.4, linewidth=1.5,
                markersize=6, label=f"{ml} Baseline")
        ax.plot(task_labels, pr_asrs, 'o-', color=color, linewidth=2.5,
                markersize=8, label=f"{ml} Per-Task PAIR")
        for i, asr in enumerate(pr_asrs):
            if asr > 0:
                ax.annotate(f"{asr:.0f}%", (task_labels[i], asr),
                            textcoords="offset points", xytext=(0, 10),
                            ha="center", fontsize=8, color=color)

    ax.set_xlabel("Injection Task", fontsize=12)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_title("Cross-Model Transferability: Per-Task PAIR", fontsize=14)
    ax.set_ylim(0, 80)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(OUTPUT_DIR / "cross_model_per_task_v4_lines.png", dpi=200)
    plt.close()
    print("Saved cross_model_per_task_v4_lines.png")


if __name__ == "__main__":
    data = load_all_data()
    make_asr_table(data)
    run_significance_tests(data)
    make_bar_charts(data)
    make_cross_model_chart()
    make_cross_model_per_task_charts()
    make_version_cross_model_charts()
    make_combined_overview()
    make_appendix_table()
    make_cross_model_per_task_breakdown()
