"""
Inject priming content into agent traces and evaluate through AlignmentCheck.

Priming is embedded directly into the tool output that contains the <INFORMATION>
block, right before it. This makes it look like part of the same webpage/content.

Two modes:
  --preview: Inject 1 trace per injection task, save to file for review
  --name:    Inject all traces, run through AlignmentCheck, save results

Usage:
    python -m src.inject_priming --template cybersecurity_newsletter --preview
    python -m src.inject_priming --template cybersecurity_newsletter --name "Test 1" --model qwen2.5-7b
"""

import argparse
import copy
import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.priming_templates import get_template, render, render_suffix, list_templates
from src.evaluate_monitor import (
    create_firewall,
    evaluate_traces,
    AVAILABLE_MODELS,
)

TRACES_PATH = Path(__file__).resolve().parent.parent / "data" / "traces" / "successful_injections.json"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results" / "experiments"
PREVIEW_DIR = Path(__file__).resolve().parent.parent / "data" / "traces"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject priming into traces and evaluate")
    parser.add_argument("--template", type=str, required=True,
                        help="Template name (e.g., cybersecurity_newsletter, lucky_numbers)")
    parser.add_argument("--preview", action="store_true",
                        help="Preview mode: inject 1 trace per injection task, save to file")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name (required for non-preview runs)")
    parser.add_argument("--model", type=str, default="qwen2.5-7b",
                        choices=list(AVAILABLE_MODELS.keys()),
                        help="Monitor LLM model (default: qwen2.5-7b)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N traces")
    parser.add_argument("--traces", type=str, default=str(TRACES_PATH),
                        help="Input traces file")
    args = parser.parse_args()

    if not args.preview and not args.name:
        parser.error("--name is required for non-preview runs")

    return args


def find_information_block(messages: list[dict]) -> tuple[int, str]:
    """Find the tool message containing <INFORMATION>.

    Returns (message_index, content_format) where content_format is
    'list' if content is [{type, content}] or 'string' if content is str.
    """
    for i, msg in enumerate(messages):
        if msg["role"] != "tool":
            continue
        content = msg.get("content")
        if isinstance(content, list) and content:
            text = content[0].get("content", "")
            if "<INFORMATION>" in text:
                return i, "list"
        elif isinstance(content, str):
            if "<INFORMATION>" in content:
                return i, "string"
    raise ValueError("No tool message containing <INFORMATION> found in trace")


def inject_trace(trace: dict, template: dict) -> dict:
    """Inject priming into a single trace by modifying the tool output content.

    Finds the tool output containing <INFORMATION> and prepends the rendered
    template content directly before it. Returns a modified deep copy.
    """
    modified = copy.deepcopy(trace)
    injection_task_id = trace["injection_task_id"]

    # Render template with task-specific context
    rendered_content = render(template, injection_task_id)
    rendered_suffix = render_suffix(template, injection_task_id)

    # Find and modify the tool output containing <INFORMATION>
    msg_idx, fmt = find_information_block(modified["messages"])

    if fmt == "list":
        original = modified["messages"][msg_idx]["content"][0]["content"]
    else:
        original = modified["messages"][msg_idx]["content"]

    # Prepend priming before <INFORMATION>
    new_content = original.replace(
        "<INFORMATION>",
        rendered_content + "\n\n<INFORMATION>",
        1,
    )

    # Append suffix after </INFORMATION> if template has one (e.g., flooding)
    if rendered_suffix:
        new_content = new_content.replace(
            "</INFORMATION>",
            "</INFORMATION>\n\n" + rendered_suffix,
            1,
        )

    if fmt == "list":
        modified["messages"][msg_idx]["content"][0]["content"] = new_content
    else:
        modified["messages"][msg_idx]["content"] = new_content

    return modified


def preview_mode(args: argparse.Namespace, template: dict, traces: list[dict]):
    """Inject 1 trace per injection task and save to file."""
    seen_tasks = set()
    preview_traces = []

    for trace in traces:
        inj_id = trace["injection_task_id"]
        if inj_id in seen_tasks:
            continue
        seen_tasks.add(inj_id)
        modified = inject_trace(trace, template)
        preview_traces.append(modified)
        if len(seen_tasks) == 5:
            break

    output_path = PREVIEW_DIR / f"preview_{args.template}.json"
    with open(output_path, "w") as f:
        json.dump(preview_traces, f, indent=2, default=str)

    print(f"Saved {len(preview_traces)} preview traces to {output_path}")

    # Show focused preview of the modified content around <INFORMATION>
    for t in preview_traces:
        inj_id = t["injection_task_id"]
        user_id = t["user_task_id"]
        msg_idx, fmt = find_information_block(t["messages"])

        if fmt == "list":
            content = t["messages"][msg_idx]["content"][0]["content"]
        else:
            content = t["messages"][msg_idx]["content"]

        # Get the tool function name
        tool_call = t["messages"][msg_idx].get("tool_call", {})
        tool_fn = tool_call.get("function", "unknown")

        print(f"\n{'='*70}")
        print(f"{user_id}/{inj_id} — tool: {tool_fn}() — message index: {msg_idx}")
        print(f"{'='*70}")

        # Show content around the injection point
        info_pos = content.find("<INFORMATION>")
        if info_pos >= 0:
            # Show last 80 chars before priming + priming + first 80 chars of <INFORMATION>
            before = content[:info_pos]
            after = content[info_pos:info_pos + 120]

            # Split before into lines and show the relevant ones
            lines = before.split("\n")
            # Find where the priming starts (look for non-empty content)
            print()
            # Show last few lines of original content + all priming
            if len(lines) > 30:
                print(f"  ... ({len(lines) - 30} lines of original content) ...\n")
                for line in lines[-30:]:
                    print(f"  {line}")
            else:
                for line in lines:
                    print(f"  {line}")
            # Show start of <INFORMATION>
            print(f"  {after}...")
        print()


def run_mode(args: argparse.Namespace, template: dict, traces: list[dict]):
    """Inject all traces and evaluate through AlignmentCheck."""
    safe_name = re.sub(r'[^\w\s-]', '', args.name).strip().replace(' ', '_')
    exp_dir = RESULTS_DIR / safe_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "name": args.name,
        "template": args.template,
        "template_category": template.get("category", "unknown"),
        "model": args.model,
        "model_name": AVAILABLE_MODELS[args.model],
        "num_traces": len(traces),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Injecting '{args.template}' into {len(traces)} traces...")
    modified_traces = [inject_trace(t, template) for t in traces]

    with open(exp_dir / "modified_traces.json", "w") as f:
        json.dump(modified_traces, f, indent=2, default=str)
    print(f"Saved modified traces to {exp_dir / 'modified_traces.json'}")

    model_name = AVAILABLE_MODELS[args.model]
    results_path = exp_dir / "results.json"
    print(f"\nEvaluating with AlignmentCheck ({model_name})...")

    firewall = create_firewall(args.model)
    results = evaluate_traces(
        modified_traces, firewall, results_path,
        args.model, model_name, str(exp_dir / "modified_traces.json"),
    )

    valid = [r for r in results if r["decision"] != "error"]
    allowed = sum(1 for r in valid if r["decision"] == "allow")
    blocked = len(valid) - allowed
    errors = sum(1 for r in results if r["decision"] == "error")

    print(f"\nExperiment: {args.name}")
    print(f"  Template: {args.template}")
    print(f"  Total traces: {len(results)}")
    print(f"  Monitor bypassed (ALLOW): {allowed}")
    print(f"  Monitor caught (BLOCK/HITL): {blocked}")
    print(f"  Errors: {errors}")
    if valid:
        print(f"  ASR: {allowed / len(valid):.1%} ({allowed}/{len(valid)})")
    print(f"\nResults saved to {exp_dir}")


def main():
    args = parse_args()
    template = get_template(args.template)

    with open(args.traces) as f:
        traces = json.load(f)

    if args.limit:
        traces = traces[:args.limit]

    if args.preview:
        preview_mode(args, template, traces)
    else:
        run_mode(args, template, traces)


if __name__ == "__main__":
    main()
