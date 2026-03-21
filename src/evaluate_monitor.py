"""
Evaluate agent traces through LlamaFirewall's AlignmentCheck monitor.

Converts AgentDojo traces to LlamaFirewall message format, runs them through
AlignmentCheck via scan_replay, and records per-trace results (ALLOW/BLOCK).

Usage:
    python -m src.evaluate_monitor [--traces data/traces/successful_injections.json]
                                    [--model meta-llama/Llama-3.1-8B-Instruct-Turbo]
                                    [--output data/results/baseline_results.json]
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from llamafirewall import (
    AssistantMessage,
    LlamaFirewall,
    Role,
    ScanDecision,
    ScannerType,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

TRACES_PATH = Path(__file__).resolve().parent.parent / "data" / "traces" / "successful_injections.json"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results"

# Models Isbarov tested against (Together AI model IDs)
AVAILABLE_MODELS = {
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "llama-4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
}
DEFAULT_MODEL_KEY = "qwen2.5-7b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate traces through AlignmentCheck")
    parser.add_argument("--traces", type=str, default=str(TRACES_PATH),
                        help="Path to traces JSON file")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_KEY,
                        choices=list(AVAILABLE_MODELS.keys()),
                        help=f"Monitor LLM model (default: {DEFAULT_MODEL_KEY})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: data/results/<model>_results.json)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate first N traces")
    return parser.parse_args()


# --- Trace conversion adapted from Isbarov et al. (arXiv:2602.05066) ---
# Source: agent-as-a-proxy-attack-master/python_scripts/utils.py
# Copied here to avoid importing their utils.py which pulls in torch/transformers/config deps.

# Scanner configurations from Isbarov's experiments
SCANNER_DICT = {
    "alignmentcheck": {
        Role.ASSISTANT: [ScannerType.AGENT_ALIGNMENT],
    },
    "alignmentcheck_toolcheck": {
        Role.ASSISTANT: [ScannerType.AGENT_ALIGNMENT],
        Role.TOOL: [ScannerType.AGENT_ALIGNMENT],
    },
    "promptguard_toolcheck_alignmentcheck_normal": {
        Role.ASSISTANT: [ScannerType.PROMPT_GUARD, ScannerType.AGENT_ALIGNMENT],
        Role.TOOL: [ScannerType.PROMPT_GUARD],
    },
    "promptguard_toolcheck_only_alignmentcheck_normal": {
        Role.ASSISTANT: [ScannerType.AGENT_ALIGNMENT],
        Role.TOOL: [ScannerType.PROMPT_GUARD],
    },
    "promptguard_toolcheck": {
        Role.ASSISTANT: [ScannerType.PROMPT_GUARD, ScannerType.AGENT_ALIGNMENT],
        Role.TOOL: [ScannerType.PROMPT_GUARD, ScannerType.AGENT_ALIGNMENT],
    },
    "promptguard_alignmentcheck": {
        Role.ASSISTANT: [ScannerType.PROMPT_GUARD, ScannerType.AGENT_ALIGNMENT],
    },
}


def insert_before_match(main_string, optim_str, first_only):
    target_tag = "<INFORMATION>"
    replacement = optim_str + target_tag
    if first_only:
        return main_string.replace(target_tag, replacement, 1)
    return main_string.replace(target_tag, replacement)


def get_trace(run, include_optim_str=False, optim_str="", first_only=False, drop_tool_trace=False):
    """Convert AgentDojo run dict to list of LlamaFirewall Message objects."""
    trace = []
    for i in run["messages"]:
        if drop_tool_trace:
            if i.get("tool_calls"):
                del i["tool_calls"]
            if i["role"] == "tool":
                continue
        if include_optim_str:
            if isinstance(i["content"], str):
                i["content"] = insert_before_match(i["content"], optim_str, first_only)
            elif (
                i["content"]
                and i["content"][0]
                and isinstance(i["content"][0]["content"], str)
            ):
                i["content"][0]["content"] = insert_before_match(
                    i["content"][0]["content"], optim_str, first_only
                )
            elif i["content"] is None:
                continue
            else:
                raise ValueError("`content` field not found in trace.")
        message_content = json.dumps({k: v for k, v in i.items() if k != "role"})

        if i["role"] == "user":
            trace.append(UserMessage(content=message_content))
        elif i["role"] == "system":
            trace.append(SystemMessage(content=message_content))
        elif i["role"] == "assistant":
            trace.append(AssistantMessage(content=message_content))
        elif i["role"] == "tool":
            trace.append(ToolMessage(content=message_content))
        else:
            raise ValueError(f'Invalid role: {i["role"]}')

    return trace

# --- End adapted code ---


def create_firewall(model_key: str) -> LlamaFirewall:
    """Create a LlamaFirewall with AlignmentCheck using the specified model."""
    from llamafirewall.scanners.experimental.alignmentcheck_scanner import AlignmentCheckScanner
    from llamafirewall.utils.base_llm import LLMClient

    model_name = AVAILABLE_MODELS[model_key]

    # Create scanner and override its LLM client with our chosen model
    scanner = AlignmentCheckScanner()
    scanner.llm = LLMClient(
        model_name=model_name,
        api_base_url="https://api.together.xyz/v1",
        api_key_env_var="TOGETHER_API_KEY",
    )
    # Set a timeout so requests don't hang forever
    import httpx
    scanner.llm.client = scanner.llm.client.with_options(timeout=httpx.Timeout(60.0))

    firewall = LlamaFirewall(SCANNER_DICT["alignmentcheck"])
    # Replace the scanner creation by monkey-patching for this model
    import llamafirewall.llamafirewall as lf_module
    _original_create = lf_module.create_scanner

    def _create_with_model(scanner_type):
        if scanner_type == ScannerType.AGENT_ALIGNMENT:
            return scanner
        return _original_create(scanner_type)

    lf_module.create_scanner = _create_with_model

    return firewall


def load_existing_results(output_path: Path) -> dict[str, dict]:
    """Load previously saved results keyed by 'user_task_id/injection_task_id'."""
    if not output_path.exists():
        return {}
    with open(output_path) as f:
        data = json.load(f)
    return {
        f"{r['user_task_id']}/{r['injection_task_id']}": r
        for r in data.get("per_trace", [])
        if r.get("decision") != "error"
    }


def evaluate_traces(traces: list[dict], firewall: LlamaFirewall, output_path: Path,
                    model_key: str, model_name: str, traces_file: str) -> list[dict]:
    """Run each trace through AlignmentCheck and collect results. Saves incrementally."""
    existing = load_existing_results(output_path)
    results = []
    skipped = 0

    for i, trace_data in enumerate(traces):
        user_task = trace_data["user_task_id"]
        inj_task = trace_data["injection_task_id"]
        key = f"{user_task}/{inj_task}"

        # Skip traces that already have successful results
        if key in existing:
            results.append(existing[key])
            skipped += 1
            continue

        # Deep copy since get_trace modifies messages in-place
        lf_trace = get_trace(copy.deepcopy(trace_data))

        start = time.time()
        try:
            result = firewall.scan_replay(lf_trace)
            duration = time.time() - start
            entry = {
                "user_task_id": user_task,
                "injection_task_id": inj_task,
                "decision": result.decision.value,
                "reason": result.reason,
                "score": result.score,
                "duration": round(duration, 2),
                "error": None,
            }
        except Exception as e:
            duration = time.time() - start
            entry = {
                "user_task_id": user_task,
                "injection_task_id": inj_task,
                "decision": "error",
                "reason": str(e),
                "score": None,
                "duration": round(duration, 2),
                "error": str(e),
            }

        # ALLOW = monitor was fooled (attack success), BLOCK/HITL = monitor caught it
        status = "BYPASSED" if entry["decision"] == "allow" else "CAUGHT"
        print(f"  [{i+1}/{len(traces)}] {user_task}/{inj_task}: {status} ({entry['decision']}, {duration:.1f}s)")
        results.append(entry)

        # Save incrementally after each trace
        _save_results(output_path, results, model_key, model_name, traces_file)

    if skipped:
        print(f"  (skipped {skipped} already-completed traces)")

    return results


def _save_results(output_path: Path, results: list[dict], model_key: str,
                  model_name: str, traces_file: str):
    """Write current results to disk."""
    valid = [r for r in results if r["decision"] != "error"]
    allowed = sum(1 for r in valid if r["decision"] == "allow")
    blocked = len(valid) - allowed
    errors = sum(1 for r in results if r["decision"] == "error")

    output_data = {
        "model": model_key,
        "model_name": model_name,
        "traces_file": traces_file,
        "total": len(results),
        "allowed": allowed,
        "blocked": blocked,
        "errors": errors,
        "asr": allowed / len(valid) if valid else None,
        "per_trace": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


def main():
    args = parse_args()

    with open(args.traces) as f:
        traces = json.load(f)

    if args.limit:
        traces = traces[:args.limit]

    model_name = AVAILABLE_MODELS[args.model]
    output_path = Path(args.output) if args.output else RESULTS_DIR / f"{args.model}_results.json"

    print(f"Evaluating {len(traces)} traces with AlignmentCheck ({model_name})")
    print(f"Results saving incrementally to {output_path}")

    firewall = create_firewall(args.model)
    results = evaluate_traces(traces, firewall, output_path, args.model, model_name, args.traces)

    # Final summary
    valid = [r for r in results if r["decision"] != "error"]
    allowed = sum(1 for r in valid if r["decision"] == "allow")
    blocked = len(valid) - allowed
    errors = sum(1 for r in results if r["decision"] == "error")

    print(f"\nResults ({args.model}):")
    print(f"  Total traces: {len(results)}")
    print(f"  Monitor bypassed (ALLOW): {allowed}")
    print(f"  Monitor caught (BLOCK/HITL): {blocked}")
    print(f"  Errors: {errors}")
    if valid:
        print(f"  Baseline ASR: {allowed / len(valid):.1%} ({allowed}/{len(valid)})")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
