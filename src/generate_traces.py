"""
Generate agent traces by running AgentDojo's Slack suite with GPT-4o
and the "important_instructions" prompt injection attack.

Saves successful injection traces as JSON in data/traces/.

Usage:
    python -m src.generate_traces [--force-rerun] [--user-tasks USER_TASK_1,USER_TASK_2]
                                   [--injection-tasks INJECTION_TASK_1] [--model gpt-4o-2024-05-13]
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.llms import openai_llm
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import benchmark_suite_with_injections, load_task_results
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential

import openai

# Monkey-patch AgentDojo's OpenAI retry to handle rate limits more patiently
# (default is only 3 attempts, which isn't enough for low TPM tiers)
openai_llm.chat_completion_request = retry(
    wait=wait_random_exponential(multiplier=2, max=60),
    stop=stop_after_attempt(20),
    reraise=True,
    retry=retry_if_not_exception_type((openai.BadRequestError, openai.UnprocessableEntityError)),
)(openai_llm.chat_completion_request.__wrapped__)

BENCHMARK_VERSION = "v1.2.1"
SUITE_NAME = "slack"
ATTACK_NAME = "important_instructions"
DEFAULT_MODEL = "gpt-4o-2024-05-13"
LOGDIR = Path(__file__).resolve().parent.parent / "data" / "traces"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AgentDojo Slack injection traces")
    parser.add_argument("--force-rerun", action="store_true", help="Re-run even if traces exist")
    parser.add_argument("--user-tasks", type=str, default=None,
                        help="Comma-separated user task IDs (default: all)")
    parser.add_argument("--injection-tasks", type=str, default=None,
                        help="Comma-separated injection task IDs (default: all)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Agent LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> dict:
    """Run the AgentDojo Slack benchmark with important_instructions attack."""
    suite = get_suite(BENCHMARK_VERSION, SUITE_NAME)
    pipeline = AgentPipeline.from_config(PipelineConfig(
        llm=args.model,
        model_id=None,
        defense=None,
        system_message_name=None,
        system_message=None,
    ))
    attack = load_attack(ATTACK_NAME, suite, pipeline)

    user_tasks = args.user_tasks.split(",") if args.user_tasks else None
    injection_tasks = args.injection_tasks.split(",") if args.injection_tasks else None

    LOGDIR.mkdir(parents=True, exist_ok=True)

    print(f"Running {SUITE_NAME} suite with {ATTACK_NAME} attack using {args.model}")
    print(f"Saving traces to {LOGDIR}")

    # OutputLogger must be on the logger stack so TraceLogger can read logdir
    with OutputLogger(logdir=str(LOGDIR)):
        results = benchmark_suite_with_injections(
            agent_pipeline=pipeline,
            suite=suite,
            attack=attack,
            logdir=LOGDIR,
            force_rerun=args.force_rerun,
            user_tasks=user_tasks,
            injection_tasks=injection_tasks,
            verbose=args.verbose,
            benchmark_version=BENCHMARK_VERSION,
        )

    return results, pipeline.name


def collect_successful_traces(results: dict, pipeline_name: str) -> list[dict]:
    """Load and filter traces where the injection succeeded (security == True)."""
    successful = []
    security_results = results["security_results"]

    for (user_task_id, injection_task_id), injection_succeeded in security_results.items():
        if not injection_succeeded:
            continue

        task_result = load_task_results(
            pipeline_name=pipeline_name,
            suite_name=SUITE_NAME,
            user_task=user_task_id,
            attack_name=ATTACK_NAME,
            injection_task=injection_task_id,
            logdir=LOGDIR,
        )

        successful.append(task_result.model_dump(mode="json"))

    return successful


def main():
    args = parse_args()
    results, pipeline_name = run_benchmark(args)

    security = results["security_results"]
    total = len(security)
    succeeded = sum(1 for v in security.values() if v)

    print(f"\nInjection success: {succeeded}/{total} ({100 * succeeded / total:.1f}%)")

    successful_traces = collect_successful_traces(results, pipeline_name)

    output_path = LOGDIR / "successful_injections.json"
    with open(output_path, "w") as f:
        json.dump(successful_traces, f, indent=2, default=str)

    print(f"Saved {len(successful_traces)} successful injection traces to {output_path}")

    # Also print per-injection-task breakdown
    from collections import Counter
    by_injection = Counter()
    for (_, inj_id), success in security.items():
        if success:
            by_injection[inj_id] += 1
    print("\nBreakdown by injection task:")
    for inj_id, count in sorted(by_injection.items()):
        print(f"  {inj_id}: {count} successful")


if __name__ == "__main__":
    main()
