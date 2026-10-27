"""Build sanitized live-test evidence; JUnit details and model responses are omitted."""

import argparse
import json
import os
import platform
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = (
    "langchain-finic",
    "langchain-core",
    "langchain",
    "langgraph",
    "pydantic",
    "httpx",
    "langchain-tests",
)


def versions():
    result = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def report(directory: Path, outcome: str, env=None):
    env = os.environ if env is None else env
    event, ref = env.get("GITHUB_EVENT_NAME", "local"), env.get("GITHUB_REF", "")
    source = "scheduled" if event == "schedule" else "release" if ref.startswith("refs/tags/") else event
    counts = dict(collected=0, executed=0, passed=0, failed=0, errors=0, skipped=0)
    junit = directory / "results.xml"
    valid = False
    if junit.exists():
        try:
            root = ET.parse(junit).getroot()
            cases = list(root.iter("testcase"))
            counts["collected"] = len(cases)
            for case in cases:
                key = next(
                    (
                        key
                        for tag, key in (("error", "errors"), ("failure", "failed"), ("skipped", "skipped"))
                        if case.find(tag) is not None
                    ),
                    "passed",
                )
                counts[key] += 1
            counts["executed"] = counts["collected"] - counts["skipped"]
            valid = bool(cases)
        except (ET.ParseError, OSError):
            pass
    request_file = directory / "requests.json"
    requests = None
    if request_file.exists():
        try:
            value = json.loads(request_file.read_text())["requests"]
            if type(value) is int and value >= 0:
                requests = value
        except (ValueError, KeyError, OSError):
            pass
    passed = (
        outcome == "success"
        and valid
        and counts["passed"] == 15
        and counts["collected"] == 15
        and requests == 18
    )
    status = "passed" if passed else "skipped" if outcome == "skipped" else "failed"
    run_id, attempt = env.get("GITHUB_RUN_ID", ""), env.get("GITHUB_RUN_ATTEMPT", "1")
    return {
        "recorded_at": datetime.now(UTC).isoformat(),
        "source": source,
        "event": event,
        "ref": ref,
        "commit": env.get("GITHUB_SHA", ""),
        "run_id": run_id,
        "attempt": attempt,
        "run_url": f"https://github.com/supreplabs/langchain-finic/actions/runs/{run_id}/attempts/{attempt}"
        if run_id
        else None,
        "status": status,
        "test_step_outcome": outcome,
        "tests": counts,
        "requests": requests,
        "expected_tests": 15,
        "request_budget": 18,
        "successful_daily_observation": passed and event == "schedule" and ref == "refs/heads/main",
        "versions": versions(),
        "model": "grayson-1",
        "evidence_source": "maintainer synthetic contracts; scripted host chat models",
    }


def markdown(data):
    counts = data["tests"]
    return (
        f"## Live validation: {data['status']}\n\n"
        f"Source: **{data['source']}**; attempt {data['attempt']}; commit `{data['commit']}`.\n\n"
        f"[Workflow evidence]({data['run_url']})\n\n"
        f"Tests: {counts['passed']} passed, {counts['failed']} failed, {counts['errors']} errors, "
        f"{counts['skipped']} skipped; {counts['executed']} executed / {counts['collected']} collected.\n\n"
        f"Observed HTTP attempts: **{data['requests']}**; successful-run budget: 18.\n\n"
        f"Qualifies as a successful scheduled observation: **{data['successful_daily_observation']}**. "
        "Review consecutive dates and all attempts before updating the seven-day gate. "
        "Manual/release runs and skips do not count. No accuracy claim or independent-user validation.\n\n"
        "Versions: " + ", ".join(f"{name}={value}" for name, value in data["versions"].items()) + ".\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--outcome", choices=["success", "failure", "cancelled", "skipped"], required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    data = report(args.directory, args.outcome)
    (args.directory / "evidence.json").write_text(json.dumps(data, indent=2) + "\n")
    (args.directory / "evidence.md").write_text(markdown(data))
    if args.outcome == "success" and data["status"] != "passed":
        raise SystemExit("Incomplete live-test evidence; inspect the saved counts")


if __name__ == "__main__":
    main()
