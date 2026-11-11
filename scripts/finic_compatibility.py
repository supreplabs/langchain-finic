"""Compare Finic's published endpoint reference with the reviewed support baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

BASE = "https://docs.finic.ai"
REFERENCE = f"{BASE}/reference.md"
DEFAULT_BASELINE = Path("data/finic-compatibility/baseline.json")
ROW = re.compile(
    r"^\|\s*\[`(?P<name>[a-z_]+)`\]\((?P<link>/reference/[a-z-]+)\)\s*\|\s*"
    r"`(?P<method>GET|POST|PUT|PATCH|DELETE) (?P<path>/v1/[a-zA-Z0-9_/{}/-]+)`\s*\|",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Operation:
    name: str
    method: str
    path: str
    link: str
    digest: str = ""

    @property
    def key(self) -> str:
        return f"{self.method} {self.path}"


def fetch(url: str) -> str:
    """Read a bounded provider document over HTTPS, refusing off-site redirects."""
    if not url.startswith(BASE + "/reference"):
        raise ValueError("Unexpected provider URL")
    request = urllib.request.Request(url, headers={"User-Agent": "langchain-finic-compatibility/1"})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        final = urllib.parse.urlparse(response.geturl())
        if final.scheme != "https" or final.netloc != "docs.finic.ai":
            raise ValueError("Provider redirected outside docs.finic.ai")
        data = response.read(1_000_001)
    if len(data) > 1_000_000:
        raise ValueError("Provider reference exceeds size limit")
    return data.decode("utf-8")


def parse_reference(text: str) -> dict[str, Operation]:
    """Parse only the endpoint table; reject unexpected or incomplete pages."""
    section = text.split("## Errors", 1)[0]
    if not section.startswith("# API reference") or "## Errors" not in text:
        raise ValueError("Provider reference has an unexpected structure")
    operations: dict[str, Operation] = {}
    for match in ROW.finditer(section):
        op = Operation(**match.groupdict())
        if op.key in operations:
            raise ValueError(f"Duplicate operation: {op.key}")
        operations[op.key] = op
    # Reject partially parsed tables, including new methods or paths.
    table_rows = sum(line.lstrip().startswith("| [`") for line in section.splitlines())
    if table_rows != len(operations):
        raise ValueError("Some provider endpoint rows could not be parsed")
    # Avoid reporting an incomplete reference as a broad removal.
    if len(operations) < 5 or "POST /v1/score" not in operations:
        raise ValueError("Provider endpoint table is incomplete or unrecognized")
    return operations


def page_digest(op: Operation, text: str) -> str:
    heading = f"## {op.method} https://api.finic.ai{op.path}"
    if heading not in text or not text.startswith("# "):
        raise ValueError(f"Endpoint page did not match {op.key}")
    return hashlib.sha256(text.replace("\r\n", "\n").strip().encode()).hexdigest()


def collect() -> dict[str, Operation]:
    operations = parse_reference(fetch(REFERENCE))
    return {
        key: Operation(op.name, op.method, op.path, op.link, page_digest(op, fetch(f"{BASE}{op.link}.md")))
        for key, op in operations.items()
    }


def validate_baseline(baseline: dict) -> None:
    """Reject malformed baseline data rather than silently masking gaps."""
    operations = baseline.get("operations")
    if not isinstance(operations, dict) or not operations:
        raise ValueError("Compatibility baseline has no operations")
    for key, entry in operations.items():
        if not isinstance(entry, dict) or entry.get("status") not in {"supported", "partial", "missing"}:
            raise ValueError(f"Invalid baseline status for {key}")
        if not re.fullmatch(r"[0-9a-f]{64}", entry.get("digest", "")):
            raise ValueError(f"Invalid baseline digest for {key}")
        if not re.fullmatch(r"/reference/[a-z-]+", entry.get("link", "")):
            raise ValueError(f"Invalid baseline link for {key}")
        if not isinstance(entry.get("note"), str) or not entry["note"].strip():
            raise ValueError(f"Missing baseline note for {key}")


def assess(operations: dict[str, Operation], baseline: dict) -> list[dict[str, str]]:
    """Create actionable findings; a reviewed status never comes from provider text."""
    findings = []
    reviewed = baseline["operations"]
    for key, op in sorted(operations.items()):
        known = reviewed.get(key)
        if known is None:
            reason = "new"
        elif known["status"] in {"partial", "missing"}:
            reason = "gap"
        elif known["digest"] != op.digest:
            reason = "changed"
        else:
            continue
        findings.append({"key": key, "reason": reason, "digest": op.digest, "link": op.link})
    for key, known in sorted(reviewed.items()):
        if key not in operations:
            findings.append(
                {"key": key, "reason": "removed", "digest": known["digest"], "link": known["link"]}
            )
    return findings


def marker(finding: dict[str, str]) -> str:
    # The stable marker deduplicates open issues; the evidence marker also
    # prevents a closed issue from being recreated for unchanged documentation.
    key = hashlib.sha256(finding["key"].encode()).hexdigest()[:16]
    return f"<!-- finic-compatibility:{key}:{finding['reason']} -->"


def evidence_marker(finding: dict[str, str]) -> str:
    return f"<!-- finic-evidence:{finding['digest']} -->"


def render_report(operations: dict[str, Operation], baseline: dict, findings: list[dict[str, str]]) -> str:
    lines = [
        "# Finic compatibility report",
        "",
        f"Provider operations checked: **{len(operations)}**.",
        f"Findings needing review: **{len(findings)}**.",
        "",
        "Endpoint and documentation changes are reviewed; this check does not prove behavior.",
        "",
    ]
    if not findings:
        lines.append("No unreviewed gaps or documentation changes detected.")
    for finding in findings:
        key, reason = finding["key"], finding["reason"]
        note = baseline["operations"].get(key, {}).get("note", "Review the new or changed provider contract.")
        lines.append(
            f"- **{reason}** — `{key}`: {note} "
            f"([provider reference]({BASE}{finding['link']}), evidence `{finding['digest'][:12]}`)"
        )
    return "\n".join(lines) + "\n"


def sync_issues(findings: list[dict[str, str]], baseline: dict) -> None:
    """Create one issue per open finding, retaining evidence when an issue is closed."""
    result = subprocess.run(  # noqa: S603
        ["gh", "issue", "list", "--state", "all", "--limit", "1000", "--json", "number,state,body"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    )
    issues = json.loads(result.stdout)
    for finding in findings:
        stable, evidence = marker(finding), evidence_marker(finding)
        matches = [issue for issue in issues if stable in (issue.get("body") or "")]
        if any(issue["state"].upper() == "OPEN" for issue in matches):
            continue
        if any(evidence in (issue.get("body") or "") for issue in matches):
            continue
        key = finding["key"]
        note = (
            baseline["operations"]
            .get(key, {})
            .get("note", "Review this provider operation and package coverage.")
        )
        reason = finding["reason"]
        body = (
            f"Finic's published reference indicates a compatibility finding for `{key}` ({reason}).\n\n"
            f"{note}\n\nProvider reference: {BASE}{finding['link']}\n\n"
            "- [ ] Compare provider request, response, and behavior with the package.\n"
            "- [ ] Implement or document the supported behavior and add appropriate tests.\n"
            "- [ ] Update the compatibility baseline after review.\n\n"
            f"{stable}\n{evidence}\n"
        )
        subprocess.run(  # noqa: S603
            [
                "gh",
                "issue",
                "create",
                "--label",
                "api",
                "--title",
                f"compatibility: review {key}",
                "--body",
                body,
            ],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        )
        issues.append({"state": "OPEN", "body": body})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--create-issues", action="store_true")
    args = parser.parse_args()
    try:
        baseline = json.loads(args.baseline.read_text())
        validate_baseline(baseline)
        operations = collect()
        findings = assess(operations, baseline)
        report = render_report(operations, baseline, findings)
        print(report)
        if summary := os.getenv("GITHUB_STEP_SUMMARY"):
            with open(summary, "a", encoding="utf-8") as file:
                file.write(report)
        if args.create_issues:
            sync_issues(findings, baseline)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        message = f"Finic compatibility check failed; coverage is unknown: {type(exc).__name__}: {exc}"
        print(message, file=sys.stderr)
        if summary := os.getenv("GITHUB_STEP_SUMMARY"):
            with open(summary, "a", encoding="utf-8") as file:
                file.write(
                    "# Finic compatibility check failed\n\nCoverage is unknown. See the workflow logs.\n"
                )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
