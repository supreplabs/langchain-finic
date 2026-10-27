import asyncio
import json
import runpy
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
reporter = runpy.run_path(str(ROOT / "scripts/live_evidence.py"))
Counter = runpy.run_path(str(ROOT / "scripts/live_requests.py"))["RequestCounter"]


def inputs(tmp_path, passed=15, skipped=0, failed=0, requests=18):
    cases = "<testcase/>" * passed + "<testcase><skipped/></testcase>" * skipped
    cases += (
        "<testcase><failure>private-model-response</failure><system-out>secret</system-out></testcase>"
        * failed
    )
    (tmp_path / "results.xml").write_text("<testsuites><testsuite>" + cases + "</testsuite></testsuites>")
    (tmp_path / "requests.json").write_text(json.dumps({"requests": requests}))


@pytest.mark.parametrize(
    "event,ref,eligible",
    [
        ("schedule", "refs/heads/main", True),
        ("schedule", "refs/heads/feature", False),
        ("workflow_dispatch", "refs/heads/main", False),
        ("push", "refs/tags/v0.1.0a6", False),
    ],
)
def test_only_successful_scheduled_main_runs_qualify(tmp_path, event, ref, eligible):
    inputs(tmp_path)
    result = reporter["report"](
        tmp_path,
        "success",
        {"GITHUB_EVENT_NAME": event, "GITHUB_REF": ref, "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "2"},
    )
    assert result["successful_daily_observation"] is eligible
    assert result["status"] == "passed"
    assert result["tests"]["executed"] == 15
    assert result["run_url"].endswith("/123/attempts/2")


@pytest.mark.parametrize(
    "passed,skipped,failed,requests,outcome",
    [
        (0, 15, 0, 0, "success"),
        (14, 0, 1, 18, "failure"),
        (15, 0, 0, 18, "failure"),
        (15, 0, 0, 17, "success"),
        (15, 0, 0, 19, "success"),
        (0, 0, 0, 0, "success"),
        (15, 0, 0, True, "success"),
    ],
)
def test_incomplete_or_failed_evidence_never_passes(tmp_path, passed, skipped, failed, requests, outcome):
    inputs(tmp_path, passed, skipped, failed, requests)
    result = reporter["report"](
        tmp_path, outcome, {"GITHUB_EVENT_NAME": "schedule", "GITHUB_REF": "refs/heads/main"}
    )
    assert result["status"] == "failed"
    assert not result["successful_daily_observation"]
    assert "private-model-response" not in json.dumps(result)
    assert "secret" not in reporter["markdown"](result)


@pytest.mark.parametrize(
    "outcome,status", [("skipped", "skipped"), ("failure", "failed"), ("success", "failed")]
)
def test_missing_reports_do_not_fabricate_success(tmp_path, outcome, status):
    result = reporter["report"](tmp_path, outcome, {})
    assert result["status"] == status and result["requests"] is None
    assert not result["successful_daily_observation"]
    (tmp_path / "results.xml").write_text("incomplete xml")
    (tmp_path / "requests.json").write_text("incomplete json")
    assert reporter["report"](tmp_path, "success", {})["status"] == "failed"


def test_counter_persists_failed_and_concurrent_attempts_without_payloads(tmp_path):
    path = tmp_path / "requests.json"

    def fail(request):
        assert json.loads(path.read_text())["requests"] >= 1
        raise httpx.ReadTimeout("private", request=request)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
            return await asyncio.gather(
                *(client.get("https://api.finic.ai/v1/decide") for _ in range(3)), return_exceptions=True
            )

    original_send = httpx.Client.send
    with Counter(path):
        with httpx.Client(transport=httpx.MockTransport(fail)) as client:
            with pytest.raises(httpx.ReadTimeout):
                client.get("https://api.finic.ai/v1/decide", headers={"authorization": "private"})
        results = asyncio.run(run())
        assert all(isinstance(result, httpx.ReadTimeout) for result in results)
        with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as client:
            client.get("https://example.test/")
    assert httpx.Client.send is original_send
    assert json.loads(path.read_text()) == {"requests": 4}


def test_workflow_preserves_reporting_on_failure_and_missing_credentials():
    workflow = (ROOT / ".github/workflows/integration.yml").read_text()
    assert "if: always()" in workflow and "issues: write" in workflow
    assert "artifacts/live/evidence.*" in workflow
    assert "path: artifacts/live/results.xml" not in workflow
    assert "github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'" in workflow
    assert "FINIC_LIFECYCLE_TESTS" not in workflow
