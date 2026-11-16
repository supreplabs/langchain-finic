"""Public recipes remain offline and deterministic until explicitly opted in."""

import csv
import hashlib
import io
import json
import runpy
import sys
import types
import zipfile
from pathlib import Path

import httpx
import pytest

MODULE = Path(__file__).resolve().parents[2] / "examples/public_dataset_scoring.py"


@pytest.fixture
def recipe():
    return runpy.run_path(str(MODULE))


def archive_fixture(dataset, path):
    fields = (
        ["contact", "month", "campaign", "pdays", "previous", "poutcome", "y", "duration"]
        if dataset == "bank"
        else [
            "Month",
            "SpecialDay",
            "OperatingSystems",
            "Browser",
            "Region",
            "TrafficType",
            "VisitorType",
            "Weekend",
            "Revenue",
            "PageValues",
        ]
    )
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter=";" if dataset == "bank" else ",")
    writer.writeheader()
    for i in range(1000 if dataset == "bank" else 6500):
        if dataset == "bank":
            writer.writerow(
                dict(
                    contact="telephone",
                    month="may",
                    campaign="1",
                    pdays=str(i),
                    previous="0",
                    poutcome="unknown",
                    y="yes" if i % 3 else "no",
                    duration="99",
                )
            )
        else:
            writer.writerow(
                dict(
                    Month="May",
                    SpecialDay="0.0",
                    OperatingSystems="2",
                    Browser="1",
                    Region="1",
                    TrafficType="2",
                    VisitorType="New_Visitor",
                    Weekend="FALSE",
                    Revenue="TRUE" if i % 3 else "FALSE",
                    PageValues="99",
                )
            )
    with zipfile.ZipFile(path, "w") as outer:
        if dataset == "bank":
            inner = io.BytesIO()
            with zipfile.ZipFile(inner, "w") as nested:
                nested.writestr("bank-full.csv", stream.getvalue())
            outer.writestr("bank.zip", inner.getvalue())
        else:
            outer.writestr("online_shoppers_intention.csv", stream.getvalue())


@pytest.mark.parametrize("dataset", ["bank", "shopper"])
def test_preparation_is_reproducible_and_excludes_targets(recipe, tmp_path, monkeypatch, dataset):
    archive = tmp_path / "source.zip"
    archive_fixture(dataset, archive)
    monkeypatch.setitem(
        recipe["SOURCES"][dataset], "sha256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    first, report = recipe["prepare"](dataset, archive)
    assert recipe["prepare"](dataset, archive) == (first, report)
    ids = [set(part) for part in report["split_identities"].values()]
    assert all(not left & right for i, left in enumerate(ids) for right in ids[i + 1 :])
    for records in first.values():
        for row in records:
            assert set(row["features"]) == set(recipe["SOURCES"][dataset]["features"])
            assert not {"duration", "PageValues", "Revenue", "y", "is_bad"} & row["features"].keys()
    assert report["remote_calls"] == 0
    assert report["license"] == "CC BY 4.0"


def test_checksum_is_checked_before_archive_parsing(recipe, tmp_path):
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"not an archive")
    with pytest.raises(ValueError, match="checksum"):
        recipe["prepare"]("bank", archive)


@pytest.mark.parametrize(
    "extra",
    [[], ["--allow-remote-writes"], ["--allow-remote-writes", "--threshold", "500", "--holdout-limit", "21"]],
)
def test_live_requires_opt_in_and_bounded_inputs(recipe, monkeypatch, tmp_path, extra):
    monkeypatch.setenv("FINIC_API_KEY", "test")
    with pytest.raises(SystemExit):
        recipe["main"](
            [
                "bank",
                "--archive",
                str(tmp_path / "absent.zip"),
                "--output",
                str(tmp_path / "out"),
                "--live",
                *extra,
            ]
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.asyncio
async def test_live_budget_stops_requests_and_records_failure(recipe, monkeypatch, tmp_path):
    monkeypatch.setenv("FINIC_API_KEY", "test")
    calls = []
    original = httpx.AsyncClient

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={"job_id": "job", "complete": False})

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs)
    )

    async def fake_trial(client, **kwargs):
        for _ in range(4):
            await client.aget_job_status("job")

    monkeypatch.setitem(sys.modules, "disposable_scoring", types.SimpleNamespace(run_trial=fake_trial))
    splits = {name: [] for name in ("train", "calibration", "holdout")}
    with pytest.raises(RuntimeError, match="budget exhausted"):
        await recipe["run_live"]("bank", splits, tmp_path, holdout_limit=1, threshold=500, request_budget=2)
    assert len(calls) == 2
    report = json.loads((tmp_path / "execution.json").read_text())
    assert report["request_attempts"] == 2 and report["status"] == "failed"
    assert "test" not in json.dumps(report)
