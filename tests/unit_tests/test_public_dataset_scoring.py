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


@pytest.mark.parametrize("threshold", ["-1", "1001", "not-an-integer"])
def test_invalid_threshold_is_rejected_before_preparation(recipe, monkeypatch, tmp_path, threshold):
    monkeypatch.setenv("FINIC_API_KEY", "test")
    with pytest.raises(SystemExit) as error:
        recipe["main"](
            [
                "bank",
                "--archive",
                str(tmp_path / "absent.zip"),
                "--output",
                str(tmp_path / "out"),
                "--live",
                "--allow-remote-writes",
                "--threshold",
                threshold,
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()


def test_live_missing_key_is_rejected_before_preparation(recipe, monkeypatch, tmp_path):
    monkeypatch.delenv("FINIC_API_KEY", raising=False)
    with pytest.raises(SystemExit) as error:
        recipe["main"](
            [
                "bank",
                "--archive",
                str(tmp_path / "absent.zip"),
                "--output",
                str(tmp_path / "out"),
                "--live",
                "--allow-remote-writes",
                "--threshold",
                "500",
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("live", [False, True])
def test_existing_output_is_preserved_without_remote_execution(recipe, monkeypatch, tmp_path, live):
    monkeypatch.setenv("FINIC_API_KEY", "test")
    output = tmp_path / "out"
    output.mkdir()
    sentinel = output / "execution.json"
    sentinel.write_text("existing evidence")
    globals_ = recipe["main"].__globals__
    monkeypatch.setitem(globals_, "prepare", lambda *_: ({}, {}))

    def unexpected(*args, **kwargs):
        pytest.fail("Existing output must block remote execution")

    monkeypatch.setitem(globals_, "run_live", unexpected)
    args = ["bank", "--archive", str(tmp_path / "unused.zip"), "--output", str(output)]
    if live:
        args += ["--live", "--allow-remote-writes", "--threshold", "500"]
    with pytest.raises(FileExistsError):
        recipe["main"](args)
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text() == "existing evidence"


@pytest.mark.parametrize("dataset", ["bank", "shopper"])
def test_label_polarity_matches_source_outcomes(recipe, tmp_path, monkeypatch, dataset):
    archive = tmp_path / "source.zip"
    archive_fixture(dataset, archive)
    monkeypatch.setitem(
        recipe["SOURCES"][dataset], "sha256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    source = recipe["read_archive"](dataset, archive)
    splits = recipe["split_rows"](dataset, source)
    observed = set()
    for records in splits.values():
        for row in records:
            original = source[row["source_row"]]
            expected = original["y"] == "no" if dataset == "bank" else original["Revenue"] == "FALSE"
            assert row["is_bad"] is expected
            observed.add(expected)
    assert observed == {False, True}


def test_bank_duration_duplicates_removed_before_chronological_split(recipe):
    source = [
        dict(
            contact="telephone",
            month="may",
            campaign="1",
            pdays=str(i),
            previous="0",
            poutcome="unknown",
            y="no" if i % 2 else "yes",
            duration="10",
        )
        for i in range(10)
    ]
    source.insert(1, dict(source[0], duration="999"))
    splits = recipe["split_rows"]("bank", source)
    ids = {name: {row["source_row"] for row in records} for name, records in splits.items()}
    assert ids == {"train": {0, 2, 3, 4, 5}, "calibration": {6, 7, 8}, "holdout": {9, 10}}
    assert max(ids["train"]) < min(ids["calibration"]) < min(ids["holdout"])


@pytest.mark.parametrize("dataset", ["bank", "shopper"])
def test_csv_missing_required_columns_is_rejected(recipe, tmp_path, monkeypatch, dataset):
    archive = tmp_path / "malformed.zip"
    with zipfile.ZipFile(archive, "w") as outer:
        if dataset == "bank":
            inner = io.BytesIO()
            with zipfile.ZipFile(inner, "w") as nested:
                nested.writestr("bank-full.csv", "y;duration\nyes;20\n")
            outer.writestr("bank.zip", inner.getvalue())
        else:
            outer.writestr("online_shoppers_intention.csv", "Revenue,PageValues\nTRUE,10\n")
    # Exercise parsing after checksum verification, without weakening production pins.
    monkeypatch.setitem(
        recipe["SOURCES"][dataset], "sha256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    output = tmp_path / "out"
    with pytest.raises(KeyError):
        recipe["main"]([dataset, "--archive", str(archive), "--output", str(output)])
    assert not output.exists()
