"""Reproduce UCI trial splits offline; remote lifecycle requires explicit CLI opt-in."""

import argparse
import asyncio
import csv
import hashlib
import io
import json
import os
import random
import zipfile
from pathlib import Path

from langchain_finic import CalibrationRequest, CreateScoringModelRequest, FinicClient, ScoringRequest

SOURCES = {
    "bank": {
        "sha256": "e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4",
        "url": "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
        "doi": "https://doi.org/10.24432/C5K306",
        "citation": "Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing. UCI.",
        "features": ["contact", "month", "campaign", "pdays", "previous", "poutcome"],
        "criteria": (
            "Historical campaign non-subscription. Bad means no term-deposit subscription; not fraud."
        ),
    },
    "shopper": {
        "sha256": "2972e6184d3ad7beaaa831d9fc2b059dc3ee29df69d1ec593c466a5cd8485d14",
        "url": "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip",
        "doi": "https://doi.org/10.24432/C5F88Q",
        "citation": "Sakar, C., & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. UCI.",
        "features": [
            "Month",
            "SpecialDay",
            "OperatingSystems",
            "Browser",
            "Region",
            "TrafficType",
            "VisitorType",
            "Weekend",
        ],
        "criteria": (
            "Historical early-session metadata. Bad means no purchase; high scores mean nonpurchase risk."
        ),
    },
}


def read_archive(dataset: str, archive: Path) -> list[dict]:
    """Reject changed archives before parsing; read members without extracting files."""
    raw = archive.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCES[dataset]["sha256"]:
        raise ValueError("Archive checksum mismatch; review the source before changing the pinned digest")
    with zipfile.ZipFile(io.BytesIO(raw)) as outer:
        if dataset == "bank":
            with zipfile.ZipFile(io.BytesIO(outer.read("bank.zip"))) as inner:
                text = inner.read("bank-full.csv").decode("utf-8")
            return list(csv.DictReader(io.StringIO(text), delimiter=";"))
        text = outer.read("online_shoppers_intention.csv").decode("utf-8")
        return list(csv.DictReader(io.StringIO(text)))


def split_rows(dataset: str, rows: list[dict]) -> dict[str, list[dict]]:
    """Match the original trial seeds and retain zero-based source row identities."""
    parsed = []
    seen = set()
    for index, row in enumerate(rows):
        features = {key: row[key] for key in SOURCES[dataset]["features"]}
        if dataset == "bank":
            identity = tuple(sorted((key, value) for key, value in row.items() if key != "duration"))
            if identity in seen:
                continue
            seen.add(identity)
            if row["y"] not in {"yes", "no"}:
                raise ValueError("Unexpected bank label")
            for key in ("campaign", "pdays", "previous"):
                features[key] = int(features[key])
            bad = row["y"] == "no"
        else:
            if row["Revenue"].upper() not in {"TRUE", "FALSE"}:
                raise ValueError("Unexpected shopper label")
            if row["Weekend"].upper() not in {"TRUE", "FALSE"}:
                raise ValueError("Unexpected Weekend value")
            features["Weekend"] = row["Weekend"].upper() == "TRUE"
            features["SpecialDay"] = float(row["SpecialDay"])
            bad = row["Revenue"].upper() == "FALSE"
        parsed.append({"source_row": index, "features": features, "is_bad": bad})
    if dataset == "bank":
        a, b = int(len(parsed) * 0.5), int(len(parsed) * 0.8)
        rng = random.Random(16)
        pools = {"train": parsed[:a], "calibration": parsed[a:b], "holdout": parsed[b:]}
        return {
            name: rng.sample(pool, min(2000, len(pool))) if name != "holdout" else pool
            for name, pool in pools.items()
        }
    random.Random(468).shuffle(parsed)
    return {"holdout": parsed[:2466], "train": parsed[2466:4466], "calibration": parsed[4466:6466]}


def labeled(records: list[dict]) -> list[dict]:
    return [{**row["features"], "is_bad": row["is_bad"]} for row in records]


def prepare(dataset: str, archive: Path) -> tuple[dict, dict]:
    rows = read_archive(dataset, archive)
    splits = split_rows(dataset, rows)
    criteria = SOURCES[dataset]["criteria"]
    CreateScoringModelRequest(
        name="Offline public dataset trial", criteria=criteria, data=labeled(splits["train"])
    )
    CalibrationRequest(model_id="pending", data=labeled(splits["calibration"]))
    if not splits["holdout"]:
        raise ValueError("Empty holdout")
    for row in splits["holdout"]:
        ScoringRequest(model_id="pending", score_id=str(row["source_row"]), data=row["features"])
    identities = {name: [row["source_row"] for row in records] for name, records in splits.items()}
    all_ids = [identity for ids in identities.values() for identity in ids]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Overlapping source rows")
    report = {
        "dataset": dataset,
        "source": SOURCES[dataset],
        "license": "CC BY 4.0",
        "source_rows": len(rows),
        "split_identities": identities,
        "splits": {
            name: {
                "rows": len(records),
                "bad": sum(row["is_bad"] for row in records),
                "good": sum(not row["is_bad"] for row in records),
            }
            for name, records in splits.items()
        },
        "payload_validation": "passed",
        "remote_calls": 0,
    }
    return splits, report


async def run_live(dataset, splits, output, *, holdout_limit, threshold, request_budget):
    """Bound every HTTP request, including polls and cleanup; never auto-retry a write."""
    import httpx
    from disposable_scoring import run_trial

    count = 0
    report = {"status": "running", "request_budget": request_budget, "request_attempts": 0}

    async def count_request(request):
        nonlocal count
        if count >= request_budget:
            raise RuntimeError("Request budget exhausted; reconcile the ledger before cleanup")
        count += 1

    try:
        async with httpx.AsyncClient(event_hooks={"request": [count_request]}) as http:
            async with FinicClient(max_attempts=1, http_async_client=http) as client:
                ledger = await run_trial(
                    client,
                    training=labeled(splits["train"]),
                    calibration=labeled(splits["calibration"]),
                    holdout=labeled(splits["holdout"][:holdout_limit]),
                    criteria=SOURCES[dataset]["criteria"],
                    threshold=threshold,
                    ledger_path=output / "lifecycle-ledger.json",
                )
        report.update(status="passed", cleanup=ledger["cleanup"])
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        report["request_attempts"] = count
        (output / "execution.json").write_text(json.dumps(report, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=SOURCES)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; existing paths are refused"
    )
    parser.add_argument(
        "--live", action="store_true", help="Create, score, recalibrate and permanently delete"
    )
    parser.add_argument("--allow-remote-writes", action="store_true")
    parser.add_argument("--holdout-limit", type=int, default=20)
    parser.add_argument("--threshold", type=int)
    parser.add_argument("--request-budget", type=int, default=120)
    args = parser.parse_args(argv)
    if args.live:
        if not args.allow_remote_writes or not os.environ.get("FINIC_API_KEY"):
            parser.error("Live mode requires --allow-remote-writes and FINIC_API_KEY")
        if args.threshold is None or not 0 <= args.threshold <= 1000:
            parser.error("Live mode requires an explicit --threshold from 0 to 1000")
        if not 1 <= args.holdout_limit <= 20 or not 10 <= args.request_budget <= 200:
            parser.error("Live mode permits 1–20 holdout rows and a 10–200 request budget")
    splits, report = prepare(args.dataset, args.archive)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, records in splits.items():
        (args.output / f"{name}.json").write_text(json.dumps(labeled(records)) + "\n")
    (args.output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.live:
        asyncio.run(
            run_live(
                args.dataset,
                splits,
                args.output,
                holdout_limit=args.holdout_limit,
                threshold=args.threshold,
                request_budget=args.request_budget,
            )
        )
    print(json.dumps({"dataset": args.dataset, "splits": report["splits"], "live": args.live}))


if __name__ == "__main__":
    main()
