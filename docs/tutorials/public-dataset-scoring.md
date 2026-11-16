---
description: Reproduce UCI Bank Marketing and Online Shoppers scoring trials with pinned sources, offline validation, and explicit bounded live execution.
---

# Public dataset scoring recipes

Use [public_dataset_scoring.py](https://github.com/supreplabs/langchain-finic/blob/main/examples/public_dataset_scoring.py)
with [disposable_scoring.py](https://github.com/supreplabs/langchain-finic/blob/main/examples/disposable_scoring.py)
in the same directory. The examples use the installed package and Python's standard
library; no pandas or dataset SDK is required. Preparation does not read credentials
or call Finic. Generated data and reports belong outside version control.

## Sources and attribution

Both datasets are published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Retain these citations and note the feature selection, label inversion and splitting
when sharing derived data. The package's MIT license does not replace the dataset license.

- Moro, S., Rita, P., & Cortez, P. (2014). **Bank Marketing**. UCI Machine Learning
  Repository. [DOI: 10.24432/C5K306](https://doi.org/10.24432/C5K306).
- Sakar, C., & Kastro, Y. (2018). **Online Shoppers Purchasing Intention Dataset**.
  UCI Machine Learning Repository. [DOI: 10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q).

From a source checkout with development dependencies installed:

```bash
mkdir -p artifacts/public-datasets
curl --fail --location 'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip' -o artifacts/public-datasets/bank.zip
curl --fail --location 'https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip' -o artifacts/public-datasets/shopper.zip
uv run python examples/public_dataset_scoring.py bank --archive artifacts/public-datasets/bank.zip --output artifacts/public-datasets/bank-prepared
uv run python examples/public_dataset_scoring.py shopper --archive artifacts/public-datasets/shopper.zip --output artifacts/public-datasets/shopper-prepared
```

Each command verifies SHA-256 before opening the archive. A mismatch stops preparation;
review the source change rather than replacing the checksum to silence the error.

| Archive | SHA-256 |
| --- | --- |
| Bank | `e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4` |
| Shopper | `2972e6184d3ad7beaaa831d9fc2b059dc3ee29df69d1ec593c466a5cd8485d14` |

## Labels, timing and reproducible splits

For bank records, `is_bad = (y == "no")`; this is non-subscription, not fraud or
creditworthiness. Inputs are `contact`, `month`, `campaign`, `pdays`, `previous`,
and `poutcome`. Duration, target and demographic/financial fields are excluded.
Campaign count includes the current contact: interpret this recipe at the planned
contact, with that count known. These historical records do not prove a production
feature-availability contract or suitability for Indian customers.

The `bank-full.csv` source has 45,211 rows in date order. Deduplication across all
fields except duration (including outcome) retains 45,195 rows; it is exact-record
filtering, not entity deduplication. The first 50%, next 30%, and final 20% form
chronological pools. Seed 16 samples 2,000 training then 2,000 calibration rows;
the full final pool is held out. The source has no stable customer identifier, so
customer separation cannot be established. Expect chronological distribution shift.

For shoppers, `is_bad = (Revenue == FALSE)` means nonpurchase. Inputs are `Month`,
`SpecialDay`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`,
and `Weekend`. PageValues, bounce/exit rates, and final session counts/durations
are excluded. Check availability of these metadata in your application before use.
Seed 468 shuffles all 12,330 source rows. The first 2,466 are holdout, the next 2,000
training and the next 2,000 calibration; remaining rows are unused. This reproduces
the earlier trial, not a chronological generalization test. UCI describes distinct
users per session; the recipe itself only verifies disjoint source row identities.

| Dataset | Split | Rows | `is_bad=True` | `is_bad=False` |
| --- | --- | ---: | ---: | ---: |
| Bank | Train | 2,000 | 1,908 | 92 |
| Bank | Calibration | 2,000 | 1,818 | 182 |
| Bank | Holdout | 9,039 | 6,184 | 2,855 |
| Shopper | Train | 2,000 | 1,684 | 316 |
| Shopper | Calibration | 2,000 | 1,718 | 282 |
| Shopper | Holdout | 2,466 | 2,079 | 387 |

`preparation.json` records source attribution, digest, class counts and zero-based
CSV data-row identities (header excluded). Separate train/calibration/holdout JSON
files contain labeled features; identities are not sent as model features. All
payloads are validated locally. Output directories must be new to avoid overwrites.
Do not tune thresholds using the final holdout; choose them separately in advance.

## Explicit bounded live trial

Read the [disposable scoring procedure](disposable-scoring.md) first. An authorized
run creates one remote model, scores at most 20 predetermined holdout records twice,
recalibrates, and permanently deletes that model. Use an admin-created key in
`FINIC_API_KEY`. The command does not load `.env`; export credentials privately.
These are direct client calls, with no LangSmith callbacks or host LLM.

After choosing a threshold using a separate development set or business policy:

```bash
uv run python examples/public_dataset_scoring.py bank \
  --archive artifacts/public-datasets/bank.zip \
  --output artifacts/public-datasets/bank-live \
  --live --allow-remote-writes --threshold "$TRIAL_THRESHOLD" \
  --holdout-limit 20 --request-budget 120
```

Live mode requires both flags, a key and an explicit threshold. The hard request
budget (maximum 200) includes polls, errors and cleanup verification. Each request
gets one attempt; each training/calibration wait has a 900-second deadline. Budget
exhaustion or an ambiguous write can leave a model requiring manual reconciliation:
inspect `lifecycle-ledger.json` and reserve capacity for cleanup. Do not rerun with a
new directory to bypass an unresolved ledger. A successful run verifies deletion.
`execution.json` records status, request count and exception type without raw errors.
The first N holdout rows are chosen without balancing; these small trials check
integration behavior, not model quality or independent-user acceptance.

## Sanitized report template

Record these fields in an issue comment; omit credentials, raw records and response text:

- Version, source commit, Python version and dependency versions:
- Dataset citation, archive digest, split counts and preparation validation:
- Execution mode: offline / authorized live; actual calls and request budget:
- Holdout selection rule, size, class counts, preselected threshold and rationale:
- Initial trained / recalibrated confusion matrix, Brier error and training-prior baseline:
- Question counts; limitations and distribution shift:
- Trial status, cleanup status, any unresolved operation and reconciliation outcome:
- Steps actually executed; reviewer type and confirmation, if applicable:

Local ledgers contain operational IDs; inspect and redact them before sharing.
