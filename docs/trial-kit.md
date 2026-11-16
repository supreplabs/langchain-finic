# langchain-finic 0.1.0b3 trial kit

Purpose: reproduce developer onboarding and migration tasks, record actual results,
and distinguish observed execution from static review. Simulated agent sessions
must label their reports synthetic; no attribution to named people.

Record the exact source commit and working-tree changes under review.
When testing an unpublished fix, distinguish it from the installed PyPI release.

Task A: Follow README and docs/usage.md to configure a typed decision and construct
an agent. Record prerequisites, confusing steps, and which commands were executed.
Task B: Follow docs/scoring.md and examples/lifecycle_graph.py to discover models,
request question evidence, migrate calibration records, and construct a checkpointed
workflow. Check score tool schema privileges and legacy payload diagnostics.

For this local simulation: inspect source and docs only. Do not read .env, access
credentials, install packages, call APIs, modify source, or publish comments.
Report exact file/line references and severity. Confirm which earlier findings
are resolved and which remain. Never claim installation or live API success.

For a later human-run trial, create a fresh venv with Python >=3.11, install the
exact PyPI version, configure FINIC_API_KEY outside recorded logs, and run fictional
cases only. Lifecycle writes need explicit approval and leave remote state.

Feedback template:
- Reviewer type: human / simulated agent
- Version, commit or working-tree state, Python and dependency versions
- Task and expected result
- Steps actually executed and result (pass / fail / not executed)
- Friction, file references, reproducible errors, suggested fix
- Limits: credentials, network, real host model, and API calls used or absent
- Human reviewer confirmation, if any; never invent this field

A release candidate can use these findings as maintainer quality evidence.
Independent-user acceptance requires separately recorded actual user feedback.
