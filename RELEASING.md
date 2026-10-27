# Releasing

Release target: **0.1.0a6**, repository **supreplabs/langchain-finic**.
The package remains alpha. A release is complete only when PyPI installation is verified.

## One-time account setup

Create separate accounts on [PyPI](https://pypi.org/account/register/) and
[TestPyPI](https://test.pypi.org/account/register/). Verify each email address and
configure two-factor authentication (security key/passkey or authenticator), saving
recovery codes privately. Do not share passwords, recovery codes, or API keys here.

Register a **pending publisher** on each site's account publishing page:
[PyPI](https://pypi.org/manage/account/publishing/) and
[TestPyPI](https://test.pypi.org/manage/account/publishing/).
Choose GitHub Actions and enter these exact values:

| Field | TestPyPI | PyPI |
| --- | --- | --- |
| PyPI project name | `langchain-finic` | `langchain-finic` |
| GitHub owner | `supreplabs` | `supreplabs` |
| Repository name | `langchain-finic` | `langchain-finic` |
| Workflow filename | `release.yml` | `release.yml` |
| Environment name | `testpypi` | `pypi` |

Pending publishers create the project on first successful upload; they do not reserve
its name. If a project already exists, verify ownership and configure its publisher
through the project settings. Never rename or upload to another project automatically.
No PyPI API token or `PYPI_TOKEN` GitHub secret is needed.

Reference: [PyPI pending publishers](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## Repository configuration

GitHub environments `testpypi` and `pypi` permit deployments only from tags matching
`v*`. Enable GitHub Pages with GitHub Actions as the source, private vulnerability
reporting, and required CI checks on the default branch. CI and documentation builds
need no API credentials. Publishing jobs alone receive OIDC permission.

## Validate and release

1. Update `pyproject.toml` and add a nonempty `## VERSION` section to `CHANGELOG.md`
   (for example `## 0.1.0a6`). Run `uv lock` and commit the release.
2. Run `make install` and `make check`. Confirm all compatibility CI jobs pass.
3. Inspect wheel/sdist contents, generated documentation, and `scripts/check_dist.py`
   hash output. No benchmark datasets, research code, credentials, or case artifacts
   belong in this repository or its distributions.
4. Confirm both pending publishers are configured and the package name is still
   available to this maintainer. Create and push the matching version tag:

```bash
git tag -a v0.1.0a6 -m 'Release 0.1.0a6'
git push origin v0.1.0a6
```

The release workflow reruns CI, checks the tag against metadata, builds the release
wheel and sdist once, and uploads immutable workflow artifacts with SHA-256 hashes.
It publishes to TestPyPI, retrieves the exact files and checks their hashes, then
installs the wheel in a fresh environment. Dependencies are resolved from PyPI only;
TestPyPI is not used as an extra dependency index.

The same artifacts are then published to PyPI with attestations, downloaded and
hash-verified, and installed by exact public version. Public installation refreshes this package's index
metadata and retries up to five times (120 seconds per attempt, with 5/10/20/40-second
delays) to tolerate short PyPI index propagation delays. Exhausted retries fail the
workflow; rerun failed jobs using the original artifacts.

A GitHub release is created
only after installation succeeds. Alpha, beta, release-candidate, and development
versions are marked as prereleases; final and post releases are not. All action references are pinned to commit SHAs.

If publishing succeeds but a later job fails, rerun failed jobs in the **same run**
to preserve the exact build artifacts. Existing files are skipped during upload,
but registry hash verification still requires an exact match. Do not rebuild and
replace an already published version; use a new version for changed artifacts.
`workflow_dispatch` must be dispatched on the version tag, not `main`.

```bash
pip install langchain-finic==0.1.0a6
pip install 'langchain-finic[middleware]==0.1.0a6'
```

## Live smoke checks

Default tests and installation checks use mock HTTP and spend no credits. To make
eighteen explicit live requests with fictional data (fifteen checks, including
LangChain's standard tool integration suite, all question types, batches, LangGraph,
middleware, and documented data/context inputs):

```bash
export FINIC_API_KEY=your_key
make integration_test
```

Do not commit the key or record production inputs. Tests verify the response contract,
not an exact probability. No live calls run in pull requests. Future release
workflows run the same live suite before building and publishing when the repository
secret `FINIC_API_KEY` is configured. A missing secret skips the live job without
blocking the release; a failed live test blocks publishing. The separate live
workflow also runs daily and supports manual runs on `main`.

## After publication

Use [listing metadata](LANGCHAIN_LISTING.md) for a LangChain integration listing request; do not
claim LangChain or Finic endorsement. Obtain maintainer authorization before submitting
an issue on their behalf.

If a release is broken, publish a corrected version and consider yanking the bad
version with a clear reason. Do not delete published files or move release tags.


## Release presentation

`scripts/release_metadata.py` selects only the changelog section matching the
package version. It generates the title, prerelease flag, installation instructions,
compatibility information, and checksum instructions before any registry upload.
Missing, empty, or duplicate version sections stop the release. Release notes and
metadata travel alongside the validated distribution artifacts through the workflow.

Preview a future release's presentation locally:

```bash
uv run --locked --group build python scripts/release_metadata.py --output-dir release-info
```

Review `release-info/notes.md` and `release-info/metadata.json`. Updating an existing
GitHub release title or description does not change its tag, distribution files, or
PyPI description. Do not rerun an old release with new source to update presentation.

## Promotion gates

The package remains alpha until maintainers review the readiness evidence and
explicitly authorize a promotion. Merging a readiness PR does not authorize a
release. The progression is `0.1.0aN` → `0.1.0bN` → optional `0.1.0rcN` → `0.1.0`.
These are project acceptance criteria, not a certification from LangChain or Finic.

| Gate | Evidence required | Tracking |
| --- | --- | --- |
| Live contracts | All question types, bounded batches, LangGraph tools, and middleware pass against Finic; request budget documented | [#7](https://github.com/supreplabs/langchain-finic/issues/7) |
| Failure handling | Offline coverage for timeouts, retry exhaustion, malformed answers, cancellation, partial batches, and ownership; no unresolved blocking defect | [#8](https://github.com/supreplabs/langchain-finic/issues/8) |
| Public contract | Supported API, dependency policy, deprecation, and security support reviewed | [#9](https://github.com/supreplabs/langchain-finic/issues/9) |
| Publishing | Bounded fresh-index installation retries; exact-version installation and artifact hashes verified | [#10](https://github.com/supreplabs/langchain-finic/issues/10) |
| Beta observation | Seven daily runs of the expanded suite and two or three representative application validations, with failures investigated | [#11](https://github.com/supreplabs/langchain-finic/issues/11) |
| RC/final acceptance | Independent beta feedback, upgrade checks, resolved blockers, and installed release-candidate validation | [#12](https://github.com/supreplabs/langchain-finic/issues/12) |

The seven-day observation window starts after the expanded suite is merged to
`main`. A skipped live run is not evidence of a pass. Preserve the initial failed
attempts when recording reruns, distinguish provider/environment failures from
package defects, and link the investigation. This observation target can be revised
explicitly by maintainers; it must not be silently marked complete.

Record each application or daily-run result in its tracking issue with:

- Date, source commit, installed package/dependency versions, and workflow/run link.
- Synthetic scenario, tests actually executed, outcome, and request count.
- Failures, investigation, related fix, and retest links, if any.
- Evidence source: maintainer-run test, scripted host model, or independent user.

Do not include credentials, customer profiles, or raw sensitive responses.
Maintainer smoke tests do not count as independent-user feedback. No outreach is
required or authorized merely by creating a readiness issue.

Before final release, test representative consumers upgrading from the last alpha
and beta versions to the release candidate. Include imports, request schemas,
serialization, sync/async composition, and middleware where used. Record migrations
and dependency changes. A release candidate needs the same distribution, CI, live,
and documentation checks as a final release; any skipped check must be explained.

### Promotion checklist

- [ ] Review the linked gates, evidence, known issues, and explicit go/no-go decision.
- [ ] Update `pyproject.toml` version and run `uv lock`.
- [ ] Set the development-status classifier appropriately (`4 - Beta` for beta/RC;
      `5 - Production/Stable` for final only when the stated acceptance gates are met).
- [ ] Add a version-specific changelog section, including migrations and known limits.
- [ ] Update README/docs installation pins and maturity wording, the release target,
      and the security/support policy; preserve historical changelog entries.
- [ ] Run `make check`, review compatibility CI, and review actual live-test results.
- [ ] Validate release notes, archive contents, and the exact source commit to publish.
- [ ] Obtain explicit maintainer authorization, then create and push the new matching tag.
- [ ] Verify TestPyPI/PyPI hashes, public installation, GitHub assets, and documentation.

Keep readiness issues open until their evidence exists. The workflow's permitted
missing-key skip does not by itself satisfy the live-test promotion gate. Never move
an existing tag or replace a published artifact to promote a release.
