# Releasing

Release target: **0.1.0a4**, repository **supreplabs/langchain-finic**.
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

1. Update `pyproject.toml` version and changelog, run `uv lock`, and commit the release.
2. Run `make install` and `make check`. Confirm all compatibility CI jobs pass.
3. Inspect wheel/sdist contents, generated documentation, and `scripts/check_dist.py`
   hash output. No benchmark datasets, research code, credentials, or case artifacts
   belong in this repository or its distributions.
4. Confirm both pending publishers are configured and the package name is still
   available to this maintainer. Create and push the matching version tag:

```bash
git tag -a v0.1.0a4 -m 'Release 0.1.0a4'
git push origin v0.1.0a4
```

The release workflow reruns CI, checks the tag against metadata, builds the release
wheel and sdist once, and uploads immutable workflow artifacts with SHA-256 hashes.
It publishes to TestPyPI, retrieves the exact files and checks their hashes, then
installs the wheel in a fresh environment. Dependencies are resolved from PyPI only;
TestPyPI is not used as an extra dependency index.

The same artifacts are then published to PyPI with attestations, downloaded and
hash-verified, and installed by exact public version. A GitHub prerelease is created
only after installation succeeds. All action references are pinned to commit SHAs.

If publishing succeeds but a later job fails, rerun failed jobs in the **same run**
to preserve the exact build artifacts. Existing files are skipped during upload,
but registry hash verification still requires an exact match. Do not rebuild and
replace an already published version; use a new version for changed artifacts.
`workflow_dispatch` must be dispatched on the version tag, not `main`.

```bash
pip install langchain-finic==0.1.0a4
pip install 'langchain-finic[middleware]==0.1.0a4'
```

## Live smoke checks

Default tests and installation checks use mock HTTP and spend no credits. To make
two explicit live requests with fictional data:

```bash
export FINIC_API_KEY=your_key
make integration_test
```

Do not commit the key or record production inputs. Tests verify the response contract,
not an exact probability. No live calls run in pull requests or release CI.

## After publication

Use [listing metadata](listing.md) for a LangChain integration listing request; do not
claim LangChain or Finic endorsement. Obtain maintainer authorization before submitting
an issue on their behalf.

The original research application should migrate only after public installation is
verified: replace its workspace requirement with `langchain-finic==0.1.0a4`, remove
its workspace source/member declarations and duplicated package, adjust obsolete CI
paths, update its lockfile, and run its application suite. Until then it continues
using the local package. Future integration development belongs in this repository.

If a release is broken, publish a corrected version and consider yanking the bad
version with a clear reason. Do not delete published files or move release tags.
