"""Prepare version-specific GitHub release notes and metadata before publishing."""

import argparse
import json
import re
import tomllib
from pathlib import Path

from packaging.version import Version


def changelog_entry(changelog: str, version: str) -> str:
    """Select exactly one matching level-two version heading, excluding other releases."""
    parts = re.split(r"^## ([^\n]+)\n", changelog, flags=re.MULTILINE)
    matches = [parts[i + 1].strip() for i in range(1, len(parts), 2) if parts[i].strip() == version]
    if len(matches) != 1 or not matches[0]:
        raise ValueError(f"CHANGELOG.md must contain one nonempty '## {version}' section")
    return matches[0]


def release_metadata(project: dict, changelog: str) -> tuple[dict, str]:
    """Use PEP 440 version semantics for alpha/beta/RC/development versus stable releases."""
    raw_version = project["version"]
    version = Version(raw_version)
    name = project["name"]
    title = f"{name} {raw_version}"
    if version.is_devrelease:
        title += " — Development"
    elif version.pre:
        title += " — " + {"a": "Alpha", "b": "Beta", "rc": "Release candidate"}[version.pre[0]]
    highlights = changelog_entry(changelog, raw_version)
    metadata = {
        "version": raw_version,
        "tag": f"v{raw_version}",
        "title": title,
        "prerelease": version.is_prerelease,
    }
    dependencies = "\n".join(f"- `{requirement}`" for requirement in project["dependencies"])
    middleware = ", ".join(f"`{item}`" for item in project["optional-dependencies"]["middleware"])
    docs = project["urls"]["Documentation"]
    repository = project["urls"]["Repository"]
    tag = metadata["tag"]
    notes = f"""## Highlights

{highlights}

## Installation

```bash
pip install {name}=={raw_version}
# Optional agent middleware:
pip install '{name}[middleware]=={raw_version}'
```

## Compatibility

- Python `{project["requires-python"]}`.
{dependencies}
- The middleware extra additionally requires {middleware}.

## Checksums and provenance

The attached wheel and source distribution have SHA-256 hashes in `SHA256SUMS`.
Put the wheel and source distribution in a `dist/` directory beside `SHA256SUMS`, then run:

```bash
sha256sum --check SHA256SUMS
```

On macOS, use `shasum -a 256 --check SHA256SUMS`. Registry files are published
with attestations through GitHub Actions Trusted Publishing.

[Documentation]({docs}) · [PyPI](https://pypi.org/project/{name}/{raw_version}/) ·
[Source at this tag]({repository}/tree/{tag})
"""
    return metadata, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    metadata, notes = release_metadata(project, Path("CHANGELOG.md").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "notes.md").write_text(notes)
    print(metadata["title"])


if __name__ == "__main__":
    main()
