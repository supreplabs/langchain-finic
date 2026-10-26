"""Release presentation must select one version and classify PEP 440 releases correctly."""

import json
import runpy
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.version import InvalidVersion

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/release_metadata.py"
helpers = runpy.run_path(str(SCRIPT))


@pytest.fixture
def project():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


@pytest.mark.parametrize(
    ("version", "suffix", "prerelease"),
    [
        ("0.1.0a4", " — Alpha", True),
        ("0.1.0b1", " — Beta", True),
        ("0.1.0rc1", " — Release candidate", True),
        ("0.1.0.dev1", " — Development", True),
        ("0.1.0", "", False),
        ("0.1.0.post1", "", False),
    ],
)
def test_version_specific_notes_and_release_status(project, version, suffix, prerelease):
    project["version"] = version
    changelog = (
        f"# Changelog\n\n## Unreleased\nFuture work\n\n## {version}\nCurrent fix\n\n## 0.0.1\nOld fix\n"
    )
    metadata, notes = helpers["release_metadata"](project, changelog)
    assert metadata == {
        "version": version,
        "tag": f"v{version}",
        "title": f"langchain-finic {version}{suffix}",
        "prerelease": prerelease,
    }
    assert "Current fix" in notes
    assert "Future work" not in notes and "Old fix" not in notes
    assert f"pip install langchain-finic=={version}" in notes
    assert f"langchain-finic[middleware]=={version}" in notes
    assert "Python `>=3.11`" in notes
    assert "`langchain-core>=1,<2`" in notes
    assert "SHA256SUMS" in notes and "`dist/`" in notes


@pytest.mark.parametrize(
    "changelog",
    ["# Changelog\n", "## 0.1.0\n\n## 0.0.1\nOld fix", "## 0.1.0\nOne\n## 0.1.0\nTwo"],
)
def test_missing_empty_and_duplicate_entries_fail(changelog):
    with pytest.raises(ValueError, match="one nonempty"):
        helpers["changelog_entry"](changelog, "0.1.0")


def test_invalid_version_rejected(project):
    project["version"] = "not-a-version"
    with pytest.raises(InvalidVersion):
        helpers["release_metadata"](project, "## not-a-version\nFix")


def test_cli_generates_artifacts_from_project_metadata(tmp_path, monkeypatch, project):
    output = tmp_path / "presentation"
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output-dir", str(output)])
    runpy.run_path(str(SCRIPT), run_name="__main__")
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["version"] == project["version"]
    assert metadata["tag"] == f"v{project['version']}"
    assert "## Highlights" in (output / "notes.md").read_text()
