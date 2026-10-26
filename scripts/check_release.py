"""Reject a release tag that does not match the package version."""

import os
import tomllib
from pathlib import Path

project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
assert os.environ["GITHUB_REF_TYPE"] == "tag", "Dispatch the release workflow on a version tag"
assert os.environ["GITHUB_REF_NAME"] == f"v{project['version']}", "Tag and package version differ"
print(f"Validated release {project['name']} {project['version']}")
