"""Validate the public wheel and sdist without trusting checkout imports."""

import email
import hashlib
import json
import tarfile
import tomllib
import zipfile
from pathlib import Path


def main():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    version = project["version"]
    paths = [Path(f"dist/langchain_finic-{version}{suffix}") for suffix in ("-py3-none-any.whl", ".tar.gz")]
    assert sorted(p for p in Path("dist").iterdir() if p.name != ".gitignore") == sorted(paths), (
        "Stale or unexpected distribution files"
    )
    with zipfile.ZipFile(paths[0]) as wheel:
        names = wheel.namelist()
        assert "langchain_finic/py.typed" in names
        metadata_name = f"langchain_finic-{version}.dist-info/METADATA"
        metadata = email.message_from_bytes(wheel.read(metadata_name))
        assert metadata["Name"] == "langchain-finic"
        assert metadata["Version"] == version
        assert metadata["License-Expression"] == "MIT"
        assert metadata["Requires-Python"] == ">=3.11"
        assert f"langchain_finic-{version}.dist-info/licenses/LICENSE" in names
        assert any("extra == 'middleware'" in value for value in metadata.get_all("Requires-Dist"))
        for name in names:
            assert name.startswith(("langchain_finic/", f"langchain_finic-{version}.dist-info/")), name
            assert "__pycache__" not in name and not name.endswith(".pyc"), name
            if name.startswith("langchain_finic/"):
                assert name.endswith(".py") or name == "langchain_finic/py.typed", name
    allowed = {
        "src",
        "tests",
        "examples",
        "docs",
        "theme",
        "scripts",
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "RELEASING.md",
        "LANGCHAIN_LISTING.md",
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
        "Makefile",
        "mkdocs.yml",
        "uv.lock",
        "PKG-INFO",
        ".gitignore",
    }
    with tarfile.open(paths[1]) as sdist:
        names = []
        for member in sdist.getmembers():
            path = Path(member.name)
            assert path.parts[0] == f"langchain_finic-{version}", member.name
            assert len(path.parts) > 1 and path.parts[1] in allowed, member.name
            assert not member.issym() and not member.islnk(), member.name
            assert ".." not in path.parts and "__pycache__" not in path.parts, member.name
            assert not any(part.startswith(".env") for part in path.parts), member.name
            assert path.suffix not in {".pem", ".key", ".pyc"}, member.name
            names.append(str(Path(*path.parts[1:])))
        assert "LICENSE" in names and "src/langchain_finic/py.typed" in names
        assert "tests/unit_tests/test_standard.py" in names
        assert "theme/main.html" in names
    print(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2))


if __name__ == "__main__":
    main()
