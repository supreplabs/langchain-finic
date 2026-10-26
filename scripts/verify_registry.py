"""Download exact released files, matching local hashes; never resolve dependencies here."""

import argparse
import hashlib
import json
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", choices=["testpypi", "pypi"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    host = "test.pypi.org" if args.registry == "testpypi" else "pypi.org"
    url = f"https://{host}/pypi/langchain-finic/{version}/json"
    for attempt in range(12):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                files = json.load(response)["urls"]
            if len(files) >= 2:
                break
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
        if attempt == 11:
            raise RuntimeError("Release files did not appear in registry")
        time.sleep(10)
    args.output.mkdir(parents=True, exist_ok=False)
    local = {p.name: p for p in Path("dist").iterdir() if p.name != ".gitignore"}
    assert set(local) == {f["filename"] for f in files}, "Registry file set differs"
    for info in files:
        filename = info["filename"]
        expected = hashlib.sha256(local[filename].read_bytes()).hexdigest()
        assert info["digests"]["sha256"] == expected, f"Registry hash mismatch: {filename}"
        with urllib.request.urlopen(info["url"], timeout=60) as response:
            content = response.read()
        assert hashlib.sha256(content).hexdigest() == expected, f"Downloaded hash mismatch: {filename}"
        (args.output / filename).write_bytes(content)
        print(f"Verified {args.registry}: {filename} sha256={expected}")


if __name__ == "__main__":
    main()
