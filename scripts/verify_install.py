"""Install the exact release from public PyPI despite short index propagation delays."""

import argparse
import subprocess
import time
import tomllib
from pathlib import Path


def install_from_pypi(python: str, version: str) -> int:
    command = [
        "uv",
        "pip",
        "install",
        "--python",
        python,
        "--index-url",
        "https://pypi.org/simple",
        "--refresh-package",
        "langchain-finic",
        f"langchain-finic=={version}",
    ]
    for attempt in range(5):
        try:
            result = subprocess.run(command, check=False, timeout=120)
            status = result.returncode
        except subprocess.TimeoutExpired:
            status = 1
        if status == 0:
            return 0
        if attempt < 4:
            delay = 5 * 2**attempt
            print(f"Public installation attempt {attempt + 1} failed; retrying in {delay}s", flush=True)
            time.sleep(delay)
    print("Public installation failed after five attempts", flush=True)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, help="Python executable in the verification environment")
    args = parser.parse_args()
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    return install_from_pypi(args.python, project["version"])


if __name__ == "__main__":
    raise SystemExit(main())
