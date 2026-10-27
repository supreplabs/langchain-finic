"""Publishing checks must tolerate temporary invisibility without masking permanent failure."""

import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

install = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/verify_install.py"))[
    "install_from_pypi"
]


@pytest.mark.parametrize("failures", [0, 1, 4])
def test_install_retries_then_succeeds(monkeypatch, failures):
    calls, delays = [], []

    def run(command, *, check, timeout):
        calls.append(command)
        assert check is False and timeout == 120
        return SimpleNamespace(returncode=1 if len(calls) <= failures else 0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("time.sleep", delays.append)
    assert install("/tmp/venv/bin/python", "0.1.0b1") == 0
    assert len(calls) == failures + 1
    assert delays == [5, 10, 20, 40][:failures]
    for command in calls:
        assert command == [
            "uv",
            "pip",
            "install",
            "--python",
            "/tmp/venv/bin/python",
            "--index-url",
            "https://pypi.org/simple",
            "--refresh-package",
            "langchain-finic",
            "langchain-finic==0.1.0b1",
        ]


@pytest.mark.parametrize("timeout", [False, True])
def test_install_exhausts_finite_budget(monkeypatch, timeout):
    calls, delays = [], []

    def run(command, **kwargs):
        calls.append(command)
        if timeout:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("time.sleep", delays.append)
    assert install("/tmp/venv/bin/python", "0.1.0b1") == 1
    assert len(calls) == 5
    assert delays == [5, 10, 20, 40]
