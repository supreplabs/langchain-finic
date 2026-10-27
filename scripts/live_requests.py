"""Pytest plugin recording only Finic request counts, never headers or payloads."""

import json
import os
from pathlib import Path
from threading import Lock
from unittest.mock import patch

import httpx


class RequestCounter:
    def __init__(self, path: Path):
        self.path = path
        self.count = 0
        self.lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text('{"requests": 0}\n')

    def record(self, request):
        if request.url.host == "api.finic.ai":
            with self.lock:
                self.count += 1
                temporary = self.path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"requests": self.count}) + "\n")
                temporary.replace(self.path)

    def __enter__(self):
        sync_send, async_send = httpx.Client.send, httpx.AsyncClient.send

        def send(client, request, **kwargs):
            self.record(request)
            return sync_send(client, request, **kwargs)

        async def asend(client, request, **kwargs):
            self.record(request)
            return await async_send(client, request, **kwargs)

        self.patches = [
            patch.object(httpx.Client, "send", send),
            patch.object(httpx.AsyncClient, "send", asend),
        ]
        for item in self.patches:
            item.start()
        return self

    def __exit__(self, *args):
        for item in reversed(self.patches):
            item.stop()


def pytest_sessionstart(session):
    path = Path(os.environ.get("FINIC_REQUEST_COUNTS", "artifacts/live/requests.json"))
    session.finic_counter = RequestCounter(path)
    session.finic_counter.__enter__()


def pytest_sessionfinish(session, exitstatus):
    session.finic_counter.__exit__()
