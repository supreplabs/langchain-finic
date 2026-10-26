"""Environment defaults shared by the transport and typed request models."""

import os


def env_option(name: str, default):
    return os.getenv(f"FINIC_{name}", default)
