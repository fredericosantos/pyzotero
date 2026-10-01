"""Shared test fixtures."""

from __future__ import annotations

import pytest

from pyzotero._config import ENV_VARS


@pytest.fixture(autouse=True)
def _no_user_settings(monkeypatch, tmp_path_factory):
    """Keep every test away from the user's real settings and credentials.

    Without this, a configured remote/WebDAV setup on the machine would make
    tests talk to the real library and WebDAV server.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))
    for var in ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)
