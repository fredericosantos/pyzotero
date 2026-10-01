"""Settings for the CLI and MCP server: library mode and file storage.

The settings come from two sources. An environment variable takes precedence
over the value in the config file, ``pyzotero/config.json`` under
``$XDG_CONFIG_HOME`` (or ``~/.config``). ``pyzotero setup`` writes the file.

Two settings are independent of each other:

- ``mode``: ``local`` talks to the Zotero desktop app on this computer;
  ``remote`` talks to the zotero.org Web API and needs an API key.
- ``storage``: where attachment files live. ``zotero`` is Zotero File
  Storage; ``webdav`` is a WebDAV server that Zotero syncs files to.
"""

from __future__ import annotations

import dataclasses
import json
import os
import stat
from pathlib import Path
from typing import Any

MODES = ("local", "remote")
STORAGES = ("zotero", "webdav")

# Setting name -> environment variable that overrides it.
ENV_VARS = {
    "mode": "PYZOTERO_MODE",
    "api_key": "PYZOTERO_API_KEY",
    "library_id": "PYZOTERO_LIBRARY_ID",
    "library_type": "PYZOTERO_LIBRARY_TYPE",
    "storage": "PYZOTERO_STORAGE",
    "webdav_url": "PYZOTERO_WEBDAV_URL",
    "webdav_username": "PYZOTERO_WEBDAV_USERNAME",
    "webdav_password": "PYZOTERO_WEBDAV_PASSWORD",
    "unpaywall_email": "PYZOTERO_UNPAYWALL_EMAIL",
}


class ConfigError(RuntimeError):
    """The settings are incomplete or hold an invalid value."""


@dataclasses.dataclass(frozen=True)
class Settings:
    """Resolved settings. Use :func:`load_settings` to build one."""

    mode: str = "local"
    api_key: str | None = None
    library_id: str | None = None
    library_type: str = "user"
    storage: str = "zotero"
    webdav_url: str | None = None
    webdav_username: str | None = None
    webdav_password: str | None = None
    unpaywall_email: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            msg = f"mode must be one of {', '.join(MODES)}, got {self.mode!r}"
            raise ConfigError(msg)
        if self.storage not in STORAGES:
            msg = f"storage must be one of {', '.join(STORAGES)}, got {self.storage!r}"
            raise ConfigError(msg)
        if self.library_type not in {"user", "group"}:
            msg = f"library_type must be 'user' or 'group', got {self.library_type!r}"
            raise ConfigError(msg)

    def require_remote(self) -> None:
        """Raise ConfigError unless the Web API settings are complete."""
        missing = [n for n in ("api_key", "library_id") if not getattr(self, n)]
        if missing:
            names = ", ".join(f"{n} ({ENV_VARS[n]})" for n in missing)
            msg = f"Remote mode needs {names}. Run 'pyzotero setup'."
            raise ConfigError(msg)

    def webdav_credentials(self) -> tuple[str, str, str]:
        """Return ``(url, username, password)``.

        Raises ConfigError if any of them is not set.
        """
        url, user, password = (
            self.webdav_url,
            self.webdav_username,
            self.webdav_password,
        )
        if not (url and user and password):
            names = ("webdav_url", "webdav_username", "webdav_password")
            missing = [n for n in names if not getattr(self, n)]
            listed = ", ".join(f"{n} ({ENV_VARS[n]})" for n in missing)
            msg = f"WebDAV storage needs {listed}. Run 'pyzotero setup'."
            raise ConfigError(msg)
        return url, user, password

    def to_file_dict(self) -> dict[str, Any]:
        """Return the settings that are set, for the config file."""
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}


def config_path() -> Path:
    """Return the path of the config file."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "pyzotero" / "config.json"


def _read_file() -> dict[str, Any]:
    path = config_path()
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except ValueError as exc:
        msg = f"{path} does not hold valid JSON: {exc}"
        raise ConfigError(msg) from exc
    if not isinstance(data, dict):
        msg = f"{path} must hold a JSON object"
        raise ConfigError(msg)
    unknown = set(data) - set(ENV_VARS)
    if unknown:
        msg = f"{path} has unknown settings: {', '.join(sorted(unknown))}"
        raise ConfigError(msg)
    return data


def load_settings() -> Settings:
    """Return the settings from the environment and the config file."""
    values = _read_file()
    for name, var in ENV_VARS.items():
        env = os.environ.get(var)
        if env:
            values[name] = env
    return Settings(**{k: str(v) for k, v in values.items()})


def save_settings(settings: Settings) -> Path:
    """Write ``settings`` to the config file with mode 0600. Return its path."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create the file with 0600 before writing, so the secrets are never
    # readable by others, even briefly.
    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR
    )
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(settings.to_file_dict(), indent=2) + "\n")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return path
