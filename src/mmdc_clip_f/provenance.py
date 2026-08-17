"""Small, dependency-light helpers for experiment provenance."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def combined_sha256(paths: Iterable[str | os.PathLike[str]]) -> str:
    records = [(str(Path(path).name), sha256_file(path)) for path in paths]
    return sha256_json(records)


def atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def git_state(directory: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(directory)

    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain=v1")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def environment_state() -> dict[str, Any]:
    packages: dict[str, str] = {}
    try:
        from importlib.metadata import PackageNotFoundError, version

        for package in (
            "numpy",
            "pillow",
            "pydicom",
            "scikit-learn",
            "torch",
            "torchvision",
            "transformers",
        ):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                pass
    except ImportError:
        pass
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def create_run_dir(root: str | os.PathLike[str], prefix: str, run_name: str | None = None) -> Path:
    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    name = (
        run_name or f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    )
    name_path = Path(name)
    if (
        not name
        or name_path.is_absolute()
        or len(name_path.parts) != 1
        or name_path.name != name
        or name in {".", ".."}
    ):
        raise ValueError("run_name must be one non-empty path component")
    run_dir = root_path / name
    run_dir.mkdir(exist_ok=False)
    return run_dir
