"""Reproducibility manifest -- research plan §8, "Reproducibility artifact".

Every experiment run writes a manifest alongside its results capturing:
git commit hash (dirty-tree flag included), the exact resolved config,
seed, package versions, and timestamp. Without this, a reviewer (or
future you) cannot tell whether two runs claimed to be "the same
experiment" actually used the same code.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _git_info() -> dict:
    def run(cmd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    commit = run(["git", "rev-parse", "HEAD"])
    dirty = run(["git", "status", "--porcelain"])
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "commit": commit,
        "dirty": bool(dirty) if dirty is not None else None,
        "branch": branch,
    }


def _package_versions() -> dict:
    versions = {}
    for pkg in ["numpy", "scipy", "sklearn", "rdkit", "pandas", "matplotlib", "pymoo", "yaml"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


@dataclass
class RunManifest:
    experiment: str
    method: str
    split: str
    seed: int
    config: dict
    git: dict = field(default_factory=_git_info)
    packages: dict = field(default_factory=_package_versions)
    python_version: str = field(default_factory=platform.python_version)
    platform_info: str = field(default_factory=platform.platform)
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    prompt_version: str | None = None

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))


def warn_if_dirty(manifest: RunManifest) -> None:
    if manifest.git.get("commit") is None:
        print("[reproducibility] WARNING: not a git repository -- commit hash unavailable. "
              "Results from this run cannot be tied to a specific code version.", file=sys.stderr)
    elif manifest.git.get("dirty"):
        print("[reproducibility] WARNING: uncommitted changes present. Commit before running "
              "any result that will be reported in the paper (research plan §8 reproducibility "
              "artifact requirement).", file=sys.stderr)
