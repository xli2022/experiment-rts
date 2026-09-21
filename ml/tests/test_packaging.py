"""Frozen observation contracts must survive non-editable installation."""

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def test_wheel_contains_every_frozen_migration_spec(tmp_path):
    project = Path(__file__).resolve().parents[1]
    shutil.copy2(project / "pyproject.toml", tmp_path / "pyproject.toml")
    shutil.copytree(project / "rtsml", tmp_path / "rtsml", ignore=shutil.ignore_patterns("__pycache__"))
    # Build away from the source tree so packaging neither reads stale build
    # output nor leaves generated files in the developer's checkout.
    subprocess.run(
        [sys.executable, "-c", "from setuptools.build_meta import build_wheel; build_wheel('dist')"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    wheel, = (tmp_path / "dist").glob("*.whl")
    expected = sorted((project / "rtsml").glob("spec*.json"))
    assert any(path.name == "spec-v5.json" for path in expected)
    with zipfile.ZipFile(wheel) as archive:
        for path in expected:
            packaged = archive.read(f"rtsml/{path.name}")
            assert json.loads(packaged) == json.loads(path.read_bytes())
