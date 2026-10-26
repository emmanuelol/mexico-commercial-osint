import subprocess
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_quickstart_pipeline_example_runs():
    script_path = REPO_ROOT / "examples" / "01_quickstart_pipeline.py"
    assert script_path.exists(), f"Missing example script: {script_path}"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"Script failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "RANK" in result.stdout or "PROSPECT" in result.stdout or "Pipeline" in result.stdout
    assert "Direct Email" in result.stdout or "WhatsApp" in result.stdout or "Score" in result.stdout


def test_custom_scian_scoring_example_runs():
    script_path = REPO_ROOT / "examples" / "02_custom_scian_scoring.py"
    assert script_path.exists(), f"Missing example script: {script_path}"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"Script failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "CUSTOM" in result.stdout.upper() or "SCIAN" in result.stdout.upper()
