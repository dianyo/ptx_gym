"""CLI and report semantics without faking GPU execution."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ptx_gym.benchmarks.softmax_ce_hvp import _full_report_complete, main, parser


@pytest.mark.parametrize("value", ["0x3", "2x0", "2x8193", "2x3x4", "abc", "2.5x3"])
def test_rejects_invalid_sweep_shapes(value):
    with pytest.raises(SystemExit):
        parser().parse_args(["--mode", "sweep", "--shape", value])


def test_custom_shapes_are_sweep_only():
    with pytest.raises(SystemExit):
        main(["--mode", "full", "--shape", "2x33"])


def test_help_has_no_triton_dependency():
    result = subprocess.run(
        [sys.executable, "-m", "ptx_gym.benchmarks.softmax_ce_hvp", "--help"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "smoke" in result.stdout and "sweep" in result.stdout


def test_cpu_run_reports_setup_failure_without_gpu_results(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert main(["--output-dir", str(tmp_path)]) == 1
    [run_directory] = list(tmp_path.iterdir())
    summary = json.loads((run_directory / "summary.json").read_text())
    assert summary["passed"] is False
    assert summary["tasks"] == []
    assert "CUDA is unavailable" in summary["error"]
    assert "device_name" not in json.loads(
        (run_directory / "environment.json").read_text()
    )


def test_full_success_requires_sanitizer_and_successful_profile():
    result = {
        "passed": True,
        "p50": 0.01,
        "sanitizer_report": {"clean": True},
        "ncu_report": {"available": True, "return_code": 0, "metrics": {"example": 1}},
    }
    assert _full_report_complete(result)
    result["ncu_report"]["return_code"] = 1
    assert not _full_report_complete(result)
    result["ncu_report"] = {}
    assert not _full_report_complete(result)
    result["sanitizer_report"] = {}
    assert not _full_report_complete(result)
