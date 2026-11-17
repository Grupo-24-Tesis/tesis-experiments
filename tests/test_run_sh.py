import os
import shutil
import stat
import subprocess

import pytest

PIPELINE_DIR = os.path.abspath(os.environ.get(
    "TESIS_PIPELINE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "tesis_face_recognition_pipeline")))

FAKE_PYTHON = """#!/bin/sh
venv=$(basename "$(dirname "$(dirname "$0")")")
echo "$venv $*" >> "$RUN_LOG"
"""


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    shutil.copy(os.path.join(PIPELINE_DIR, "run.sh"), root / "run.sh")
    for venv in ("track", "faces"):
        bin_dir = root / "venvs" / venv / "bin"
        bin_dir.mkdir(parents=True)
        python = bin_dir / "python"
        python.write_text(FAKE_PYTHON)
        python.chmod(python.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "v.mp4").write_bytes(b"")
    return root


def run_script(repo, *args, **env_overrides):
    log = repo / "calls.log"
    env = {k: v for k, v in os.environ.items()
           if k not in ("FACE_GATING", "FACE_ADAPTIVE", "PIPELINE_MODE")}
    env.update(env_overrides)
    env["RUN_LOG"] = str(log)
    video = str(repo.parent / "v.mp4")
    result = subprocess.run(["bash", str(repo / "run.sh"), *args, video, str(repo / "out")],
                            env=env, capture_output=True, text=True)
    calls = log.read_text().splitlines() if log.exists() else []
    return result, calls


def test_gate_runs_before_the_face_pass_by_default(repo):
    result, calls = run_script(repo)

    assert result.returncode == 0
    assert [call.split()[:2] for call in calls] == [["track", "presence_gate.py"],
                                                    ["faces", "face_identities.py"]]
    assert "--presence" in calls[1]


def test_face_gating_zero_skips_the_gate(repo):
    result, calls = run_script(repo, FACE_GATING="0")

    assert result.returncode == 0
    assert len(calls) == 1
    assert calls[0].startswith("faces face_identities.py")
    assert "--presence" not in calls[0]


def test_no_gating_option_wins_over_face_gating_one(repo):
    result, calls = run_script(repo, "--no-gating", FACE_GATING="1")

    assert result.returncode == 0
    assert len(calls) == 1
    assert "--presence" not in calls[0]


def test_invalid_face_gating_stops_with_a_clear_message(repo):
    result, calls = run_script(repo, FACE_GATING="si")

    assert result.returncode == 1
    assert "FACE_GATING debe ser 0 o 1, no 'si'" in result.stderr
    assert calls == []


def test_adaptive_sampling_with_gating_stops_before_the_gate(repo):
    result, calls = run_script(repo, FACE_ADAPTIVE="1")

    assert result.returncode == 1
    assert "FACE_ADAPTIVE=1" in result.stderr
    assert calls == []


def test_adaptive_sampling_without_gating_runs_the_face_pass(repo):
    result, calls = run_script(repo, FACE_ADAPTIVE="1", FACE_GATING="0")

    assert result.returncode == 0
    assert len(calls) == 1
