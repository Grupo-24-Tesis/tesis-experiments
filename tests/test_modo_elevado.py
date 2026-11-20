import json
import os
import subprocess
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import pytest

from bench import bench_compuerta, estadisticas_runner, modo_elevado, mot, pistas
from harness import run as harness_run
from harness import runlog

TRACKER = {"tracker_type": "tracktrack", "match_thresh": 0.7, "min_track_len": 3,
           "with_reid": True, "model": "yolo26m-reid.onnx"}
FAKE_ENV = {"PYTHONPATH": "exp"}


def identity(correctos, **extra):
    base = {"personas_etiquetadas": 10, "identidades": correctos, "encontradas": correctos,
            "duplicados": 0, "mezcladas": 0, "sin_asignar": 0, "pureza": 1.0,
            "correctos": correctos, "cara_correcta": correctos, "cara_sin_verificar": 0}
    return dict(base, **extra)


def test_setting_names_round_trip():
    assert modo_elevado.setting_name(1280, 0.2, 2) == "i1280_d0p2_m2"
    assert modo_elevado.setting_name(640, 0.5, 3) == "i640_d0p5_m3"
    assert modo_elevado.setting_name(640, 1.0, 3) == "i640_d1_m3"
    assert modo_elevado.parse_setting("i1280_d0p2_m2") == {"imgsz": 1280, "intervalo": 0.2,
                                                           "min_track": 2}
    assert modo_elevado.parse_setting("i640_d1_m3")["intervalo"] == 1.0
    assert modo_elevado.parse_setting("tiempos.json") is None
    grid = modo_elevado.settings_grid([640, 1280], [0.5, 0.2, 0.1], [2, 3])
    assert len(grid) == 12
    assert grid[0] == ("i640_d0p5_m2", 640, 0.5, 2)
    assert grid[-1] == ("i1280_d0p1_m3", 1280, 0.1, 3)


def test_build_config_overrides_only_the_given_values(tmp_path):
    path = tmp_path / "tracker.yaml"
    path.write_text("tracker_type: tracktrack\nmatch_thresh: 0.7\nmin_track_len: 3\n"
                    "with_reid: True\nmodel: yolo26m-reid.onnx\n")
    tracker = modo_elevado.load_tracker(str(path))
    assert tracker == TRACKER
    config = modo_elevado.build_config("v/MOT20-01.mp4", 1280, 0.2, 2, tracker)
    assert config == {"video": os.path.abspath("v/MOT20-01.mp4"),
                      "stage1": {"YOLO_IMGSZ": 1280, "DETECT_INTERVAL_S": 0.2},
                      "tracker": dict(TRACKER, min_track_len=2), "stage2": {}}
    assert tracker["min_track_len"] == 3


def test_unique_videos_rejects_repeated_names():
    with pytest.raises(SystemExit, match="mismo nombre"):
        modo_elevado.unique_videos(["a/x.mp4", "b/x.mp4"])


def write_stage1_outputs(directory, n_tracks=2):
    candidates = os.path.join(directory, "candidates")
    os.makedirs(candidates, exist_ok=True)
    manifest = {"video": "x.mp4", "fps": 25.0, "detect_every": 12, "tracks": {}, "boxes": {}}
    runlog.write_json(os.path.join(candidates, "manifest.json"), manifest)
    runlog.write_json(os.path.join(candidates, "stage1_summary.json"),
                      {"n_tracks": n_tracks, "n_crops": 5})


def test_resumability_checks(tmp_path):
    directory = str(tmp_path / "i640_d0p5_m3" / "x")
    assert not modo_elevado.stage1_done(directory)
    os.makedirs(os.path.join(directory, "candidates"))
    runlog.write_json(os.path.join(directory, "candidates", "manifest.json"), {})
    assert not modo_elevado.stage1_done(directory)
    runlog.write_json(os.path.join(directory, "candidates", "stage1_summary.json"), {})
    assert modo_elevado.stage1_done(directory)
    assert not modo_elevado.stage2_done(directory)
    os.makedirs(os.path.join(directory, "faces"))
    runlog.write_json(os.path.join(directory, "faces", "results.json"), {"faces": []})
    assert modo_elevado.stage2_done(directory)


class FakeRun:
    def __init__(self, on_call=None, code=0):
        self.calls = []
        self.on_call = on_call
        self.code = code

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        if self.on_call:
            self.on_call(cmd)
        return types.SimpleNamespace(returncode=self.code)


def stage1_effect(cmd):
    write_stage1_outputs(cmd[cmd.index("--run-dir") + 1])


def stage2_effect(cmd):
    out = cmd[cmd.index("--out") + 1]
    os.makedirs(out, exist_ok=True)
    runlog.write_json(os.path.join(out, "results.json"), {"tracks_checked": 2, "faces": []})
    runlog.write_json(os.path.join(out, "stage2_summary.json"), {"n_faces": 0})


def args(**values):
    return types.SimpleNamespace(**values)


@pytest.fixture
def fake_env(monkeypatch):
    monkeypatch.setattr(modo_elevado.runlog, "isolated_env", lambda: dict(FAKE_ENV))
    monkeypatch.setattr(modo_elevado, "load_tracker", lambda: dict(TRACKER))


def test_seguimiento_runs_pending_settings_and_skips_done(tmp_path, monkeypatch, fake_env):
    out = tmp_path / "corridas"
    write_stage1_outputs(str(out / "i640_d0p5_m3" / "x"))
    stale = out / "i640_d0p5_m2" / "x" / "candidates" / "7"
    stale.mkdir(parents=True)
    (stale / "candidate_000.jpg").write_text("")
    fake = FakeRun(stage1_effect)
    monkeypatch.setattr(subprocess, "run", fake)
    code = modo_elevado.command_seguimiento(args(out=str(out), imgsz="640", intervalo="0.5",
                                                 min_track="2,3", videos=["v/x.mp4"]))
    assert code == 0
    assert len(fake.calls) == 1
    directory = str(out / "i640_d0p5_m2" / "x")
    assert fake.calls[0][0][fake.calls[0][0].index("--run-dir") + 1] == directory
    assert not stale.exists()
    config = runlog.read_json(os.path.join(directory, "config.json"))
    assert config["tracker"]["min_track_len"] == 2
    times = runlog.read_json(str(out / "tiempos.json"))
    assert list(times) == ["i640_d0p5_m2"]
    assert "seguimiento_s" in times["i640_d0p5_m2"]["x"]
    fake.calls.clear()
    modo_elevado.command_seguimiento(args(out=str(out), imgsz="640", intervalo="0.5",
                                          min_track="2,3", videos=["v/x.mp4"]))
    assert fake.calls == []


def test_seguimiento_reports_failure_without_recording_time(tmp_path, monkeypatch, fake_env):
    monkeypatch.setattr(subprocess, "run", FakeRun(code=1))
    out = tmp_path / "corridas"
    code = modo_elevado.command_seguimiento(args(out=str(out), imgsz="640", intervalo="0.5",
                                                 min_track="3", videos=["v/x.mp4"]))
    assert code == 1
    assert runlog.read_json(str(out / "tiempos.json"), {}) == {}
    assert runlog.read_json(str(out / "i640_d0p5_m3" / "x" / "etapas.json")) == {"stage1": 1}


def test_rostros_needs_stage1_and_is_resumable(tmp_path, monkeypatch, fake_env):
    out = tmp_path / "corridas"
    write_stage1_outputs(str(out / "i640_d0p5_m3" / "x"))
    fake = FakeRun(stage2_effect)
    monkeypatch.setattr(subprocess, "run", fake)
    code = modo_elevado.command_rostros(args(out=str(out), ajuste="i640_d0p5_m3,i1280_d0p5_m3",
                                             videos=["v/x.mp4"]))
    assert code == 1
    assert len(fake.calls) == 1
    assert "rostros_s" in runlog.read_json(str(out / "tiempos.json"))["i640_d0p5_m3"]["x"]
    fake.calls.clear()
    modo_elevado.command_rostros(args(out=str(out), ajuste="i640_d0p5_m3", videos=["v/x.mp4"]))
    assert fake.calls == []
    with pytest.raises(SystemExit, match="--ajuste"):
        modo_elevado.command_rostros(args(out=str(out), ajuste="otro", videos=["v/x.mp4"]))


def write_stage_codes(directory, **codes):
    runlog.write_json(os.path.join(directory, "etapas.json"), codes)


def write_stage2_outputs(directory, summary=True):
    faces = os.path.join(directory, "faces")
    os.makedirs(faces, exist_ok=True)
    runlog.write_json(os.path.join(faces, "results.json"), {"tracks_checked": 9, "faces": []})
    if summary:
        runlog.write_json(os.path.join(faces, "stage2_summary.json"), {"n_faces": 0})


def test_seguimiento_reruns_stage1_that_exited_with_error(tmp_path, monkeypatch, fake_env, capsys):
    out = tmp_path / "corridas"
    directory = str(out / "i640_d0p5_m3" / "x")
    write_stage1_outputs(directory)
    write_stage_codes(directory, stage1=1)
    fake = FakeRun(stage1_effect)
    monkeypatch.setattr(subprocess, "run", fake)
    code = modo_elevado.command_seguimiento(args(out=str(out), imgsz="640", intervalo="0.5",
                                                 min_track="3", videos=["v/x.mp4"]))
    assert code == 0
    assert len(fake.calls) == 1
    assert "ya hecho" not in capsys.readouterr().out
    assert runlog.read_json(os.path.join(directory, "etapas.json")) == {"stage1": 0}
    assert "seguimiento_s" in runlog.read_json(str(out / "tiempos.json"))["i640_d0p5_m3"]["x"]


def test_seguimiento_skips_clean_stage1(tmp_path, monkeypatch, fake_env, capsys):
    out = tmp_path / "corridas"
    directory = str(out / "i640_d0p5_m3" / "x")
    write_stage1_outputs(directory)
    write_stage_codes(directory, stage1=0)
    fake = FakeRun(stage1_effect)
    monkeypatch.setattr(subprocess, "run", fake)
    code = modo_elevado.command_seguimiento(args(out=str(out), imgsz="640", intervalo="0.5",
                                                 min_track="3", videos=["v/x.mp4"]))
    assert code == 0
    assert fake.calls == []
    assert "seguimiento ya hecho, se omite" in capsys.readouterr().out


@pytest.mark.parametrize("stage2_code, summary", [(1, True), (0, False)])
def test_rostros_reruns_stage2_that_did_not_finish(tmp_path, monkeypatch, fake_env, capsys,
                                                   stage2_code, summary):
    out = tmp_path / "corridas"
    directory = str(out / "i640_d0p5_m3" / "x")
    write_stage1_outputs(directory)
    write_stage2_outputs(directory, summary=summary)
    write_stage_codes(directory, stage1=0, stage2=stage2_code)
    fake = FakeRun(stage2_effect)
    monkeypatch.setattr(subprocess, "run", fake)
    code = modo_elevado.command_rostros(args(out=str(out), ajuste="i640_d0p5_m3",
                                             videos=["v/x.mp4"]))
    assert code == 0
    assert len(fake.calls) == 1
    assert "ya hecho" not in capsys.readouterr().out
    assert runlog.read_json(os.path.join(directory, "etapas.json")) == {"stage1": 0, "stage2": 0}
    assert runlog.read_json(os.path.join(directory, "faces", "results.json"))["tracks_checked"] == 2


def test_rostros_does_not_run_on_failed_stage1(tmp_path, monkeypatch, fake_env, capsys):
    out = tmp_path / "corridas"
    directory = str(out / "i640_d0p5_m3" / "x")
    write_stage1_outputs(directory)
    write_stage_codes(directory, stage1=1)
    fake = FakeRun(stage2_effect)
    monkeypatch.setattr(subprocess, "run", fake)
    code = modo_elevado.command_rostros(args(out=str(out), ajuste="i640_d0p5_m3",
                                             videos=["v/x.mp4"]))
    assert code == 1
    assert fake.calls == []
    assert "terminó con código 1" in capsys.readouterr().out
    assert not os.path.exists(os.path.join(directory, "faces"))


def capture(monkeypatch):
    fake = FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    return fake


def comparable(call):
    cmd, kwargs = call
    return cmd, {key: (value.name if hasattr(value, "name") else value)
                 for key, value in kwargs.items()}


def test_stage_commands_match_harness_run(tmp_path, monkeypatch, fake_env):
    directory = str(tmp_path / "i640_d0p5_m3" / "x")
    os.makedirs(directory)
    cfg_path = os.path.join(directory, "config.json")
    fake = capture(monkeypatch)
    modo_elevado.run_command(modo_elevado.stage1_command(directory),
                             os.path.join(directory, "stage1"))
    harness_run.build_stage1(cfg_path)(directory, dict(FAKE_ENV))
    assert comparable(fake.calls[0]) == comparable(fake.calls[1])
    candidates = os.path.join(directory, "candidates")
    modo_elevado.run_command(modo_elevado.stage2_command(directory),
                             os.path.join(directory, "stage2"))
    harness_run.build_stage2(cfg_path, candidates)(directory, dict(FAKE_ENV))
    assert comparable(fake.calls[2]) == comparable(fake.calls[3])
    cmd, kwargs = fake.calls[0]
    assert cmd == [harness_run.PY["track"], "-m", "harness.stage1_runner", "--config", cfg_path,
                   "--run-dir", directory, "--out", candidates]
    assert kwargs["cwd"] == runlog.PIPELINE_DIR
    assert kwargs["env"] == FAKE_ENV
    assert fake.calls[2][0][0] == harness_run.PY["faces"]


def test_statistics_command_uses_track_venv_and_pipeline_cwd(tmp_path, monkeypatch, fake_env):
    fake = capture(monkeypatch)
    raw = str(tmp_path / "x.json")
    modo_elevado.run_command(modo_elevado.statistics_command("v/x.mp4", raw),
                             str(tmp_path / "x"))
    cmd, kwargs = fake.calls[0]
    assert cmd == [harness_run.PY["track"], "-m", "bench.estadisticas_runner", "--video",
                   os.path.abspath("v/x.mp4"), "--out", raw]
    assert kwargs["cwd"] == runlog.PIPELINE_DIR
    assert kwargs["env"] == FAKE_ENV


def test_footage_from_checks():
    raw = {"width": 1920, "height": 1080,
           "checks": [[0, [100.0, 200.0, 300.0]], [12, []], [24, [400.0]], [36, [50.0, 60.0]]]}
    stats = modo_elevado.footage_from_checks(raw)
    assert stats["altura_mediana_px"] == 150.0
    assert stats["altura_relativa"] == pytest.approx(150 / 1080)
    assert stats["personas_por_revision_mediana"] == 1.5
    assert stats["fraccion_con_personas"] == 0.75
    assert (stats["ancho"], stats["alto"], stats["revisiones"]) == (1920, 1080, 4)
    empty = modo_elevado.footage_from_checks({"width": 10, "height": 0, "checks": []})
    assert empty["altura_mediana_px"] == 0.0
    assert empty["altura_relativa"] == 0.0
    assert empty["fraccion_con_personas"] == 0.0


class FakeCapture:
    def __init__(self, frames, unreadable=()):
        self.frames = frames
        self.unreadable = set(unreadable)
        self.index = -1

    def grab(self):
        self.index += 1
        return self.index < self.frames

    def retrieve(self):
        return self.index not in self.unreadable, self.index


def test_statistics_runner_checks_twice_per_second():
    assert estadisticas_runner.check_step(25.0) == 12
    assert estadisticas_runner.check_step(30.0) == 15
    assert estadisticas_runner.check_step(1.0) == 1
    assert (estadisticas_runner.CONF, estadisticas_runner.IMGSZ) == (0.25, 640)
    assert estadisticas_runner.YOLO_WEIGHTS == os.path.join(runlog.PIPELINE_DIR, "yolo26m.pt")
    checks, unreadable, frames = estadisticas_runner.read_checks(
        FakeCapture(30, unreadable=[24]), 12, lambda frame: [frame * 1.0] if frame else [])
    assert checks == [[0, []], [12, [12.0]]]
    assert (unreadable, frames) == (1, 30)


def test_best_mode_and_tie_rule():
    assert modo_elevado.best_mode(identity(3), {"i640_d0p5_m3": identity(4)}) == {
        "modo": "pistas", "ajuste": "i640_d0p5_m3", "correctos": 4}
    assert modo_elevado.best_mode(identity(4), {"i640_d0p5_m3": identity(4)})["modo"] == "rostros"
    assert modo_elevado.best_mode(identity(5), {"i640_d0p5_m3": identity(4)})["modo"] == "rostros"
    tied = {"i640_d0p5_m3": identity(6), "i1280_d0p2_m2": identity(6), "i640_d0p1_m2": None}
    assert modo_elevado.best_mode(identity(1), tied)["ajuste"] == "i1280_d0p2_m2"
    assert modo_elevado.best_mode(None, {"i640_d0p5_m3": identity(4)}) is None
    assert modo_elevado.best_text(None) == "n/d"
    assert modo_elevado.best_text({"modo": "rostros", "ajuste": None, "correctos": 1}) == "rostros"


def tracking(cubiertas, pureza=1.0):
    return {"personas_etiquetadas": 74, "pistas": cubiertas + 1, "personas_cubiertas": cubiertas,
            "pistas_duplicadas": 0, "pistas_mezcladas": 1, "pistas_sin_asignar": 0,
            "pureza": pureza}


SCORES = {
    "i640_d0p5_m3": {
        "MOT20-01": {"seguimiento": tracking(12, 0.9), "identidad": identity(4), "con_rostros": True},
        "MOT20-02": {"seguimiento": tracking(20, 0.7), "identidad": None, "con_rostros": False},
        "Camera3": {"seguimiento": None, "identidad": identity(2), "con_rostros": True},
    },
    "i1280_d0p2_m2": {
        "MOT20-01": {"seguimiento": tracking(30, 0.8), "identidad": identity(3), "con_rostros": True},
    },
}


def test_tracking_rows_with_totals():
    times = {"i640_d0p5_m3": {"MOT20-01": {"seguimiento_s": 10.0},
                              "MOT20-02": {"seguimiento_s": 5.5}}}
    rows = modo_elevado.tracking_rows(SCORES, times)
    assert rows == [
        ["i1280_d0p2_m2", "MOT20-01", "74", "31", "30", "0", "1", "0", "80.0%", "n/d"],
        ["i1280_d0p2_m2", "total", "74", "31", "30", "0", "1", "0", "80.0%", "n/d"],
        ["i640_d0p5_m3", "MOT20-01", "74", "13", "12", "0", "1", "0", "90.0%", "10.0 s"],
        ["i640_d0p5_m3", "MOT20-02", "74", "21", "20", "0", "1", "0", "70.0%", "5.5 s"],
        ["i640_d0p5_m3", "total", "148", "34", "32", "0", "2", "0", "80.0%", "15.5 s"],
    ]


def test_identity_rows_mark_best_mode():
    faces = {"MOT20-01": identity(3, pureza=0.842)}
    best = {stem: modo_elevado.best_mode(faces.get(stem),
                                         modo_elevado.tracks_identities(SCORES, stem))
            for stem in modo_elevado.compared_videos(SCORES)}
    rows = modo_elevado.identity_rows(faces, SCORES, best)
    faces_rule, tracks_rule = modo_elevado.FACES_RULE, pistas.VOTE_RULE
    assert tracks_rule == "pistas: votos de las cajas de persona"
    assert rows == [
        ["Camera3", "rostros", "-"] + ["n/d"] * 10 + ["", faces_rule],
        ["Camera3", "pistas", "i640_d0p5_m3", "10", "2", "2", "0", "0", "0", "100.0%", "2", "2",
         "0", "", tracks_rule],
        ["MOT20-01", "rostros", "-", "10", "3", "3", "0", "0", "0", "84.2%", "3", "-", "-", "",
         faces_rule],
        ["MOT20-01", "pistas", "i1280_d0p2_m2", "10", "3", "3", "0", "0", "0", "100.0%", "3", "3",
         "0", "", tracks_rule],
        ["MOT20-01", "pistas", "i640_d0p5_m3", "10", "4", "4", "0", "0", "0", "100.0%", "4", "4",
         "0", "sí", tracks_rule],
    ]


def test_footage_rows_and_table_format():
    stats = {"MOT20-01": {"altura_mediana_px": 175.4, "altura_relativa": 0.1624,
                          "personas_por_revision_mediana": 48, "fraccion_con_personas": 1.0,
                          "ancho": 1920, "alto": 1080}}
    best = {"MOT20-01": {"modo": "pistas", "ajuste": "i640_d0p5_m3", "correctos": 4}}
    rows = modo_elevado.footage_rows(stats, best)
    assert rows == [["MOT20-01", "175", "16.2%", "48", "100.0%", "1920x1080",
                     "pistas i640_d0p5_m3"]]
    text = modo_elevado.format_table(["video", "x"], [["MOT20-01", "1"]])
    assert text.splitlines() == ["video     x", "--------  -", "MOT20-01  1"]


def test_spanish_identity_adds_correctos():
    result = modo_elevado.spanish_identity({"labelled": 14, "identities": 9, "found": 7,
                                            "duplicates": 2, "mixed": 1, "without_person": 0,
                                            "purity": 0.99, "faces_total": 1,
                                            "faces_matched": 1})
    assert result == {"personas_etiquetadas": 14, "identidades": 9, "encontradas": 7,
                      "duplicados": 2, "mezcladas": 1, "sin_asignar": 0, "pureza": 0.99,
                      "correctos": 4}


MOT_GT = "1,1,100,200,40,100,1,1,1\n1,2,500,200,40,100,1,1,1\n"
TRACK_MANIFEST = {"video": "m.mp4", "fps": 25.0, "detect_every": 12,
                  "tracks": {"1": {"candidate_000.jpg": 0}, "2": {"candidate_000.jpg": 0}},
                  "boxes": {"1": {"candidate_000.jpg": [100, 200, 140, 300]},
                            "2": {"candidate_000.jpg": [500, 200, 540, 300]}}}
TRACK_RESULTS = {"tracks_checked": 2, "faces": [
    {"track_id": "1", "candidate": "candidate_000.jpg", "face_box": [15, 2, 25, 14],
     "frame_idx": 0, "score": 0.99},
    {"track_id": "2", "candidate": "candidate_000.jpg", "face_box": [15, 2, 25, 14],
     "frame_idx": 0, "score": 0.99}]}


def write_run(directory, manifest, results, summary=True):
    os.makedirs(os.path.join(directory, "candidates"))
    runlog.write_json(os.path.join(directory, "candidates", "manifest.json"), manifest)
    runlog.write_json(os.path.join(directory, "candidates", "stage1_summary.json"), {})
    if results is not None:
        os.makedirs(os.path.join(directory, "faces"))
        runlog.write_json(os.path.join(directory, "faces", "results.json"), results)
        if summary:
            runlog.write_json(os.path.join(directory, "faces", "stage2_summary.json"), {})


def test_tracks_identity_uses_track_votes_without_mot_gt():
    gt = [(0.0, [([115, 202, 125, 214], "C3-01")])]
    identities = mot_identities()
    result = modo_elevado.tracks_identity("Camera3", TRACK_MANIFEST, identities, None,
                                          lambda stem: (bench_compuerta.KIND_CCTV, gt, None))
    assert (result["identidades"], result["encontradas"], result["sin_asignar"]) == (2, 1, 1)
    assert (result["correctos"], result["cara_correcta"]) == (1, 1)
    assert result["regla"] == "pistas: votos de las cajas de persona"
    chokepoint_gt = {3: [("0007", (520.0, 210.0))]}
    result = modo_elevado.tracks_identity(
        "P1E_S1_C1", TRACK_MANIFEST, identities, None,
        lambda stem: (bench_compuerta.KIND_CHOKEPOINT, chokepoint_gt, [3]))
    assert (result["encontradas"], result["cara_correcta"]) == (1, 1)
    assert modo_elevado.tracks_identity("otro", TRACK_MANIFEST, identities, None,
                                        lambda stem: (None, None, None)) is None


def mot_identities():
    return mot.tracks_to_identities(TRACK_MANIFEST, TRACK_RESULTS)


def test_puntuar_end_to_end(tmp_path, capsys):
    corridas = tmp_path / "corridas"
    write_run(str(corridas / "i640_d0p5_m3" / "m"), TRACK_MANIFEST, TRACK_RESULTS)
    write_run(str(corridas / "i1280_d0p5_m3" / "m"), TRACK_MANIFEST, None)
    runlog.write_json(str(corridas / "tiempos.json"),
                      {"i640_d0p5_m3": {"m": {"seguimiento_s": 3.0}}})
    (tmp_path / "gt" / "m").mkdir(parents=True)
    (tmp_path / "gt" / "m" / "gt.txt").write_text(MOT_GT)
    base = tmp_path / "base"
    (base / "m").mkdir(parents=True)
    runlog.write_json(str(base / "m" / "identities.json"), {"identities": [
        {"id": 1, "faces": [{"frame": 0, "box": [115, 202, 125, 214], "score": 0.9},
                            {"frame": 0, "box": [515, 202, 525, 214], "score": 0.9}]}]})
    runlog.write_json(str(base / "results.json"), {"videos": {
        "m": {"detections_file": "m/detections.json", "identity": None, "kind": None},
        "roto": {"error": "x"}}})
    stats = tmp_path / "estadisticas.json"
    runlog.write_json(str(stats), {"m": {"altura_mediana_px": 100, "altura_relativa": 0.1,
                                         "personas_por_revision_mediana": 2,
                                         "fraccion_con_personas": 1.0, "ancho": 1920,
                                         "alto": 1080}})
    out = tmp_path / "puntuar.json"
    code = modo_elevado.command_puntuar(args(corridas=str(corridas), base=[str(base / "results.json")],
                                             mot_gt=str(tmp_path / "gt"), estadisticas=str(stats),
                                             out=str(out)))
    assert code == 0
    document = json.loads(out.read_text())
    tracks = document["ajustes"]["i640_d0p5_m3"]["videos"]["m"]
    assert tracks["identidad"]["encontradas"] == 2
    assert tracks["identidad"]["correctos"] == 2
    assert tracks["identidad"]["cara_correcta"] == 2
    assert tracks["identidad"]["regla"] == "pistas: votos de las cajas de persona"
    assert tracks["seguimiento"]["personas_cubiertas"] == 2
    assert document["ajustes"]["i1280_d0p5_m3"]["videos"]["m"]["identidad"] is None
    assert document["rostros"]["m"]["mezcladas"] == 1
    assert document["rostros"]["m"]["correctos"] == 0
    assert document["mejor_modo"]["m"] == {"modo": "pistas", "ajuste": "i640_d0p5_m3",
                                           "correctos": 2}
    assert document["totales_seguimiento"]["i640_d0p5_m3"]["videos"] == 1
    assert "más correctos" in document["nota"]
    assert document["reglas"]["pistas"] == "pistas: votos de las cajas de persona"
    assert document["omitidos"] == [
        {"modo": "pistas", "origen": "i1280_d0p5_m3", "video": "m",
         "motivo": "sin faces/results.json"},
        {"modo": "rostros", "origen": str(base / "results.json"), "video": "roto",
         "motivo": "la base marcó un error: x"}]
    printed = capsys.readouterr().out
    assert "pistas i640_d0p5_m3" in printed
    assert "3.0 s" in printed
    assert "pistas: votos de las cajas de persona" in printed
    assert "cara correcta" in printed
    assert "Omitidos" in printed


def test_puntuar_lists_every_omitted_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TESIS_CHOKEPOINT_DIR", str(tmp_path / "chokepoint"))
    corridas = tmp_path / "corridas"
    setting = corridas / "i640_d0p5_m3"
    (setting / "sin_etapa1").mkdir(parents=True)
    runlog.write_json(str(setting / "sin_etapa1" / "etapas.json"), {"stage1": "timeout"})
    write_run(str(setting / "m"), TRACK_MANIFEST, TRACK_RESULTS)
    runlog.write_json(str(setting / "m" / "etapas.json"), {"stage1": 0, "stage2": 1})
    write_run(str(setting / "traza"), TRACK_MANIFEST, TRACK_RESULTS)
    (setting / "traza" / "stage2.stderr.log").write_text(
        "Traceback (most recent call last):\n  File x\nValueError: roto\n")
    write_run(str(setting / "sin_resumen"), TRACK_MANIFEST, TRACK_RESULTS, summary=False)
    write_run(str(setting / "video_sin_verdad_xyz"), TRACK_MANIFEST, TRACK_RESULTS)
    for stem in ("m", "traza", "sin_resumen"):
        (tmp_path / "gt" / stem).mkdir(parents=True)
        (tmp_path / "gt" / stem / "gt.txt").write_text(MOT_GT)
    base = tmp_path / "base"
    base.mkdir()
    runlog.write_json(str(base / "results.json"), {"videos": {
        "roto": {"error": "código de salida 1"},
        "s5": {"identity": None, "kind": None},
        "cam": {"identity": None, "kind": "cctv"}}})
    stats = tmp_path / "estadisticas.json"
    runlog.write_json(str(stats), {})
    out = tmp_path / "puntuar.json"
    code = modo_elevado.command_puntuar(args(corridas=str(corridas),
                                             base=[str(base / "results.json")],
                                             mot_gt=str(tmp_path / "gt"),
                                             estadisticas=str(stats), out=str(out)))
    assert code == 0
    document = json.loads(out.read_text())
    reasons = {(item["origen"], item["video"], item["motivo"]) for item in document["omitidos"]}
    origin = str(base / "results.json")
    assert reasons == {
        ("i640_d0p5_m3", "m", "la etapa de rostros terminó con código 1"),
        ("i640_d0p5_m3", "sin_etapa1", "la etapa de seguimiento terminó con código timeout"),
        ("i640_d0p5_m3", "sin_etapa1", "sin etapa 1 completa (falta candidates/manifest.json o "
                                       "candidates/stage1_summary.json)"),
        ("i640_d0p5_m3", "sin_resumen",
         "la etapa de rostros no terminó (falta faces/stage2_summary.json)"),
        ("i640_d0p5_m3", "traza", "la etapa de rostros dejó un error (Traceback) en "
                                  "stage2.stderr.log"),
        ("i640_d0p5_m3", "video_sin_verdad_xyz",
         "tipo de verdad de terreno desconocido: no hay gt.txt de MOT, ni verdad de terreno de "
         "ChokePoint o CCTV para este video"),
        (origin, "roto", "la base marcó un error: código de salida 1"),
        (origin, "s5", "tipo de verdad de terreno desconocido: no hay gt.txt de MOT y la base no "
                       "tiene métricas de identidad"),
        (origin, "cam", "la base no tiene métricas de identidad"),
    }
    videos = document["ajustes"]["i640_d0p5_m3"]["videos"]
    assert "sin_etapa1" not in videos
    assert videos["m"]["identidad"] is None
    assert videos["m"]["seguimiento"]["personas_cubiertas"] == 2
    printed = capsys.readouterr().out
    omitted_section = printed[printed.index("Omitidos"):]
    assert "sin_etapa1" in omitted_section
    assert "la base marcó un error" in omitted_section
