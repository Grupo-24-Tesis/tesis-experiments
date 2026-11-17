import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import pytest

from bench import bench_compuerta, barrido

PIPELINE_DIR = os.path.abspath(os.environ.get(
    "TESIS_PIPELINE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "tesis_face_recognition_pipeline")))

FPS = 10.0
TOTAL = 100
FRAMES = list(range(0, 100, 5))
BOX_P1 = [5, 5, 15, 15]
BOX_P2 = [45, 45, 55, 55]
GT = {30: [("p1", (10.0, 10.0))], 35: [("p1", (10.0, 10.0))], 40: [("p1", (10.0, 10.0))],
      80: [("p2", (50.0, 50.0))]}


@pytest.fixture(autouse=True)
def pipeline(monkeypatch):
    monkeypatch.setenv("TESIS_PIPELINE_DIR", PIPELINE_DIR)


def detection(frame, box):
    return {"frame": frame, "time_s": frame / FPS, "box": box, "score": 0.99, "px": 10}


DETECTIONS = [detection(30, BOX_P1), detection(35, BOX_P1), detection(40, BOX_P1),
              detection(80, BOX_P2)]


def make_case(base_wall=100.0):
    case = {"name": "v", "kind": bench_compuerta.KIND_CHOKEPOINT, "gt": GT, "frame_map": None,
            "fps": FPS, "frames": list(FRAMES), "detections": DETECTIONS,
            "base_wall_s": base_wall}
    case["base_detection"] = barrido.detection_metrics(case, DETECTIONS, FRAMES)
    case["person_frames"] = barrido.person_true_positive_frames(DETECTIONS, FRAMES, GT, None)
    return case


def make_presence(with_confidences=True):
    checks_frames = list(range(0, TOTAL, 10))
    values = {30: 0.6, 40: 0.2, 80: 0.3}
    document = {"fps": FPS, "total_frames": TOTAL, "wall_s": 3.0,
                "checks": [[f, values.get(f, 0.0) >= 0.05] for f in checks_frames]}
    if with_confidences:
        document["confidences"] = [[f, values.get(f, 0.0)] for f in checks_frames]
    return document


def test_retained_frames_are_the_grid_frames_inside_the_intervals():
    kept = barrido.retained_frames(FRAMES, [(20, 40), (70, 71)])
    assert kept == [20, 25, 30, 35, 40, 70]


def test_retained_frames_with_no_intervals_is_empty():
    assert barrido.retained_frames(FRAMES, []) == []


def test_confidence_threshold_selects_the_checks():
    presence = make_presence()
    low, _ = barrido.simulate_intervals(presence, 0.1, 1.0, 1.0)
    mid, _ = barrido.simulate_intervals(presence, 0.25, 1.0, 1.0)
    high, _ = barrido.simulate_intervals(presence, 0.5, 1.0, 1.0)
    assert low == [(20, 50), (70, 90)]
    assert mid == [(20, 40), (70, 90)]
    assert high == [(20, 40)]


def test_missing_confidences_use_the_found_checks_and_are_flagged():
    presence = make_presence(with_confidences=False)
    intervals, missing = barrido.simulate_intervals(presence, 0.25, 1.0, 1.0)
    assert missing is True
    assert intervals == [(20, 50), (70, 90)]


def test_score_video_with_every_person_kept():
    result = barrido.score_video(make_case(), make_presence(), 0.25, 1.0, 1.0)
    assert result["lost"] == []
    assert result["persons_lost"] == []
    assert result["recall_drop_points"] == 0.0
    assert result["detection"]["tp"] == 4


def test_a_person_with_every_detection_lost_is_flagged_and_the_loss_is_located():
    result = barrido.score_video(make_case(), make_presence(), 0.5, 1.0, 1.0)
    assert result["persons_lost"] == ["p2"]
    assert [lost["frame"] for lost in result["lost"]] == [80]
    assert result["lost"][0]["distance_s"] == pytest.approx(3.9)
    assert result["recall_drop_points"] == pytest.approx(25.0)


def test_a_person_with_some_detections_kept_is_not_flagged():
    presence = make_presence()
    presence["confidences"] = [[f, 0.6 if f == 30 else 0.0] for f in range(0, TOTAL, 10)]
    result = barrido.score_video(make_case(), presence, 0.25, 0.0, 0.0)
    assert [lost["frame"] for lost in result["lost"]] == [35, 40, 80]
    assert "p1" not in result["persons_lost"]
    assert result["persons_lost"] == ["p2"]


def test_estimated_face_time_scales_with_the_analysed_frames():
    result = barrido.score_video(make_case(base_wall=100.0), make_presence(), 0.5, 1.0, 1.0)
    assert result["analysed_frames"] == 5
    assert result["analysed_fraction"] == pytest.approx(0.25)
    assert result["estimated_face_s"] == pytest.approx(25.0)


def summary(change=0.0, lost=0, worst=0.0, total=10.0):
    return {"recall_change_points": change, "persons_lost": lost, "max_video_drop_points": worst,
            "estimated_total_s": total}


def setting(exit_s, rewind_s, **kwargs):
    return {"exit_s": exit_s, "rewind_s": rewind_s, "summary": summary(**kwargs)}


def test_recommendation_picks_the_cheapest_setting_that_is_safe():
    cheap_unsafe = setting(1, 1, total=5.0, lost=1)
    cheap_drop = setting(1, 1, total=6.0, change=-0.6)
    cheap_video = setting(1, 1, total=7.0, worst=1.5)
    safe = setting(1, 1, total=8.0, change=-0.5, worst=1.0)
    slower = setting(2, 1, total=9.0)
    assert barrido.recommend([cheap_unsafe, cheap_drop, cheap_video, slower, safe]) is safe


def test_recommendation_breaks_ties_with_higher_exit_then_rewind():
    a = setting(1, 2, total=8.0)
    b = setting(2, 1, total=8.0)
    c = setting(2, 2, total=8.0)
    assert barrido.recommend([a, b, c]) is c
    assert barrido.recommend([a, b]) is b


def test_recommendation_is_none_when_no_setting_is_safe():
    assert barrido.recommend([setting(1, 1, lost=2)]) is None


def test_summary_aggregates_accuracy_over_precision_videos_and_cost_over_cost_videos():
    case = make_case(base_wall=100.0)
    other = dict(make_case(base_wall=50.0), name="w")
    cases = {"v": case, "w": other}
    per_video = {"v": barrido.score_video(case, make_presence(), 0.5, 1.0, 1.0),
                 "w": barrido.score_video(other, make_presence(), 0.25, 1.0, 1.0)}
    result = barrido.summarise_setting(per_video, cases, {"v", "w"}, {"w"}, {"v": 1.0, "w": 2.0})
    assert result["recall_change_points"] == pytest.approx(-12.5)
    assert result["videos_with_drop"] == 1
    assert result["persons_lost"] == 1
    assert result["videos_cost"] == ["w"]
    assert result["gate_s"] == pytest.approx(2.0)
    assert result["estimated_face_s"] == pytest.approx(50.0 * 10 / 20)
    assert result["estimated_total_s"] == pytest.approx(27.0)
    assert result["base_total_s"] == pytest.approx(50.0)
    assert result["factor"] == pytest.approx(0.54)


def test_gate_runs_every_pair_skips_existing_files_and_records_times(tmp_path, monkeypatch):
    calls = []

    def fake_run(name, command, env, log_path):
        calls.append((name, command, env))
        out = command[command.index("--out") + 1]
        with open(out, "w") as f:
            f.write("{}")
        return 1.5, 0

    monkeypatch.setattr(bench_compuerta, "run_logged", fake_run)
    out = tmp_path / "gates"
    existing = out / "c2_i320" / "clip" / "presence.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("{}")
    args = barrido.build_parser().parse_args(
        ["gate", "--out", str(out), "--checks-per-s", "2,4", "--imgsz", "320,640",
         str(tmp_path / "clip.mp4")])

    assert args.handler(args) == 0

    assert [c[0] for c in calls] == ["c2_i640/clip", "c4_i320/clip", "c4_i640/clip"]
    assert calls[0][1][0].endswith(os.path.join("venvs", "track", "bin", "python"))
    assert calls[0][2]["GATE_CONF"] == "0.05"
    assert calls[0][2]["GATE_CHECKS_PER_S"] == "2"
    assert calls[0][2]["GATE_IMGSZ"] == "640"
    times = json.loads((out / "gate_times.json").read_text())
    assert times == {"c2_i640": {"clip": 1.5}, "c4_i320": {"clip": 1.5}, "c4_i640": {"clip": 1.5}}

    calls.clear()
    assert args.handler(args) == 0
    assert calls == []


XML = """<sequence name="V">
<frame number="30"><person id="p1"><leftEye x="9" y="10"/><rightEye x="11" y="10"/></person></frame>
<frame number="35"><person id="p1"><leftEye x="9" y="10"/><rightEye x="11" y="10"/></person></frame>
<frame number="40"><person id="p1"><leftEye x="9" y="10"/><rightEye x="11" y="10"/></person></frame>
<frame number="80"><person id="p2"><leftEye x="49" y="50"/><rightEye x="51" y="50"/></person></frame>
</sequence>"""


def test_puntuar_end_to_end_writes_json_and_marks_the_recommended_setting(tmp_path, monkeypatch,
                                                                          capsys):
    chokepoint = tmp_path / "cp"
    (chokepoint / "groundtruth").mkdir(parents=True)
    (chokepoint / "videos").mkdir()
    (chokepoint / "groundtruth" / "V.xml").write_text(XML)
    (chokepoint / "videos" / "V.frames.json").write_text(json.dumps(list(range(TOTAL))))
    monkeypatch.setenv("TESIS_CHOKEPOINT_DIR", str(chokepoint))
    base = tmp_path / "base"
    (base / "V").mkdir(parents=True)
    (base / "V" / "detections.json").write_text(json.dumps(
        {"fps": FPS, "step": 5, "frames_analysed": FRAMES, "detections": DETECTIONS}))
    (base / "results.json").write_text(json.dumps({"videos": {"V": {
        "wall_s": 100.0, "detections_file": "V/detections.json"}}}))
    gates = tmp_path / "gates"
    (gates / "c4_i640" / "V").mkdir(parents=True)
    (gates / "c4_i640" / "V" / "presence.json").write_text(json.dumps(make_presence()))
    (gates / "gate_times.json").write_text(json.dumps({"c4_i640": {"V": 4.0}}))
    out = tmp_path / "salida.json"
    args = barrido.build_parser().parse_args(
        ["puntuar", "--baseline", str(base / "results.json"), "--gates", str(gates),
         "--conf", "0.1,0.5", "--exit", "1", "--rewind", "1", "--out", str(out)])

    assert args.handler(args) == 0

    text = capsys.readouterr().out
    assert text.count("RECOMENDADO") == 1
    assert "estimación" in text
    document = json.loads(out.read_text())
    assert [s["conf"] for s in document["settings"]] == [0.1, 0.5]
    assert document["recommended"]["conf"] == 0.1
    assert document["settings"][1]["summary"]["persons_lost"] == 1
    assert document["settings"][0]["summary"]["gate_s"] == pytest.approx(4.0)
