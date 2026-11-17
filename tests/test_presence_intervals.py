import json
import os
import random
import subprocess
import sys

import pytest

PIPELINE_DIR = os.path.abspath(os.environ.get(
    "TESIS_PIPELINE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "tesis_face_recognition_pipeline")))
sys.path.insert(0, PIPELINE_DIR)

import presence_intervals as pi

FPS = 10.0


def intervals_of(checks, total=1000, rewind=1.0, after=2.0):
    return pi.active_intervals(checks, FPS, total, rewind, after)


def test_detection_opens_an_interval_with_rewind_and_exit():
    assert intervals_of([(100, True)]) == [(90, 120)]


def test_checks_without_a_person_open_nothing():
    assert intervals_of([(0, False), (5, False)]) == []
    assert intervals_of([]) == []


def test_detections_close_together_extend_one_interval():
    checks = [(100, True), (105, False), (110, True), (130, True)]
    assert intervals_of(checks) == [(90, 150)]


def test_intervals_that_touch_merge():
    assert intervals_of([(100, True), (131, True)]) == [(90, 151)]
    assert intervals_of([(100, True), (132, True)]) == [(90, 120), (122, 152)]


def test_intervals_with_a_gap_stay_apart():
    assert intervals_of([(100, True), (200, True)]) == [(90, 120), (190, 220)]


def test_start_clamps_at_zero():
    assert intervals_of([(0, True)]) == [(0, 20)]
    assert intervals_of([(4, True)]) == [(0, 24)]


def test_end_clamps_at_the_last_frame():
    assert intervals_of([(990, True)], total=1000) == [(980, 999)]
    assert intervals_of([(999, True)], total=1000) == [(989, 999)]


def test_a_single_check_gives_one_interval():
    assert intervals_of([(500, True)], total=501) == [(490, 500)]


def test_unknown_total_frames_clamps_only_at_zero():
    assert intervals_of([(3, True), (400, True)], total=0) == [(0, 23), (390, 420)]
    assert intervals_of([(400, True)], total=-1) == [(390, 420)]


def test_rewind_and_exit_scale_with_the_frame_rate():
    assert pi.active_intervals([(300, True)], 30.0, 10000, 1.0, 2.0) == [(270, 360)]


def test_frame_in_intervals():
    intervals = [(10, 20), (30, 40)]
    assert [pi.frame_in_intervals(f, intervals)
            for f in (9, 10, 20, 21, 29, 30, 40, 41)] == [False, True, True, False, False, True, True, False]
    assert not pi.frame_in_intervals(5, [])


def test_frame_in_intervals_accepts_lists_as_read_from_json():
    intervals = [[10, 20], [30, 40]]
    assert pi.frame_in_intervals(15, intervals)
    assert pi.frame_in_intervals(30, intervals)
    assert not pi.frame_in_intervals(25, intervals)
    assert not pi.frame_in_intervals(41, intervals)
    assert not pi.frame_in_intervals(0, intervals)


def test_next_allowed_stays_on_the_grid_anchored_at_zero():
    intervals = [(10, 20), (35, 50)]
    assert pi.next_allowed(0, 3, intervals) == 12
    assert pi.next_allowed(12, 3, intervals) == 12
    assert pi.next_allowed(13, 3, intervals) == 15
    assert pi.next_allowed(19, 3, intervals) == 36
    assert pi.next_allowed(21, 3, intervals) == 36
    assert pi.next_allowed(49, 3, intervals) is None
    assert pi.next_allowed(0, 3, []) is None


def test_next_allowed_skips_an_interval_that_holds_no_grid_frame():
    assert pi.next_allowed(0, 10, [(11, 19), (25, 35)]) == 30


def test_next_allowed_with_step_one_is_the_first_frame_of_the_interval():
    assert pi.next_allowed(0, 1, [(7, 9)]) == 7


def test_format_time():
    assert pi.format_time(0) == "00:00:00.0"
    assert pi.format_time(62.3) == "00:01:02.3"
    assert pi.format_time(3725.46) == "01:02:05.4"


def test_switch_log_reports_both_switches_with_their_triggers():
    checks = [(0, False), (100, True), (110, True), (200, False)]
    intervals = intervals_of(checks)
    assert pi.switch_log(intervals, checks, FPS, 1000) == [
        {"evento": "activo", "frame": 90, "tiempo": "00:00:09.0",
         "disparo_frame": 100, "disparo_tiempo": "00:00:10.0"},
        {"evento": "inactivo", "frame": 131, "tiempo": "00:00:13.1",
         "disparo_frame": 110, "disparo_tiempo": "00:00:11.0"},
    ]


def test_switch_log_omits_the_final_inactive_event_at_the_last_frame():
    checks = [(990, True)]
    events = pi.switch_log(intervals_of(checks), checks, FPS, 1000)
    assert [e["evento"] for e in events] == ["activo"]


def test_switch_log_keeps_the_inactive_event_when_the_total_is_unknown():
    checks = [(100, True)]
    events = pi.switch_log(intervals_of(checks, total=0), checks, FPS)
    assert [e["evento"] for e in events] == ["activo", "inactivo"]


def test_switch_log_orders_events_by_interval():
    checks = [(100, True), (300, True)]
    events = pi.switch_log(intervals_of(checks), checks, FPS, 1000)
    assert [(e["evento"], e["frame"]) for e in events] == [
        ("activo", 90), ("inactivo", 121), ("activo", 290), ("inactivo", 321)]


def linear_next_allowed(frame_idx, step, intervals):
    candidate = -(-max(0, frame_idx) // step) * step
    for start, end in intervals:
        if end < candidate:
            continue
        if candidate < start:
            candidate = -(-start // step) * step
        if candidate <= end:
            return candidate
    return None


def test_next_allowed_with_many_intervals_matches_a_full_scan():
    rng = random.Random(7)
    checks = [(frame, rng.random() < 0.08) for frame in range(0, 6000, 5)]
    intervals = pi.active_intervals(checks, 10.0, 6000, 0.3, 0.4)
    short = [(start, start + rng.randrange(3)) for start in range(6100, 9000, 17)]
    intervals = intervals + short
    assert len(intervals) > 150
    frames = list(range(-3, 300)) + rng.sample(range(300, 9100), 800)
    for step in (1, 2, 3, 7, 10):
        for frame_idx in frames:
            assert pi.next_allowed(frame_idx, step, intervals) == \
                linear_next_allowed(frame_idx, step, intervals)


def write_presence(tmp_path, document, name="presence.json"):
    path = tmp_path / name
    path.write_text(document if isinstance(document, str) else json.dumps(document))
    return str(path)


PRESENCE_DOC = {"fps": 25.0, "intervals": [[0, 9], [20, 29]], "active_fraction": 0.2,
                "total_frames": 100}


def test_load_presence_reads_intervals_fraction_and_frame_count(tmp_path):
    presence = pi.load_presence(write_presence(tmp_path, PRESENCE_DOC), 25.0)

    assert presence == {"intervals": [(0, 9), (20, 29)], "active_fraction": 0.2,
                        "total_frames": 100}


@pytest.mark.parametrize("document", ["{no es json", json.dumps({"fps": 25.0}),
                                      json.dumps({**PRESENCE_DOC, "intervals": [[0]]}),
                                      json.dumps({**PRESENCE_DOC, "total_frames": "muchos"})])
def test_malformed_presence_file_is_refused_with_a_clear_message(tmp_path, document):
    path = write_presence(tmp_path, document)

    with pytest.raises(pi.PresenceFileError, match="No se pudo leer el archivo de presencia"):
        pi.load_presence(path, 25.0)


def test_missing_presence_file_is_refused(tmp_path):
    with pytest.raises(pi.PresenceFileError, match="No se pudo leer el archivo de presencia"):
        pi.load_presence(str(tmp_path / "no_existe.json"), 25.0)


def test_presence_file_of_a_video_with_another_frame_rate_is_refused(tmp_path):
    path = write_presence(tmp_path, PRESENCE_DOC)

    with pytest.raises(pi.PresenceFileError,
                       match="es de un video con 25.0 fotogramas por segundo y este tiene 30.0"):
        pi.load_presence(path, 30.0)


def test_frame_rates_within_a_thousandth_are_the_same_video():
    assert pi.fps_mismatch("p.json", 15.0001, 15.0) is None
    assert pi.fps_mismatch("p.json", 15.01, 15.0) is not None


def test_gating_with_adaptive_sampling_is_refused():
    assert "FACE_ADAPTIVE=1" in pi.gating_refusal(True, True)
    assert pi.gating_refusal(True, False) is None
    assert pi.gating_refusal(False, True) is None


def test_gate_settings_defaults(monkeypatch):
    for name in ("GATE_CHECKS_PER_S", "GATE_REWIND_S", "GATE_EXIT_S", "GATE_CONF", "GATE_IMGSZ"):
        monkeypatch.delenv(name, raising=False)

    assert pi.read_gate_settings() == {"checks_per_s": 2.0, "rewind_s": 1.0, "exit_s": 2.0,
                                       "conf": 0.25, "imgsz": 640}


def test_gate_settings_refuse_a_confidence_above_one(monkeypatch):
    monkeypatch.setenv("GATE_CONF", "1.5")

    with pytest.raises(SystemExit, match="GATE_CONF debe estar entre 0 y 1"):
        pi.read_gate_settings()


def test_presence_intervals_imports_no_framework():
    code = ("import sys, presence_intervals; "
            "print([m for m in ('numpy', 'cv2', 'torch', 'tensorflow', 'ultralytics') "
            "if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], cwd=PIPELINE_DIR, capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "[]"
