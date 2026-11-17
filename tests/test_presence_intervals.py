import os
import sys

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
    assert pi.frame_in_intervals(15, [(10, 20)])


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
