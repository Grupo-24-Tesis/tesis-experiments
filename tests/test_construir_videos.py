import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import pytest

from bench import construir_videos as cv
from bench import metrics

FPS = 30


def test_source_labels_uses_map_and_marks_missing_frames():
    gt = {10: [("a", (1.0, 1.0))], 11: [], 12: [("a", (1.0, 1.0)), ("b", (2.0, 2.0))]}
    labels = cv.source_labels(gt, [10, 11, 12, 99], 5)
    assert labels == [["a"], [], ["a", "b"], None, None]


def test_presence_runs_merges_small_gaps():
    labels = [["a"], ["a"], [], ["a"], [], [], [], [], ["b"], ["b"], None]
    assert cv.presence_runs(labels, merge_gap=1) == [(0, 3), (8, 9)]
    assert cv.presence_runs(labels, merge_gap=0) == [(0, 1), (3, 3), (8, 9)]
    assert cv.presence_runs([[], None], merge_gap=2) == []


def test_empty_stretches_respect_margin_and_person_checks():
    labels = [[]] * 100
    labels[50] = ["a"]
    checks = [[i, False] for i in range(0, 100, 5)]
    found = cv.empty_stretches(labels, checks, step=5, margin=10, min_len=5)
    assert found == [(0, 39), (61, 99)]
    checks[16] = [80, True]
    found = cv.empty_stretches(labels, [[f, p] for f, p in checks], step=5, margin=10, min_len=5)
    assert (61, 75) in found and (85, 99) in found


def test_empty_stretches_block_frames_without_ground_truth_and_short_runs():
    labels = [[]] * 30 + [None] * 5 + [[]] * 3
    checks = [[i, False] for i in range(0, 38, 5)]
    assert cv.empty_stretches(labels, checks, step=5, margin=0, min_len=10) == [(0, 29)]
    assert cv.empty_stretches(labels, checks, step=5, margin=0, min_len=2) == [(0, 29), (35, 37)]


def test_empty_stretches_person_check_blocks_neighbouring_frames():
    labels = [[]] * 20
    checks = [[0, False], [5, True], [10, False], [15, False]]
    assert cv.empty_stretches(labels, checks, step=5, margin=0, min_len=1) == [(0, 0), (10, 19)]


def test_pingpong_indices_go_forward_then_backward_without_repeating_turns():
    assert cv.pingpong_indices(10, 13, 12) == [10, 11, 12, 13, 12, 11, 10, 11, 12, 13, 12, 11]
    assert cv.pingpong_indices(5, 5, 4) == [5, 5, 5, 5]
    assert cv.pingpong_indices(0, 3, 0) == []
    indices = cv.pingpong_indices(0, 9, 500)
    assert len(indices) == 500
    assert all(abs(b - a) == 1 for a, b in zip(indices, indices[1:]))


def test_first_appearances_and_densest_arrivals():
    labels = [[] for _ in range(100)]
    for frame in range(10, 20):
        labels[frame] = ["a"]
    for frame in range(30, 40):
        labels[frame] = ["a", "b"]
    for frame in range(35, 45):
        labels[frame] = ["a", "b", "c"]
    for frame in range(90, 100):
        labels[frame] = ["d"]
    first = cv.first_appearances(labels)
    assert first == {"a": 10, "b": 30, "c": 35, "d": 90}
    assert cv.densest_arrivals(first, window=30) == (10, 3)
    assert cv.densest_arrivals(first, window=6) == (30, 2)
    assert cv.densest_arrivals({}, window=5) == (0, 0)


def test_distribute_empty_sums_and_bounds():
    frames = cv.distribute_empty(total=2000, gaps=4, static_index=1, static_frames=300,
                                 weights=[1, 1, 1, 1], min_frames=100, max_frames=800)
    assert sum(frames) == 2000
    assert frames[1] == 300
    assert all(100 <= f <= 800 for f in frames)
    with pytest.raises(ValueError):
        cv.distribute_empty(total=5000, gaps=4, static_index=1, static_frames=300,
                            weights=[1, 1, 1, 1], min_frames=100, max_frames=800)


SOURCES = {
    "S1": {"frames": 100, "map": list(range(1000, 1100))},
    "S5": {"frames": 40, "map": None},
}


def test_segment_expansion_and_map_entries():
    empty = {"kind": "empty", "source": "S1", "start": 2, "end": 4, "frames": 6,
             "mode": "pingpong"}
    clip = {"kind": "clip", "source": "S1", "start": 10, "frames": 3}
    unscored = {"kind": "clip", "source": "S5", "start": 0, "frames": 2}
    repeat = {"kind": "repeat", "source": "S1", "start": 7, "frames": 2}
    assert cv.expand_segment(empty) == [("S1", i) for i in [2, 3, 4, 3, 2, 3]]
    assert cv.expand_segment(clip) == [("S1", 10), ("S1", 11), ("S1", 12)]
    assert cv.segment_map_entries(empty, SOURCES) == [None] * 6
    assert cv.segment_map_entries(clip, SOURCES) == [["S1", 1010], ["S1", 1011], ["S1", 1012]]
    assert cv.segment_map_entries(unscored, SOURCES) == [["sin_gt", -1]] * 2
    assert cv.segment_map_entries(repeat, SOURCES) == [["S1", 1007]] * 2


def test_map_entry_beyond_source_map_is_unscored():
    clip = {"kind": "clip", "source": "S1", "start": 99, "frames": 2}
    assert cv.segment_map_entries(clip, SOURCES) == [["S1", 1099], ["sin_gt", -1]]


def test_build_plan_totals_and_timeline():
    segments = [
        {"kind": "empty", "source": "S1", "start": 0, "end": 9, "frames": 60, "mode": "pingpong"},
        {"kind": "clip", "source": "S1", "start": 10, "frames": 30},
        {"kind": "empty", "source": "S1", "start": 0, "end": 0, "frames": 30, "mode": "static"},
    ]
    plan = cv.build_plan("x", segments, SOURCES, FPS)
    assert plan["total_frames"] == 120
    assert plan["duration_s"] == pytest.approx(4.0)
    assert plan["people_frames"] == 30
    assert plan["people_share"] == pytest.approx(0.25)
    assert [s["start_s"] for s in plan["segments"]] == [0.0, 2.0, 3.0]
    assert [s["duration_s"] for s in plan["segments"]] == [2.0, 1.0, 1.0]


def test_plan_frame_map_aligns_with_frame_count():
    segments = [
        {"kind": "empty", "source": "S1", "start": 0, "end": 9, "frames": 60, "mode": "pingpong"},
        {"kind": "clip", "source": "S1", "start": 10, "frames": 30},
    ]
    document = cv.plan_frame_map(segments, SOURCES)
    assert document["empty_labelled"] is True
    assert len(document["frames"]) == cv.total_frames(segments) == 90
    assert document["frames"][59] is None
    assert document["frames"][60] == ["S1", 1010]
    assert metrics.is_composite_map(document)


LABELS = {
    "S1": [[]] * 20 + [["a"]] * 400 + [[]] * 100 + [["b"], ["c"]] * 40 + [[]] * 400 + [["d"]] * 500,
    "S2": [[]] * 50,
}
LABELS["S1"] = [list(x) for x in LABELS["S1"]]
STRETCH = ("S2", 0, 49)


def test_plan_borde_inicio_starts_one_second_into_a_run():
    segments = cv.plan_borde_inicio(LABELS["S1"], "S1", FPS, clip_frames=600, lead_frames=30)
    assert len(segments) == 1
    clip = segments[0]
    assert clip["kind"] == "clip" and clip["frames"] == 600
    run_start = max(s for s, e in cv.presence_runs(LABELS["S1"], 3) if s <= clip["start"])
    assert clip["start"] - run_start >= 30
    assert clip["start"] + clip["frames"] <= len(LABELS["S1"])


def test_plan_borde_final_ends_inside_a_run_after_empty_lead():
    segments = cv.plan_borde_final(LABELS["S1"], "S1", STRETCH, FPS, empty_frames=900,
                                   lead_frames=30)
    assert segments[0]["kind"] == "empty" and segments[0]["frames"] == 900
    clip = segments[1]
    last = clip["start"] + clip["frames"] - 1
    assert LABELS["S1"][last]
    assert LABELS["S1"][last + cv.AFTER_FRAMES // 2]
    assert not LABELS["S1"][clip["start"]]


def test_plan_borde_corto_window_is_inside_a_labelled_run():
    segments = cv.plan_borde_corto(LABELS["S1"], "S1", STRETCH, FPS, empty_frames=450,
                                   window_frames=18)
    kinds = [s["kind"] for s in segments]
    assert kinds == ["empty", "clip", "empty"]
    clip = segments[1]
    assert clip["frames"] == 18
    assert all(LABELS["S1"][i] for i in range(clip["start"], clip["start"] + 18))
    assert cv.total_frames(segments) == 450 + 18 + 450


def test_plan_borde_seguidos_picks_densest_source_window():
    labels = {"S1": LABELS["S1"], "S2": [[]] * 10 + [["x"]] * 5 + [["y"]] * 5 + [["z"]] * 5}
    segments = cv.plan_borde_seguidos(labels, STRETCH, FPS, window_frames=600,
                                      empty_frames=300, lead_frames=15)
    assert [s["kind"] for s in segments] == ["empty", "clip", "empty"]
    clip = segments[1]
    assert clip["source"] == "S1"
    assert clip["frames"] == 600
    assert clip["start"] + 600 <= len(labels["S1"])
    assert cv.total_frames(segments) == 300 + 600 + 300


def test_plan_borde_quieto_repeats_a_single_person_frame():
    segments = cv.plan_borde_quieto(LABELS["S1"], "S1", STRETCH, FPS, empty_frames=300,
                                    still_frames=1800)
    assert [s["kind"] for s in segments] == ["empty", "repeat", "empty"]
    repeat = segments[1]
    assert repeat["frames"] == 1800
    assert len(LABELS["S1"][repeat["start"]]) == 1
    assert cv.total_frames(segments) == 300 + 1800 + 300


def test_plan_largo_durations_add_up_and_has_a_static_gap():
    sources = {
        "S1": {"frames": 3000, "map": list(range(3000))},
        "S5": {"frames": 300, "map": None},
    }
    clips = [
        {"kind": "clip", "source": "S1", "start": 0, "frames": 1000},
        {"kind": "clip", "source": "S1", "start": 1000, "frames": 1000},
        {"kind": "clip", "source": "S5", "start": 0, "frames": 300},
    ]
    stretches = [("S1", 2000, 2400), ("S1", 2500, 2600)]
    segments = cv.plan_largo(stretches, clips, total_frames=40000, fps=FPS,
                             static_frames=3000, weights=[1.0, 0.5, 1.0, 0.8],
                             static_index=2, min_frames=1000, max_frames=20000)
    assert cv.total_frames(segments) == 40000
    kinds = [s["kind"] for s in segments]
    assert kinds == ["empty", "clip", "empty", "clip", "empty", "clip", "empty"]
    static = [s for s in segments if s["kind"] == "empty" and s["mode"] == "static"]
    assert len(static) == 1 and static[0]["frames"] == 3000 and static[0]["start"] == static[0]["end"]
    document = cv.plan_frame_map(segments, sources)
    assert len(document["frames"]) == 40000
    plan = cv.build_plan("largo", segments, sources, FPS)
    assert plan["people_frames"] == 2300
