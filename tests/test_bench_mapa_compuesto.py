import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import pytest

from bench import metrics


def det(frame, box):
    return {"frame": frame, "box": box, "score": 0.99, "px": 10}


GTS = {
    "A": {1: [("p1", (10.0, 10.0))], 2: [("p1", (10.0, 10.0)), ("p2", (100.0, 10.0))], 3: []},
    "B": {1: [("p1", (50.0, 50.0))]},
}
BOX_A = [5, 5, 15, 15]
BOX_B = [45, 45, 55, 55]


def test_legacy_list_map_still_resolves():
    gt = {7: [("a", (1.0, 1.0))]}
    assert metrics.people_at(gt, [7, 8], 0) == [("a", (1.0, 1.0))]
    assert metrics.people_at(gt, [7, 8], 1) is None
    assert metrics.people_at(gt, [7, 8], 5) is None
    assert metrics.people_at(gt, None, 7) == [("a", (1.0, 1.0))]


def test_composite_entries_prefix_person_ids_with_the_source_name():
    frame_map = {"frames": [["A", 1], ["B", 1]], "empty_labelled": True}
    assert metrics.people_at(GTS, frame_map, 0) == [("A:p1", (10.0, 10.0))]
    assert metrics.people_at(GTS, frame_map, 1) == [("B:p1", (50.0, 50.0))]


def test_composite_null_is_empty_only_when_labelled_empty():
    labelled = {"frames": [None], "empty_labelled": True}
    unlabelled = {"frames": [None], "empty_labelled": False}
    assert metrics.people_at(GTS, labelled, 0) == []
    assert metrics.people_at(GTS, unlabelled, 0) is None
    assert metrics.people_at(GTS, [None, ["A", 1]], 0) is None


def test_composite_unscored_entries_and_unknown_frames_are_skipped():
    frame_map = {"frames": [["sin_gt", -1], ["A", 99], ["Z", 1]], "empty_labelled": True}
    assert metrics.people_at(GTS, frame_map, 0) is None
    assert metrics.people_at(GTS, frame_map, 1) is None
    assert metrics.people_at(GTS, frame_map, 2) is None
    assert metrics.people_at(GTS, frame_map, 3) is None


def test_composite_detection_counts_false_positives_on_empty_frames():
    frame_map = {"frames": [None, ["A", 1], ["sin_gt", -1]], "empty_labelled": True}
    detections = [det(0, [300, 300, 320, 320]), det(1, BOX_A), det(2, [0, 0, 5, 5])]
    r = metrics.chokepoint_detection_metrics(detections, [0, 1, 2], GTS, frame_map)
    assert (r["tp"], r["fn"], r["fp"]) == (1, 0, 1)


def test_composite_detection_without_empty_labelled_ignores_null_frames():
    frame_map = {"frames": [None, ["A", 1]], "empty_labelled": False}
    r = metrics.chokepoint_detection_metrics([det(0, [0, 0, 5, 5]), det(1, BOX_A)], [0, 1],
                                             GTS, frame_map)
    assert (r["tp"], r["fn"], r["fp"]) == (1, 0, 0)


def test_composite_identity_keeps_same_person_id_in_two_sources_distinct():
    frame_map = {"frames": [["A", 1], ["B", 1]], "empty_labelled": True}
    identities = [
        {"id": 1, "faces": [{"frame": 0, "box": BOX_A}]},
        {"id": 2, "faces": [{"frame": 1, "box": BOX_B}]},
    ]
    r = metrics.chokepoint_identity_metrics(identities, GTS, frame_map)
    assert r["labelled"] == 2
    assert r["found"] == 2
    assert r["duplicates"] == 0


def test_composite_identity_counts_only_people_in_mapped_frames():
    frame_map = {"frames": [["A", 1]], "empty_labelled": True}
    r = metrics.chokepoint_identity_metrics([], GTS, frame_map)
    assert r["labelled"] == 1


def test_composite_identity_face_in_empty_frame_is_unmatched():
    frame_map = {"frames": [None, ["A", 1]], "empty_labelled": True}
    identities = [{"id": 1, "faces": [{"frame": 0, "box": BOX_A}, {"frame": 1, "box": BOX_A}]}]
    r = metrics.chokepoint_identity_metrics(identities, GTS, frame_map)
    assert r["faces_total"] == 2
    assert r["faces_matched"] == 1
    assert r["found"] == 1


def test_referenced_sources_lists_names_without_unscored():
    frame_map = {"frames": [None, ["A", 1], ["B", 2], ["sin_gt", -1], ["A", 5]],
                 "empty_labelled": True}
    assert metrics.referenced_sources(frame_map) == ["A", "B"]
    assert metrics.referenced_sources([1, 2, 3]) == []


def test_is_composite_map():
    assert metrics.is_composite_map({"frames": [], "empty_labelled": True})
    assert metrics.is_composite_map([["A", 1]])
    assert metrics.is_composite_map([None, 3])
    assert not metrics.is_composite_map([1, 2, 3])
    assert not metrics.is_composite_map(None)
