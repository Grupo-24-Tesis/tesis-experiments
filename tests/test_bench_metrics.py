import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import pytest

from bench import metrics

XML = """<sequence name="x">
<frame number="10">
<person id="0001"><leftEye x="10" y="20"/><rightEye x="20" y="20"/></person>
<person id="0002"><leftEye x="100" y="50"/><rightEye x="110" y="50"/></person>
</frame>
<frame number="11">
<person id="0001"><leftEye x="12" y="20"/><rightEye x="22" y="20"/></person>
</frame>
<frame number="12">
</frame>
</sequence>"""


def det(frame, box, score=0.99):
    return {"frame": frame, "box": box, "score": score, "px": 10}


def test_parse_chokepoint_gt_uses_eye_midpoint():
    gt = metrics.parse_chokepoint_gt(XML)
    assert gt[10] == [("0001", (15.0, 20.0)), ("0002", (105.0, 50.0))]
    assert gt[11] == [("0001", (17.0, 20.0))]
    assert 12 not in gt or gt[12] == []


def test_parse_cctv_gt_drops_ignored_and_unknown():
    document = {"frames": {
        "0": {"time_s": 2.0, "faces": [
            {"box": [0, 0, 10, 10], "identity": "A"},
            {"box": [0, 0, 5, 5], "identity": None},
            {"box": [1, 1, 5, 5], "identity": "B", "ignore": True},
            {"box": [2, 2, 5, 5], "identity": "?x"}]},
        "1": {"time_s": 1.0, "faces": []}}}
    gt = metrics.parse_cctv_gt(document)
    assert [t for t, _ in gt] == [1.0, 2.0]
    assert gt[1][1] == [([0, 0, 10, 10], "A")]


def test_iou_and_inside():
    assert metrics.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert metrics.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert metrics.iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert metrics.inside([0, 0, 10, 10], (10, 0))
    assert not metrics.inside([0, 0, 10, 10], (11, 5))


def test_precision_recall_f1():
    r = metrics.precision_recall_f1(8, 2, 8)
    assert r["precision"] == 0.8 and r["recall"] == 0.5
    assert r["f1"] == pytest.approx(2 * 0.8 * 0.5 / 1.3)
    assert metrics.precision_recall_f1(0, 0, 0)["f1"] == 0.0


def test_chokepoint_detection_counts():
    gt = metrics.parse_chokepoint_gt(XML)
    frame_map = [10, 11, 12, 99]
    detections = [det(0, [5, 10, 25, 30]), det(0, [300, 300, 320, 320]), det(1, [0, 0, 5, 5])]
    r = metrics.chokepoint_detection_metrics(detections, [0, 1, 2, 3], gt, frame_map)
    assert (r["tp"], r["fn"], r["fp"]) == (1, 2, 2)


def test_chokepoint_detection_two_boxes_on_one_person_is_not_tp():
    gt = {5: [("a", (10.0, 10.0))]}
    detections = [det(0, [0, 0, 20, 20]), det(0, [5, 5, 15, 15])]
    r = metrics.chokepoint_detection_metrics(detections, [0], gt, [5])
    assert (r["tp"], r["fn"], r["fp"]) == (0, 1, 0)


def test_chokepoint_detection_skips_unlabelled_frames():
    gt = {5: [("a", (10.0, 10.0))]}
    r = metrics.chokepoint_detection_metrics([det(1, [0, 0, 5, 5])], [0, 1], gt, [5, 6])
    assert (r["tp"], r["fn"], r["fp"]) == (0, 1, 0)


def identity(identity_id, faces):
    return {"id": identity_id, "faces": [{"frame": f, "box": b} for f, b in faces]}


def test_chokepoint_identity_metrics_found_duplicate_mixed_purity():
    gt = {
        0: [("a", (10.0, 10.0)), ("b", (100.0, 10.0))],
        1: [("a", (10.0, 10.0)), ("b", (100.0, 10.0))],
        2: [("a", (10.0, 10.0)), ("b", (100.0, 10.0))],
    }
    box_a, box_b = [5, 5, 15, 15], [95, 5, 105, 15]
    identities = [
        identity(1, [(0, box_a), (1, box_a)]),
        identity(2, [(2, box_a)]),
        identity(3, [(0, box_b), (1, box_b), (2, box_b)]),
        identity(4, [(0, box_a), (1, box_b)]),
        identity(5, [(0, [500, 500, 510, 510])]),
    ]
    r = metrics.chokepoint_identity_metrics(identities, gt, None)
    assert r["labelled"] == 2
    assert r["identities"] == 5
    assert r["found"] == 2
    assert r["duplicates"] == 2
    assert r["mixed"] == 1
    assert r["without_person"] == 1
    assert r["faces_total"] == 9
    assert r["faces_matched"] == 8
    assert r["purity"] == pytest.approx(7 / 8)


def test_chokepoint_identity_metrics_frame_map_out_of_range():
    gt = {7: [("a", (10.0, 10.0))]}
    r = metrics.chokepoint_identity_metrics([identity(1, [(0, [0, 0, 20, 20]), (5, [0, 0, 20, 20])])],
                                            gt, [7])
    assert r["faces_total"] == 1 and r["found"] == 1


def test_cctv_identity_metrics_time_and_iou():
    gt = [(1.0, [([0, 0, 10, 10], "A")]), (2.0, [([50, 50, 60, 60], "B")])]
    identities = [
        identity(1, [(10, [0, 0, 10, 10]), (11, [0, 0, 10, 10])]),
        identity(2, [(20, [50, 50, 60, 60])]),
        identity(3, [(100, [0, 0, 10, 10])]),
        identity(4, [(10, [30, 30, 40, 40])]),
    ]
    r = metrics.cctv_identity_metrics(identities, 10.0, gt)
    assert r["labelled"] == 2 and r["found"] == 2 and r["duplicates"] == 0
    assert r["mixed"] == 0 and r["without_person"] == 2
    assert r["faces_total"] == 4 and r["faces_matched"] == 3


def test_match_boxes_is_one_to_one_greedy():
    gt = [[0, 0, 10, 10]]
    boxes = [[0, 0, 10, 10], [0, 0, 10, 9]]
    assert metrics.match_boxes(gt, boxes, 0.3) == 1
    assert metrics.match_boxes([[0, 0, 10, 10]], [[50, 50, 60, 60]], 0.3) == 0


def test_cctv_detection_metrics():
    gt = [(1.0, [([0, 0, 10, 10], "A"), ([50, 50, 60, 60], "B")]), (5.0, [([0, 0, 10, 10], "A")])]
    detections = [det(10, [0, 0, 10, 10]), det(10, [200, 200, 210, 210])]
    r = metrics.cctv_detection_metrics(detections, [10, 20], 10.0, gt)
    assert (r["tp"], r["fn"], r["fp"]) == (1, 2, 1)


def test_cctv_detection_nearest_frame_within_window():
    gt = [(1.0, [([0, 0, 10, 10], "A")])]
    near = metrics.cctv_detection_metrics([det(11, [0, 0, 10, 10])], [11], 10.0, gt)
    far = metrics.cctv_detection_metrics([det(13, [0, 0, 10, 10])], [13], 10.0, gt)
    assert near["tp"] == 1
    assert (far["tp"], far["fn"], far["fp"]) == (0, 1, 0)


def test_missing_detections():
    baseline = [det(1, [0, 0, 10, 10]), det(2, [0, 0, 10, 10]), det(3, [0, 0, 10, 10])]
    candidate = [det(1, [0, 0, 10, 10]), det(2, [3, 3, 13, 13])]
    missing = metrics.missing_detections(baseline, candidate)
    assert [d["frame"] for d in missing] == [2, 3]


def test_format_time():
    assert metrics.format_time(0) == "00:00:00.0"
    assert metrics.format_time(3725.46) == "01:02:05.4"


def test_cctv_detection_shared_analysed_frame_counts_fp_once():
    gt = [(1.0, [([0, 0, 10, 10], "A")]), (1.05, [([0, 0, 10, 10], "A")])]
    detections = [det(10, [0, 0, 10, 10]), det(10, [200, 200, 210, 210])]
    r = metrics.cctv_detection_metrics(detections, [10], 10.0, gt)
    assert (r["tp"], r["fn"], r["fp"]) == (2, 0, 1)
