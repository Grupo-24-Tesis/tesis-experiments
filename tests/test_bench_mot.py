import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from bench import metrics, mot


def person(pid, x=0, y=0, w=20, h=100):
    return (pid, [x, y, x + w, y + h])


def test_ground_truth_filters_and_shifts_frames(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("1,5,10,20,30,40,1,1,1\n1,6,0,0,5,5,0,1,1\n2,7,0,0,5,5,1,2,1\n3,5,1,2,3,4,1,1,0.5\n")
    gt = mot.load_mot_ground_truth(str(path))
    assert sorted(gt) == [0, 2]
    assert gt[0] == [(5, [10.0, 20.0, 40.0, 60.0])]
    assert gt[2] == [(5, [1.0, 2.0, 4.0, 6.0])]


MANIFEST = {
    "tracks": {"1": {"a.jpg": 4}, "2": {"b.jpg": 4}},
    "boxes": {"1": {"a.jpg": [100, 200, 140, 300]}, "2": {"b.jpg": [0, 0, 10, 10]}},
}
RESULTS = {"faces": [{"track_id": "1", "candidate": "a.jpg", "face_box": [5, 3, 15, 13],
                      "frame_idx": 4, "score": 0.97}]}


def test_tracks_to_identities_shifts_crop_and_skips_faceless_tracks():
    identities = mot.tracks_to_identities(MANIFEST, RESULTS)
    assert identities == [{"id": "1", "faces": [{"frame": 4, "box": [105, 203, 115, 213],
                                                   "score": 0.97}]}]


def test_assign_face_head_region_and_below():
    people = [person(1)]
    assert mot.assign_face([5, 5, 15, 15], people) == 1
    assert mot.assign_face([5, 60, 15, 70], people) is None
    assert mot.assign_face([50, 5, 60, 15], people) is None


def test_assign_face_prefers_nearest_head():
    people = [person(1, x=0, w=40), person(2, x=20, w=40)]
    assert mot.assign_face([42, 8, 48, 16], people) == 2
    assert mot.assign_face([12, 8, 18, 16], people) == 1


def gt_two(frames=range(10)):
    return {f: [person(1, x=0), person(2, x=100)] for f in frames}


def face(frame, x, score=0.9):
    return {"frame": frame, "box": [x, 5, x + 8, 13], "score": score}


def test_identity_metrics_found_duplicates_and_correct():
    identities = [
        {"id": "a", "faces": [face(0, 5), face(1, 5)]},
        {"id": "b", "faces": [face(2, 5)]},
        {"id": "c", "faces": [face(3, 105), face(4, 105)]},
    ]
    result = mot.identity_metrics(identities, gt_two())
    assert result["personas_etiquetadas"] == 2
    assert result["identidades"] == 3
    assert result["encontradas"] == 2
    assert result["duplicados"] == 1
    assert result["mezcladas"] == 0
    assert result["sin_asignar"] == 0
    assert result["pureza"] == 1.0
    assert result["correctos"] == 1


def test_identity_metrics_mixed_at_limit():
    five = [face(i, 5) for i in range(4)] + [face(4, 105)]
    just_below = [face(i, 5) for i in range(5)] + [face(5, 105)]
    result = mot.identity_metrics([{"id": "m", "faces": five}], gt_two())
    assert result["mezcladas"] == 1
    assert result["correctos"] == 0
    assert result["pureza"] == 0.8
    result = mot.identity_metrics([{"id": "m", "faces": just_below}], gt_two())
    assert result["mezcladas"] == 0


def test_identity_metrics_unassigned_when_minority_or_empty():
    far = {"frame": 0, "box": [500, 5, 508, 13], "score": 0.9}
    identities = [
        {"id": "x", "faces": [far, far, far, face(0, 5)]},
        {"id": "y", "faces": []},
        {"id": "z", "faces": [face(0, 5)]},
    ]
    result = mot.identity_metrics(identities, gt_two())
    assert result["sin_asignar"] == 2
    assert result["encontradas"] == 1
    assert result["pureza"] == 1.0


def test_tracking_metrics_iou_duplicates_and_mixed():
    gt = {f: [person(1, x=0), person(2, x=100)] for f in range(6)}
    box1 = [0, 0, 20, 100]
    box2 = [100, 0, 120, 100]
    manifest = {
        "tracks": {
            "t1": {"a": 0, "b": 1},
            "t2": {"c": 2},
            "t3": {"d": 3, "e": 4, "f": 5},
            "t4": {"g": 0},
        },
        "boxes": {
            "t1": {"a": box1, "b": box1},
            "t2": {"c": [2, 0, 22, 100]},
            "t3": {"d": box1, "e": box2, "f": box2},
            "t4": {"g": [500, 500, 520, 600]},
        },
    }
    result = mot.tracking_metrics(manifest, gt)
    assert result["personas_etiquetadas"] == 2
    assert result["pistas"] == 4
    assert result["personas_cubiertas"] == 2
    assert result["pistas_duplicadas"] == 1
    assert result["pistas_mezcladas"] == 1
    assert result["pistas_sin_asignar"] == 1


def test_footage_statistics():
    gt = {0: [person(1, h=100), person(2, h=200)], 1: [person(1, h=300)], 2: []}
    detections = {"frames_analysed": [0, 1, 2],
                  "detections": [{"frame": 0, "px": 10}, {"frame": 0, "px": 20},
                                 {"frame": 1, "px": 30}]}
    stats = mot.footage_statistics(gt, 1000, detections)
    assert stats["altura_mediana_px"] == 200
    assert stats["altura_relativa"] == 0.2
    assert stats["personas_por_fotograma_mediana"] == 1.5
    assert stats["rostro_px_mediana"] == 20
    assert stats["detecciones_por_fotograma_mediana"] == 1
    assert "rostro_px_mediana" not in mot.footage_statistics(gt, 1000)


def test_converted_tracks_feed_cctv_metrics():
    identities = mot.tracks_to_identities(MANIFEST, RESULTS)
    gt = [(0.16, [([105, 203, 115, 213], 7)])]
    result = metrics.cctv_identity_metrics(identities, 25.0, gt)
    assert result["found"] == 1
    assert result["faces_matched"] == 1
