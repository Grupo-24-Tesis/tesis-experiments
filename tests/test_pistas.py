import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import pytest

from bench import mot, pistas

PERSON_BOX = [0, 0, 100, 200]
FAR_BOX = [500, 500, 600, 700]


def manifest(tracks):
    return {"fps": 10.0, "detect_every": 5,
            "tracks": {tid: {f"c{i}.jpg": frame for i, (frame, _) in enumerate(items)}
                       for tid, items in tracks.items()},
            "boxes": {tid: {f"c{i}.jpg": box for i, (_, box) in enumerate(items)}
                      for tid, items in tracks.items()}}


def identity(track_id, frame, box):
    return {"id": track_id, "faces": [{"frame": frame, "box": box, "score": 0.99}]}


def test_mot_votes_use_the_best_person_box_iou_of_at_least_half():
    gt = {0: [(7, [0, 0, 100, 200]), (8, [300, 0, 400, 200])],
          1: [(7, [10, 0, 110, 200])]}
    votes = pistas.mot_track_votes(manifest({
        "1": [(0, PERSON_BOX), (1, PERSON_BOX), (0, [0, 0, 100, 60])],
        "2": [(0, FAR_BOX), (5, PERSON_BOX)]}), gt)
    assert votes == {"1": {7: 2}, "2": {}}


def test_head_point_breaks_ties_between_overlapping_people():
    people = [("lejos", (20.0, 60.0)), ("cerca", (50.0, 30.0)), ("abajo", (50.0, 80.0))]
    assert pistas.person_in_head(PERSON_BOX, people) == "cerca"
    assert pistas.person_in_head(PERSON_BOX, [people[0]]) == "lejos"
    assert pistas.person_in_head(PERSON_BOX, [people[2]]) is None
    assert pistas.person_in_head(PERSON_BOX, [("x", (101.0, 10.0))]) is None


def test_chokepoint_votes_follow_the_frame_map_and_skip_unscored_frames():
    gt = {10: [("0001", (50.0, 20.0))], 11: [("0001", (40.0, 25.0)), ("0002", (50.0, 70.0))]}
    votes = pistas.chokepoint_track_votes(manifest({
        "1": [(0, PERSON_BOX), (1, PERSON_BOX), (2, PERSON_BOX), (3, PERSON_BOX)],
        "2": [(0, FAR_BOX)]}), gt, [10, 11, 12])
    assert votes == {"1": {"0001": 2}, "2": {}}


def test_chokepoint_votes_on_a_composite_map_use_prefixed_ids():
    gts = {"P1E_S1_C1": {5: [("0001", (50.0, 20.0))]},
           "P1E_S2_C2": {9: [("0001", (50.0, 20.0)), ("0003", (50.0, 60.0))]}}
    frame_map = {"frames": [["P1E_S1_C1", 5], None, ["sin_gt", -1], ["P1E_S2_C2", 9]],
                 "empty_labelled": False}
    votes = pistas.chokepoint_track_votes(manifest({
        "1": [(0, PERSON_BOX), (1, PERSON_BOX), (2, PERSON_BOX), (3, PERSON_BOX)]}),
        gts, frame_map)
    assert votes == {"1": {"P1E_S1_C1:0001": 1, "P1E_S2_C2:0001": 1}}
    assert pistas.chokepoint_track_votes(manifest({"1": [(3, [0, 50, 100, 250])]}), gts,
                                         frame_map) == {"1": {"P1E_S2_C2:0003": 1}}


def test_cctv_votes_use_the_nearest_labelled_frame_within_half_a_second():
    gt = [(0.0, [([40, 10, 60, 30], "C3-01")]),
          (2.0, [([40, 10, 60, 30], "C3-02"), ([40, 40, 60, 60], "C3-03")])]
    votes = pistas.cctv_track_votes(manifest({
        "1": [(5, PERSON_BOX), (6, PERSON_BOX), (15, PERSON_BOX), (20, PERSON_BOX)],
        "2": [(0, FAR_BOX)]}), 10.0, gt)
    assert votes == {"1": {"C3-01": 1, "C3-02": 2}, "2": {}}
    assert pistas.cctv_track_votes(manifest({"1": [(0, PERSON_BOX)]}), 10.0, []) == {"1": {}}


def test_votes_reject_a_candidate_without_box():
    broken = {"tracks": {"1": {"c0.jpg": 0}}, "boxes": {"1": {}}}
    with pytest.raises(ValueError, match="no tiene la caja"):
        pistas.mot_track_votes(broken, {})


def test_track_identity_counts_duplicates_mixed_and_unassigned():
    votes = {"1": {"A": 4, "B": 1}, "2": {"A": 5, "C": 1}, "3": {}, "4": {"D": 2}, "9": {"E": 3}}
    faces = {"1": ["A"], "2": ["C"], "3": ["A"], "4": [pistas.UNVERIFIED]}
    result = pistas.track_identity_metrics(votes, faces, {"A", "B", "C", "D", "E", "F"})
    assert result == {
        "personas_etiquetadas": 6, "identidades": 4, "encontradas": 2, "duplicados": 1,
        "mezcladas": 1, "sin_asignar": 1, "pureza": pytest.approx(11 / 13), "correctos": 0,
        "cara_correcta": 1, "cara_sin_verificar": 1}


def test_track_identity_mixed_limit_is_exact():
    labelled = {"A", "B"}
    at_limit = pistas.track_identity_metrics({"1": {"A": 12, "B": 3}}, {"1": ["A"]}, labelled)
    below = pistas.track_identity_metrics({"1": {"A": 13, "B": 3}}, {"1": ["A"]}, labelled)
    assert at_limit["mezcladas"] == 1
    assert below["mezcladas"] == 0
    assert below["correctos"] == 1


def test_track_identity_candidates_set_the_minority_rule():
    votes = {"1": {"A": 1}}
    faces = {"1": ["A"]}
    assert pistas.track_identity_metrics(votes, faces, {"A"}, {"1": 3})["sin_asignar"] == 1
    kept = pistas.track_identity_metrics(votes, faces, {"A"}, {"1": 2})
    assert (kept["sin_asignar"], kept["encontradas"], kept["cara_correcta"]) == (0, 1, 1)
    assert pistas.track_identity_metrics(votes, faces, {"A"})["encontradas"] == 1


def test_track_identity_faces_of_unassigned_tracks_are_not_counted():
    result = pistas.track_identity_metrics({"1": {}}, {"1": ["A", pistas.UNVERIFIED]}, {"A"})
    assert (result["cara_correcta"], result["cara_sin_verificar"]) == (0, 0)
    assert result["pureza"] == 0.0


def test_saved_face_rules_per_dataset():
    mot_gt = {0: [(7, PERSON_BOX)]}
    mot_faces = mot.tracks_to_identities(
        manifest({"1": [(0, PERSON_BOX)], "2": [(3, PERSON_BOX)], "3": [(0, PERSON_BOX)]}),
        {"faces": [{"track_id": "1", "candidate": "c0.jpg", "face_box": [40, 10, 60, 30],
                    "frame_idx": 0, "score": 0.9},
                   {"track_id": "2", "candidate": "c0.jpg", "face_box": [40, 10, 60, 30],
                    "frame_idx": 3, "score": 0.9},
                   {"track_id": "3", "candidate": "c0.jpg", "face_box": [40, 150, 60, 170],
                    "frame_idx": 0, "score": 0.9}]})
    assert pistas.mot_saved_faces(mot_faces, mot_gt) == {"1": [7], "2": [pistas.UNVERIFIED],
                                                         "3": [None]}
    gt = {10: [("0001", (50.0, 20.0)), ("0002", (55.0, 20.0))], 11: [("0001", (50.0, 20.0))]}
    chokepoint = [identity("1", 1, [40, 10, 60, 30]), identity("2", 0, [40, 10, 60, 30]),
                  identity("3", 5, [40, 10, 60, 30])]
    assert pistas.chokepoint_saved_faces(chokepoint, gt, [10, 11]) == {
        "1": ["0001"], "2": [None], "3": [pistas.UNVERIFIED]}
    cctv_gt = [(0.0, [([40, 10, 60, 30], "C3-01")])]
    cctv = [identity("1", 2, [42, 12, 62, 32]), identity("2", 3, [40, 10, 60, 30]),
            identity("3", 0, [70, 10, 90, 30])]
    assert pistas.cctv_saved_faces(cctv, 10.0, cctv_gt) == {
        "1": ["C3-01"], "2": [pistas.UNVERIFIED], "3": [None]}


def test_dataset_wrappers_score_only_tracks_with_a_saved_face():
    gt = [(0.0, [([40, 10, 60, 30], "C3-01")]), (1.0, [([40, 10, 60, 30], "C3-01")])]
    tracks = manifest({"1": [(0, PERSON_BOX), (10, PERSON_BOX)], "2": [(0, PERSON_BOX)]})
    result = pistas.cctv_identity(tracks, [identity("1", 4, [40, 10, 60, 30])], 10.0, gt)
    assert (result["identidades"], result["encontradas"], result["cara_correcta"],
            result["cara_sin_verificar"]) == (1, 1, 0, 1)
    gts = {"A": {1: [("p1", (50.0, 20.0))]}}
    frame_map = {"frames": [["A", 1], ["sin_gt", -1]], "empty_labelled": True}
    result = pistas.chokepoint_identity(manifest({"1": [(0, PERSON_BOX), (1, PERSON_BOX)]}),
                                        [identity("1", 0, [40, 10, 60, 30])], gts, frame_map)
    assert (result["personas_etiquetadas"], result["encontradas"], result["cara_correcta"]) == (
        1, 1, 1)
    mot_gt = {0: [(7, PERSON_BOX)], 1: [(7, PERSON_BOX)], 2: [(7, PERSON_BOX)]}
    result = pistas.mot_identity(manifest({"1": [(0, PERSON_BOX), (1, FAR_BOX), (2, FAR_BOX)]}),
                                 [identity("1", 0, [40, 10, 60, 30])], mot_gt)
    assert result["sin_asignar"] == 1
