import importlib
import os
import sys

import numpy as np
import pytest

PIPELINE_DIR = os.path.abspath(os.environ.get(
    "TESIS_PIPELINE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "tesis_face_recognition_pipeline")))
sys.path.insert(0, PIPELINE_DIR)

import identity_linking

STEP = 3
DIM = 16


def axis(k):
    v = np.zeros(DIM, dtype=np.float32)
    v[k] = 1.0
    return v


def similar_to(k, other, similarity):
    v = similarity * axis(k) + np.sqrt(1 - similarity ** 2) * axis(other)
    return v.astype(np.float32)


def face(step_idx, box, embedding, score=0.995, sharpness=100.0):
    x1, y1, x2, y2 = box
    return {"frame": step_idx * STEP, "box": list(box), "score": score,
            "px": min(x2 - x1, y2 - y1), "sharpness": sharpness,
            "embedding": embedding}


def track(start, count, box, embedding, **kwargs):
    return [face(start + k, box, embedding, **kwargs) for k in range(count)]


def frames_of(identity):
    return [f["frame"] for f in identity]


@pytest.fixture
def min_faces(monkeypatch):
    def set_min(value):
        monkeypatch.setattr(identity_linking, "MIN_FACES", value)
    return set_min


def test_overlapping_similar_faces_in_consecutive_frames_form_one_identity(min_faces):
    min_faces(1)
    first = face(0, (100, 100, 150, 160), axis(0))
    second = face(1, (104, 102, 154, 162), similar_to(0, 1, 0.8))

    tracklets = identity_linking.build_tracklets([first, second], STEP)
    identities = identity_linking.link_identities([first, second], STEP)

    assert tracklets == [[first, second]]
    assert identities == [[first, second]]


def test_tracklet_link_needs_overlap_and_similarity():
    start = face(0, (100, 100, 150, 160), axis(0))
    far = face(1, (300, 100, 350, 160), axis(0))
    different = face(1, (102, 100, 152, 160), axis(1))

    assert len(identity_linking.build_tracklets([start, far], STEP)) == 2
    assert len(identity_linking.build_tracklets([start, different], STEP)) == 2


def test_tracklet_survives_two_skipped_frames_but_not_three():
    box = (100, 100, 150, 160)
    start = face(0, box, axis(0))

    after_two = identity_linking.build_tracklets([start, face(3, box, axis(0))], STEP)
    after_three = identity_linking.build_tracklets([start, face(4, box, axis(0))], STEP)

    assert len(after_two) == 1
    assert len(after_three) == 2


def test_identical_faces_in_the_same_frame_are_never_merged():
    left = track(0, 10, (100, 100, 150, 160), axis(0))
    right = track(5, 10, (400, 100, 450, 160), axis(0))

    tracklets = identity_linking.build_tracklets(left + right, STEP)
    groups = identity_linking.merge_tracklets(tracklets)

    assert len(tracklets) == 2
    assert len(groups) == 2
    assert sorted(len(g["faces"]) for g in groups) == [10, 10]


def test_similar_tracklets_that_never_overlap_in_time_are_merged():
    first = track(0, 10, (100, 100, 150, 160), axis(0))
    second = track(30, 10, (400, 120, 450, 180), similar_to(0, 1, 0.9))

    tracklets = identity_linking.build_tracklets(first + second, STEP)
    identities = identity_linking.link_identities(second + first, STEP)

    assert len(tracklets) == 2
    assert len(identities) == 1
    assert frames_of(identities[0]) == frames_of(first) + frames_of(second)


def test_dissimilar_tracklets_stay_apart_and_are_ordered_by_first_appearance():
    late = track(30, 10, (400, 120, 450, 180), similar_to(0, 1, 0.2))
    early = track(0, 10, (100, 100, 150, 160), axis(0))

    identities = identity_linking.link_identities(late + early, STEP)

    assert len(identities) == 2
    assert identities[0] == early
    assert identities[1] == late


def test_co_present_mirror_twin_drops_the_lower_quality_identity():
    sharp = track(0, 10, (100, 100, 160, 170), axis(0), sharpness=200.0)
    blurry = track(2, 10, (400, 100, 440, 150), similar_to(0, 1, 0.7), sharpness=20.0)

    identities = identity_linking.link_identities(blurry + sharp, STEP)

    assert identities == [sharp]


def test_co_present_people_below_the_twin_threshold_are_both_kept():
    sharp = track(0, 10, (100, 100, 160, 170), axis(0), sharpness=200.0)
    other = track(2, 10, (400, 100, 440, 150), similar_to(0, 1, 0.5), sharpness=20.0)

    identities = identity_linking.link_identities(other + sharp, STEP)

    assert identities == [sharp, other]


def test_identities_with_fewer_than_the_minimum_faces_are_dropped():
    assert identity_linking.MIN_FACES == 8
    kept = track(0, 8, (100, 100, 150, 160), axis(0))
    short = track(0, 7, (400, 100, 450, 160), axis(1))

    identities = identity_linking.link_identities(kept + short, STEP)

    assert identities == [kept]


def test_minimum_faces_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("FACE_MIN_FACES", "5")
    try:
        assert importlib.reload(identity_linking).MIN_FACES == 5
    finally:
        monkeypatch.delenv("FACE_MIN_FACES")
        importlib.reload(identity_linking)
    assert identity_linking.MIN_FACES == 8


def test_faces_below_the_minimum_score_are_ignored(min_faces):
    box = (100, 100, 150, 160)
    confident = track(0, 8, box, axis(0))
    doubtful = track(8, 2, box, axis(0), score=0.98)
    alone = track(0, 10, (400, 100, 450, 160), axis(1), score=0.985)

    identities = identity_linking.link_identities(confident + doubtful + alone, STEP)

    assert identities == [confident]
    min_faces(1)
    assert identity_linking.link_identities(alone, STEP) == []
