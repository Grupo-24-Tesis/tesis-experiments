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


def face(step_idx, box, embedding, score=0.995, sharpness=100.0, interval_s=0.1, frame=None):
    x1, y1, x2, y2 = box
    return {"frame": step_idx * STEP if frame is None else frame, "sample": step_idx,
            "interval_s": interval_s, "box": list(box), "score": score,
            "px": min(x2 - x1, y2 - y1), "sharpness": sharpness,
            "embedding": embedding}


def track(start, count, box, embedding, **kwargs):
    return [face(start + k, box, embedding, **kwargs) for k in range(count)]


def frames_of(identity):
    return [f["frame"] for f in identity]


SETTINGS = ("FACE_MIN_FACES", "FACE_MIN_PRESENCE_S", "FACE_MIN_FACES_ABS")


@pytest.fixture(autouse=True)
def default_environment(monkeypatch):
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    importlib.reload(identity_linking)
    yield
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    importlib.reload(identity_linking)


@pytest.fixture
def min_faces(monkeypatch):
    def set_min(value):
        monkeypatch.setattr(identity_linking, "MIN_FACES", value)
    return set_min


def test_overlapping_similar_faces_in_consecutive_frames_form_one_identity(min_faces):
    min_faces(1)
    first = face(0, (100, 100, 150, 160), axis(0))
    second = face(1, (104, 102, 154, 162), similar_to(0, 1, 0.8))

    tracklets = identity_linking.build_tracklets([first, second])
    identities = identity_linking.link_identities([first, second])

    assert tracklets == [[first, second]]
    assert identities == [[first, second]]


def test_tracklet_link_needs_overlap_and_similarity():
    start = face(0, (100, 100, 150, 160), axis(0))
    far = face(1, (300, 100, 350, 160), axis(0))
    different = face(1, (102, 100, 152, 160), axis(1))

    assert len(identity_linking.build_tracklets([start, far])) == 2
    assert len(identity_linking.build_tracklets([start, different])) == 2


def test_tracklet_survives_two_skipped_frames_but_not_three():
    box = (100, 100, 150, 160)
    start = face(0, box, axis(0))

    after_two = identity_linking.build_tracklets([start, face(3, box, axis(0))])
    after_three = identity_linking.build_tracklets([start, face(4, box, axis(0))])

    assert len(after_two) == 1
    assert len(after_three) == 2


def test_identical_faces_in_the_same_frame_are_never_merged():
    left = track(0, 10, (100, 100, 150, 160), axis(0))
    right = track(5, 10, (400, 100, 450, 160), axis(0))

    tracklets = identity_linking.build_tracklets(left + right)
    groups = identity_linking.merge_tracklets(tracklets)

    assert len(tracklets) == 2
    assert len(groups) == 2
    assert sorted(len(g["faces"]) for g in groups) == [10, 10]


def test_similar_tracklets_that_never_overlap_in_time_are_merged():
    first = track(0, 10, (100, 100, 150, 160), axis(0))
    second = track(30, 10, (400, 120, 450, 180), similar_to(0, 1, 0.9))

    tracklets = identity_linking.build_tracklets(first + second)
    identities = identity_linking.link_identities(second + first)

    assert len(tracklets) == 2
    assert len(identities) == 1
    assert frames_of(identities[0]) == frames_of(first) + frames_of(second)


def test_dissimilar_tracklets_stay_apart_and_are_ordered_by_first_appearance():
    late = track(30, 10, (400, 120, 450, 180), similar_to(0, 1, 0.2))
    early = track(0, 10, (100, 100, 150, 160), axis(0))

    identities = identity_linking.link_identities(late + early)

    assert len(identities) == 2
    assert identities[0] == early
    assert identities[1] == late


def test_co_present_mirror_twin_drops_the_lower_quality_identity():
    sharp = track(0, 10, (100, 100, 160, 170), axis(0), sharpness=200.0)
    blurry = track(2, 10, (400, 100, 440, 150), similar_to(0, 1, 0.7), sharpness=20.0)

    identities = identity_linking.link_identities(blurry + sharp)

    assert identities == [sharp]


def test_co_present_people_below_the_twin_threshold_are_both_kept():
    sharp = track(0, 10, (100, 100, 160, 170), axis(0), sharpness=200.0)
    other = track(2, 10, (400, 100, 440, 150), similar_to(0, 1, 0.5), sharpness=20.0)

    identities = identity_linking.link_identities(other + sharp)

    assert identities == [sharp, other]


def test_identities_with_fewer_than_the_minimum_faces_are_dropped():
    assert identity_linking.MIN_FACES == 8
    kept = track(0, 8, (100, 100, 150, 160), axis(0))
    short = track(0, 7, (400, 100, 450, 160), axis(1))

    identities = identity_linking.link_identities(kept + short)

    assert identities == [kept]


def test_minimum_faces_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("FACE_MIN_FACES", "5")

    assert importlib.reload(identity_linking).MIN_FACES == 5


@pytest.mark.parametrize("value", ["", "  "])
def test_empty_minimum_faces_uses_the_default(monkeypatch, value):
    monkeypatch.setenv("FACE_MIN_FACES", value)

    assert importlib.reload(identity_linking).MIN_FACES == 8


@pytest.mark.parametrize("value", ["ocho", "2.5", "0", "-3"])
def test_invalid_minimum_faces_stops_with_a_clear_message(monkeypatch, value):
    monkeypatch.setenv("FACE_MIN_FACES", value)

    with pytest.raises(SystemExit, match="FACE_MIN_FACES debe ser un entero positivo"):
        importlib.reload(identity_linking)


def test_positive_float_setting_reads_the_environment(monkeypatch):
    monkeypatch.delenv("FACE_INTERVAL_S", raising=False)
    assert identity_linking.env_positive_float("FACE_INTERVAL_S", 0.1) == 0.1

    monkeypatch.setenv("FACE_INTERVAL_S", "0.5")
    assert identity_linking.env_positive_float("FACE_INTERVAL_S", 0.1) == 0.5

    monkeypatch.setenv("FACE_INTERVAL_S", "")
    assert identity_linking.env_positive_float("FACE_INTERVAL_S", 0.1) == 0.1


@pytest.mark.parametrize("value", ["rapido", "0", "-0.1", "nan", "inf"])
def test_invalid_positive_float_setting_stops_with_a_clear_message(monkeypatch, value):
    monkeypatch.setenv("FACE_INTERVAL_S", value)

    with pytest.raises(SystemExit, match="FACE_INTERVAL_S debe ser un número positivo"):
        identity_linking.env_positive_float("FACE_INTERVAL_S", 0.1)


def test_faces_below_the_minimum_score_are_ignored(min_faces):
    box = (100, 100, 150, 160)
    confident = track(0, 8, box, axis(0))
    doubtful = track(8, 2, box, axis(0), score=0.98)
    alone = track(0, 10, (400, 100, 450, 160), axis(1), score=0.985)

    identities = identity_linking.link_identities(confident + doubtful + alone)

    assert identities == [confident]
    min_faces(1)
    assert identity_linking.link_identities(alone) == []


def test_tracklet_skip_window_counts_analysed_frames_not_video_frames():
    box = (100, 100, 150, 160)
    start = face(0, box, axis(0), frame=0)
    after_long_idle = face(3, box, axis(0), frame=90)
    after_short_steps = face(4, box, axis(0), frame=12)

    assert len(identity_linking.build_tracklets([start, after_long_idle])) == 1
    assert len(identity_linking.build_tracklets([start, after_short_steps])) == 2


@pytest.mark.parametrize("value", ["", "  "])
def test_empty_flag_uses_the_default(monkeypatch, value):
    monkeypatch.setenv("FACE_ADAPTIVE", value)

    assert identity_linking.env_flag("FACE_ADAPTIVE", True) is True
    assert identity_linking.env_flag("FACE_ADAPTIVE", False) is False


def test_flag_reads_zero_and_one(monkeypatch):
    monkeypatch.setenv("FACE_ADAPTIVE", "0")
    assert identity_linking.env_flag("FACE_ADAPTIVE", True) is False

    monkeypatch.setenv("FACE_ADAPTIVE", " 1 ")
    assert identity_linking.env_flag("FACE_ADAPTIVE", False) is True


@pytest.mark.parametrize("value", ["si", "true", "2", "-1", "1.0"])
def test_invalid_flag_stops_with_a_clear_message(monkeypatch, value):
    monkeypatch.setenv("FACE_ADAPTIVE", value)

    with pytest.raises(SystemExit, match="FACE_ADAPTIVE debe ser 0 o 1"):
        identity_linking.env_flag("FACE_ADAPTIVE", True)


def test_presence_defaults():
    assert identity_linking.MIN_PRESENCE_S == 0.8
    assert identity_linking.MIN_FACES_ABS == 3


def test_presence_sums_the_interval_of_every_face():
    faces = (track(0, 2, (100, 100, 150, 160), axis(0), interval_s=0.3)
             + track(2, 2, (100, 100, 150, 160), axis(0), interval_s=0.1)
             + track(4, 1, (100, 100, 150, 160), axis(0), interval_s=1.0))

    assert identity_linking.presence_s(faces) == pytest.approx(1.8)


def test_by_time_keeps_identities_present_long_enough_at_mixed_intervals():
    box = (100, 100, 150, 160)
    enough = (track(0, 2, box, axis(0), interval_s=0.3)
              + track(2, 2, box, axis(0), interval_s=0.1))
    short = (track(0, 2, (400, 100, 450, 160), axis(1), interval_s=0.3)
             + track(2, 1, (400, 100, 450, 160), axis(1), interval_s=0.1))

    identities = identity_linking.link_identities(enough + short, by_time=True)

    assert identities == [enough]


def test_by_time_eight_faces_at_the_dense_interval_reach_the_minimum():
    eight = track(0, 8, (100, 100, 150, 160), axis(0), interval_s=0.1)
    seven = track(0, 7, (400, 100, 450, 160), axis(1), interval_s=0.1)

    identities = identity_linking.link_identities(eight + seven, by_time=True)

    assert identities == [eight]


def test_by_time_tolerates_rounding_of_the_summed_intervals(monkeypatch):
    monkeypatch.setattr(identity_linking, "MIN_PRESENCE_S", 0.9)
    three = track(0, 3, (100, 100, 150, 160), axis(0), interval_s=9 / 30)

    assert identity_linking.presence_s(three) < 0.9
    assert identity_linking.link_identities(three, by_time=True) == [three]


def test_by_time_needs_the_absolute_minimum_of_faces():
    box = (100, 100, 150, 160)
    two_long = track(0, 2, box, axis(0), interval_s=1.0)
    three_long = track(0, 3, (400, 100, 450, 160), axis(1), interval_s=1.0)

    identities = identity_linking.link_identities(two_long + three_long, by_time=True)

    assert identities == [three_long]


def test_fixed_mode_keeps_the_face_count_rule_and_ignores_the_intervals():
    many_short = track(0, 8, (100, 100, 150, 160), axis(0), interval_s=0.01)
    few_long = track(0, 7, (400, 100, 450, 160), axis(1), interval_s=1.0)

    identities = identity_linking.link_identities(many_short + few_long)

    assert identities == [many_short]
    assert identity_linking.link_identities(many_short + few_long, by_time=False) == [many_short]


def test_presence_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("FACE_MIN_PRESENCE_S", "1.5")
    monkeypatch.setenv("FACE_MIN_FACES_ABS", "5")

    module = importlib.reload(identity_linking)

    assert module.MIN_PRESENCE_S == 1.5
    assert module.MIN_FACES_ABS == 5


@pytest.mark.parametrize("name, value, message", [
    ("FACE_MIN_PRESENCE_S", "0", "FACE_MIN_PRESENCE_S debe ser un número positivo"),
    ("FACE_MIN_PRESENCE_S", "mucho", "FACE_MIN_PRESENCE_S debe ser un número positivo"),
    ("FACE_MIN_FACES_ABS", "2.5", "FACE_MIN_FACES_ABS debe ser un entero positivo"),
    ("FACE_MIN_FACES_ABS", "0", "FACE_MIN_FACES_ABS debe ser un entero positivo"),
])
def test_invalid_presence_settings_stop_with_a_clear_message(monkeypatch, name, value, message):
    monkeypatch.setenv(name, value)

    with pytest.raises(SystemExit, match=message):
        importlib.reload(identity_linking)
