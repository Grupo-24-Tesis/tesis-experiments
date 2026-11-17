import os
import sys

PIPELINE_DIR = os.path.abspath(os.environ.get(
    "TESIS_PIPELINE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "tesis_face_recognition_pipeline")))
sys.path.insert(0, PIPELINE_DIR)

import face_sampling


def face(frame, box=(1, 2, 11, 22), score=0.99, px=10, **extra):
    return {"frame": frame, "box": list(box), "score": score, "px": px, **extra}


def test_record_has_expected_keys():
    record = face_sampling.detections_record(25.0, 3, [0, 3, 6], [face(3)])
    assert set(record) == {"fps", "step", "frames_analysed", "detections"}
    assert record["fps"] == 25.0
    assert record["step"] == 3
    assert record["frames_analysed"] == [0, 3, 6]


def test_detection_fields_and_time():
    record = face_sampling.detections_record(25.0, 3, [0, 50], [face(50, score=0.95, px=12)])
    assert record["detections"] == [{"frame": 50, "time_s": 2.0, "box": [1, 2, 11, 22],
                                     "score": 0.95, "px": 12}]


def test_extra_keys_are_dropped():
    record = face_sampling.detections_record(
        10.0, 1, [0], [face(0, aligned="x", sharpness=3.0, interval_s=0.1, sample=0)])
    assert set(record["detections"][0]) == {"frame", "time_s", "box", "score", "px"}


def test_frames_without_faces_are_kept():
    record = face_sampling.detections_record(10.0, 1, [0, 1, 2], [face(1)])
    assert record["frames_analysed"] == [0, 1, 2]
    assert len(record["detections"]) == 1


def test_record_is_independent_of_inputs():
    frames = [0, 1]
    box = [1, 2, 3, 4]
    record = face_sampling.detections_record(10.0, 1, frames, [face(0, box=box)])
    frames.append(2)
    box[0] = 99
    assert record["frames_analysed"] == [0, 1]
    assert record["detections"][0]["box"] == [1, 2, 3, 4]


def test_empty_input():
    record = face_sampling.detections_record(30.0, 3, [], [])
    assert record["frames_analysed"] == [] and record["detections"] == []
