import importlib
import os
import sys

import pytest

PIPELINE_DIR = os.path.abspath(os.environ.get(
    "TESIS_PIPELINE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "tesis_face_recognition_pipeline")))
sys.path.insert(0, PIPELINE_DIR)

import presence_gate

SETTINGS = ("GATE_CHECKS_PER_S", "GATE_REWIND_S", "GATE_EXIT_S", "GATE_CONF", "GATE_IMGSZ")


@pytest.fixture(autouse=True)
def default_environment(monkeypatch):
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    importlib.reload(presence_gate)
    yield
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    importlib.reload(presence_gate)


class FakeCapture:
    def __init__(self, count):
        self.count = count
        self.position = -1
        self.retrieved = []

    def grab(self):
        if self.position + 1 >= self.count:
            return False
        self.position += 1
        return True

    def retrieve(self):
        self.retrieved.append(self.position)
        return True, self.position


def test_defaults():
    assert presence_gate.settings() == {"checks_per_s": 4.0, "rewind_s": 1.0, "exit_s": 2.0,
                                        "conf": 0.25, "imgsz": 640}


def test_check_step_follows_the_frame_rate():
    assert presence_gate.check_step(30.0) == 8
    assert presence_gate.check_step(25.0) == 6
    assert presence_gate.check_step(2.0) == 1


def test_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("GATE_CHECKS_PER_S", "2")
    monkeypatch.setenv("GATE_REWIND_S", "0.5")
    monkeypatch.setenv("GATE_EXIT_S", "3")
    monkeypatch.setenv("GATE_CONF", "0.4")
    monkeypatch.setenv("GATE_IMGSZ", "960")
    importlib.reload(presence_gate)

    assert presence_gate.settings() == {"checks_per_s": 2.0, "rewind_s": 0.5, "exit_s": 3.0,
                                        "conf": 0.4, "imgsz": 960}


@pytest.mark.parametrize("name,value,message", [
    ("GATE_CHECKS_PER_S", "0", "GATE_CHECKS_PER_S debe ser un número positivo"),
    ("GATE_REWIND_S", "-1", "GATE_REWIND_S debe ser un número no negativo"),
    ("GATE_EXIT_S", "abc", "GATE_EXIT_S debe ser un número no negativo"),
    ("GATE_EXIT_S", "inf", "GATE_EXIT_S debe ser un número no negativo"),
    ("GATE_CONF", "nan", "GATE_CONF debe ser un número positivo"),
    ("GATE_CONF", "0", "GATE_CONF debe ser un número positivo"),
    ("GATE_CONF", "1.5", "GATE_CONF debe estar entre 0 y 1"),
    ("GATE_IMGSZ", "1.5", "GATE_IMGSZ debe ser un entero positivo"),
])
def test_invalid_settings_stop_with_a_clear_message(monkeypatch, name, value, message):
    monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit, match=message):
        importlib.reload(presence_gate)


def test_read_checks_retrieves_only_every_step_from_frame_zero():
    cap = FakeCapture(20)

    checks, confidences, frames_read = presence_gate.read_checks(
        cap, 8, lambda frame: frame == 8)

    assert cap.retrieved == [0, 8, 16]
    assert checks == [[0, False], [8, True], [16, False]]
    assert frames_read == 20


def test_a_failed_retrieve_counts_as_found_and_checking_goes_on(capsys):
    cap = FakeCapture(20)
    real_retrieve = cap.retrieve

    def retrieve():
        if cap.position == 8:
            return False, None
        return real_retrieve()

    cap.retrieve = retrieve

    checks, confidences, frames_read = presence_gate.read_checks(
        cap, 4, lambda frame: False)

    assert checks == [[0, False], [4, False], [8, True], [12, False], [16, False]]
    assert frames_read == 20
    assert "fotograma 8" in capsys.readouterr().err


def test_rewind_and_exit_may_be_zero(monkeypatch):
    monkeypatch.setenv("GATE_REWIND_S", "0")
    monkeypatch.setenv("GATE_EXIT_S", "0")
    importlib.reload(presence_gate)

    document = presence_gate.build_document("v.mp4", 10.0, 3, [[30, True]], 100, 0.1)

    assert document["intervals"] == [[30, 30]]


def test_conf_of_exactly_one_is_valid(monkeypatch):
    monkeypatch.setenv("GATE_CONF", "1")
    importlib.reload(presence_gate)
    assert presence_gate.settings()["conf"] == 1.0


def test_weights_path_is_absolute_and_beside_the_script():
    assert os.path.isabs(presence_gate.YOLO_WEIGHTS)
    assert os.path.dirname(presence_gate.YOLO_WEIGHTS) == os.path.abspath(PIPELINE_DIR)
    assert os.path.basename(presence_gate.YOLO_WEIGHTS) == "yolo26m.pt"


def test_document_has_the_documented_fields():
    checks = [[0, False], [8, True], [16, False]]

    document = presence_gate.build_document("/v/clip.mp4", 8.0, 2, checks, 20, 1.5)

    assert set(document) == {"video", "fps", "total_frames", "check_step", "settings", "checks",
                             "confidences", "intervals", "switches", "active_frames",
                             "active_fraction", "wall_s"}
    assert document["video"] == "clip.mp4"
    assert document["intervals"] == [[0, 19]]
    assert document["active_frames"] == 20
    assert document["active_fraction"] == 1.0
    assert document["checks"] == checks
    assert [s["evento"] for s in document["switches"]] == ["activo"]


def test_document_without_people_has_no_intervals():
    document = presence_gate.build_document("v.mp4", 10.0, 3, [[0, False]], 100, 0.1)

    assert document["intervals"] == []
    assert document["switches"] == []
    assert document["active_fraction"] == 0.0


def test_document_total_is_the_frames_actually_read():
    document = presence_gate.build_document("v.mp4", 10.0, 3, [[30, True]], 100, 0.1)

    assert document["total_frames"] == 100
    assert document["intervals"] == [[20, 50]]
    assert document["active_fraction"] == pytest.approx(0.31)


def test_a_container_that_under_declares_frames_cannot_clip_intervals():
    cap = FakeCapture(100)
    declared_frames = 40

    checks, confidences, frames_read = presence_gate.read_checks(
        cap, 10, lambda frame: frame == 90)
    document = presence_gate.build_document("v.mp4", 10.0, 10, checks, frames_read, 0.1)

    assert declared_frames < frames_read
    assert document["total_frames"] == 100
    assert document["intervals"] == [[80, 99]]


def test_read_checks_records_the_highest_confidence_of_each_check():
    cap = FakeCapture(20)
    values = {0: 0.0, 8: 0.62, 16: 0.0}

    checks, confidences, frames_read = presence_gate.read_checks(
        cap, 8, lambda frame: values[frame])

    assert checks == [[0, False], [8, True], [16, False]]
    assert confidences == [[0, 0.0], [8, 0.62], [16, 0.0]]


def test_an_unreadable_frame_counts_as_found_with_full_confidence():
    cap = FakeCapture(10)
    cap.retrieve = lambda: (False, None)

    checks, confidences, _ = presence_gate.read_checks(cap, 5, lambda frame: 0.0)

    assert checks == [[0, True], [5, True]]
    assert confidences == [[0, 1.0], [5, 1.0]]


def test_document_carries_confidences_aligned_with_checks():
    checks = [[0, False], [8, True]]
    confidences = [[0, 0.0], [8, 0.4]]

    document = presence_gate.build_document("v.mp4", 30.0, 8, checks, 16, 0.1, confidences)

    assert document["confidences"] == confidences
    assert ([frame for frame, _ in document["confidences"]]
            == [frame for frame, _ in document["checks"]])


def test_document_without_confidences_derives_them_from_checks():
    document = presence_gate.build_document("v.mp4", 30.0, 8, [[0, False], [8, True]], 16, 0.1)

    assert document["confidences"] == [[0, 0.0], [8, 1.0]]

