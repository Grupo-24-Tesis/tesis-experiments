import importlib
import os
import sys

import pytest

PIPELINE_DIR = os.path.abspath(os.environ.get(
    "TESIS_PIPELINE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "tesis_face_recognition_pipeline")))
sys.path.insert(0, PIPELINE_DIR)

import face_sampling

FPS = 30.0
HOLD = 30
SETTINGS = ("FACE_ADAPTIVE", "FACE_INTERVAL_S", "FACE_INTERVAL_DENSE_S", "FACE_INTERVAL_SPARSE_S",
            "FACE_INTERVAL_IDLE_S", "FACE_CROWD_FACES", "FACE_IDLE_FRAMES", "FACE_DENSE_HOLD_S")
LEFT = (100, 100, 150, 160)
MIDDLE = (300, 100, 350, 160)
RIGHT = (500, 100, 550, 160)


@pytest.fixture(autouse=True)
def default_environment(monkeypatch):
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    importlib.reload(face_sampling)
    yield
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    importlib.reload(face_sampling)


def shifted(box, dx):
    return (box[0] + dx, box[1], box[2] + dx, box[3])


def state(mode, boxes=(), empty=0, hold_until=-1):
    return {"mode": mode, "boxes": [list(b) for b in boxes], "empty": empty,
            "hold_until": hold_until}


def run(frames, adaptive=True):
    steps = face_sampling.sampling_steps(FPS, adaptive)
    hold = face_sampling.hold_frames(FPS)
    current = face_sampling.start_state(adaptive)
    frame_idx, visited = 0, []
    while frame_idx < len(frames):
        visited.append((frame_idx, current["mode"]))
        current = face_sampling.advance(current, frames[frame_idx], frame_idx, hold)
        frame_idx += steps[current["mode"]]
    return visited


def test_defaults_in_seconds_become_frame_steps():
    assert face_sampling.ADAPTIVE is True
    assert face_sampling.sampling_steps(FPS, True) == {
        face_sampling.DENSE: 3, face_sampling.SPARSE: 9, face_sampling.IDLE: 30}
    assert face_sampling.hold_frames(FPS) == HOLD
    assert face_sampling.CROWD_FACES == 3
    assert face_sampling.IDLE_FRAMES == 3


def test_steps_scale_with_the_frame_rate():
    assert face_sampling.sampling_steps(10.0, True) == {
        face_sampling.DENSE: 1, face_sampling.SPARSE: 3, face_sampling.IDLE: 10}
    assert face_sampling.hold_frames(10.0) == 10


def test_adaptive_sampling_starts_dense():
    assert face_sampling.start_state(True)["mode"] == face_sampling.DENSE


def test_crowd_of_continuing_faces_is_dense():
    previous = state(face_sampling.SPARSE, [LEFT, MIDDLE, RIGHT])
    boxes = [shifted(b, 4) for b in (LEFT, MIDDLE, RIGHT)]

    after = face_sampling.advance(previous, boxes, 90, HOLD)

    assert after["mode"] == face_sampling.DENSE
    assert after["hold_until"] == 90 + HOLD


def test_new_arrival_is_dense_even_with_few_faces():
    previous = state(face_sampling.SPARSE, [LEFT])

    after = face_sampling.advance(previous, [shifted(LEFT, 4), RIGHT], 90, HOLD)

    assert after["mode"] == face_sampling.DENSE


def test_faces_after_an_empty_frame_are_new_arrivals():
    previous = state(face_sampling.SPARSE, [], empty=1)

    assert face_sampling.advance(previous, [LEFT], 90, HOLD)["mode"] == face_sampling.DENSE


@pytest.mark.parametrize("boxes", [[LEFT], [LEFT, RIGHT]])
def test_one_or_two_continuing_faces_are_sparse(boxes):
    previous = state(face_sampling.DENSE, boxes, hold_until=60)

    after = face_sampling.advance(previous, [shifted(b, 4) for b in boxes], 90, HOLD)

    assert after["mode"] == face_sampling.SPARSE


def test_continuing_needs_the_linking_iou():
    previous = state(face_sampling.SPARSE, [LEFT])
    barely_moved = shifted(LEFT, 10)
    moved_away = shifted(LEFT, 40)

    assert face_sampling.advance(previous, [barely_moved], 90, HOLD)["mode"] == face_sampling.SPARSE
    assert face_sampling.advance(previous, [moved_away], 90, HOLD)["mode"] == face_sampling.DENSE


def test_idle_only_after_the_configured_empty_frames():
    current = state(face_sampling.SPARSE, [LEFT])
    modes = []
    for frame_idx in (99, 108, 117, 126):
        current = face_sampling.advance(current, [], frame_idx, HOLD)
        modes.append(current["mode"])

    assert modes == [face_sampling.SPARSE, face_sampling.SPARSE, face_sampling.IDLE,
                     face_sampling.IDLE]


def test_dense_hold_lasts_the_configured_video_time():
    current = face_sampling.advance(state(face_sampling.DENSE), [LEFT], 0, HOLD)
    modes = []
    for frame_idx in range(3, 34, 3):
        current = face_sampling.advance(current, [LEFT], frame_idx, HOLD)
        modes.append((frame_idx, current["mode"]))

    assert all(mode == face_sampling.DENSE for frame_idx, mode in modes if frame_idx < HOLD)
    assert all(mode == face_sampling.SPARSE for frame_idx, mode in modes if frame_idx >= HOLD)


def test_dense_hold_wins_over_empty_frames():
    current = face_sampling.advance(state(face_sampling.DENSE), [LEFT, MIDDLE, RIGHT], 0, HOLD)
    for frame_idx in (3, 6, 9, 12):
        current = face_sampling.advance(current, [], frame_idx, HOLD)

    assert current["mode"] == face_sampling.DENSE
    assert face_sampling.advance(current, [], 30, HOLD)["mode"] == face_sampling.IDLE


def test_simulated_video_switches_between_modes():
    frames = [[] for _ in range(300)]
    for k in range(60, 150):
        frames[k] = [shifted(LEFT, (k - 60) // 3)]
    for k in range(150, 200):
        frames[k] = [shifted(b, (k - 150) // 3) for b in (LEFT, MIDDLE, RIGHT)]

    visited = run(frames)
    mode_at = dict(visited)

    assert visited[:4] == [(0, face_sampling.DENSE), (3, face_sampling.DENSE),
                           (6, face_sampling.DENSE), (36, face_sampling.IDLE)]
    first_seen = min(k for k, _ in visited if frames[k])
    assert all(mode_at[k] == face_sampling.DENSE for k in mode_at if first_seen < k <= first_seen + HOLD)
    assert any(mode == face_sampling.SPARSE for frame_idx, mode in visited if 60 < frame_idx < 150)
    crowd_seen = min(k for k, _ in visited if len(frames[k]) == 3)
    assert all(mode_at[k] == face_sampling.DENSE for k in mode_at if crowd_seen < k < 200)
    assert len(visited) < len(range(0, 300, 3))


def test_fixed_mode_samples_every_fixed_step_whatever_it_finds():
    frames = [[] for _ in range(90)]
    for k in range(30, 60):
        frames[k] = [LEFT, MIDDLE, RIGHT]

    visited = run(frames, adaptive=False)

    assert face_sampling.sampling_steps(FPS, False) == {face_sampling.FIXED: 3}
    assert visited == [(k, face_sampling.FIXED) for k in range(0, 90, 3)]


def test_fixed_interval_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("FACE_INTERVAL_S", "0.2")
    module = importlib.reload(face_sampling)

    assert module.sampling_steps(FPS, False) == {module.FIXED: 6}


def test_adaptive_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("FACE_ADAPTIVE", "0")
    monkeypatch.setenv("FACE_INTERVAL_DENSE_S", "0.2")
    monkeypatch.setenv("FACE_INTERVAL_SPARSE_S", "0.5")
    monkeypatch.setenv("FACE_INTERVAL_IDLE_S", "2")
    monkeypatch.setenv("FACE_CROWD_FACES", "5")
    monkeypatch.setenv("FACE_IDLE_FRAMES", "4")
    monkeypatch.setenv("FACE_DENSE_HOLD_S", "0.5")

    module = importlib.reload(face_sampling)

    assert module.ADAPTIVE is False
    assert module.sampling_steps(FPS, True) == {module.DENSE: 6, module.SPARSE: 15,
                                                module.IDLE: 60}
    assert module.CROWD_FACES == 5
    assert module.IDLE_FRAMES == 4
    assert module.hold_frames(FPS) == 15


@pytest.mark.parametrize("name, value, message", [
    ("FACE_ADAPTIVE", "si", "FACE_ADAPTIVE debe ser 0 o 1"),
    ("FACE_INTERVAL_DENSE_S", "0", "FACE_INTERVAL_DENSE_S debe ser un número positivo"),
    ("FACE_INTERVAL_SPARSE_S", "rapido", "FACE_INTERVAL_SPARSE_S debe ser un número positivo"),
    ("FACE_INTERVAL_IDLE_S", "-1", "FACE_INTERVAL_IDLE_S debe ser un número positivo"),
    ("FACE_CROWD_FACES", "2.5", "FACE_CROWD_FACES debe ser un entero positivo"),
    ("FACE_IDLE_FRAMES", "0", "FACE_IDLE_FRAMES debe ser un entero positivo"),
    ("FACE_DENSE_HOLD_S", "nan", "FACE_DENSE_HOLD_S debe ser un número positivo"),
])
def test_invalid_sampling_settings_stop_with_a_clear_message(monkeypatch, name, value, message):
    monkeypatch.setenv(name, value)

    with pytest.raises(SystemExit, match=message):
        importlib.reload(face_sampling)
