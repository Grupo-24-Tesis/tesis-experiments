import json
import os
import subprocess
import sys

import pytest

PIPELINE_DIR = os.path.abspath(os.environ.get(
    "TESIS_PIPELINE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "tesis_face_recognition_pipeline")))
sys.path.insert(0, PIPELINE_DIR)

import worker

JOBS_QUEUE = "https://sqs.us-east-2.amazonaws.com/1/trabajos"
RESULTS_QUEUE = "https://sqs.us-east-2.amazonaws.com/1/resultados"
FACES = [{"track_id": 4, "face_file": "4/face_of_4.jpg", "time_s": 83.0}]


class FakeSqs:

    def __init__(self, fail_send=False):
        self.sent = []
        self.deleted = []
        self.visibility = []
        self.fail_send = fail_send

    def send_message(self, QueueUrl, MessageBody):
        if self.fail_send:
            raise ConnectionError("sin red")
        self.sent.append((QueueUrl, json.loads(MessageBody)))

    def delete_message(self, QueueUrl, ReceiptHandle):
        self.deleted.append((QueueUrl, ReceiptHandle))

    def change_message_visibility(self, QueueUrl, ReceiptHandle, VisibilityTimeout):
        self.visibility.append((QueueUrl, ReceiptHandle, VisibilityTimeout))

    def events(self):
        return [body["eventType"] for _, body in self.sent]


class FakeS3:

    def __init__(self, missing=False):
        self.missing = missing
        self.uploaded = []

    def download_file(self, bucket, key, path):
        if self.missing:
            error = Exception("no existe")
            error.response = {"Error": {"Code": "NoSuchKey"}}
            raise error
        with open(path, "wb") as f:
            f.write(b"video")

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        self.uploaded.append((bucket, key))


@pytest.fixture(autouse=True)
def settings(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "QUEUE_URL", JOBS_QUEUE)
    monkeypatch.setattr(worker, "RESULTS_QUEUE_URL", RESULTS_QUEUE)
    monkeypatch.setattr(worker, "JOBS_DIR", str(tmp_path))
    monkeypatch.setattr(worker, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(worker, "KEEP_JOB_FILES", False)
    monkeypatch.setattr(worker, "FACES_URL_BASE", "")


@pytest.fixture
def pipeline(monkeypatch):
    calls = []

    def run_pipeline(video_path, job_dir):
        calls.append(video_path)
        return os.path.join(job_dir, "faces"), FACES

    monkeypatch.setattr(worker, "run_pipeline", run_pipeline)
    return calls


def job_message(receive_count=1, **overrides):
    body = {
        "eventType": "VIDEO_UPLOAD_READY",
        "eventVersion": "1",
        "jobId": "job-1",
        "bucket": "cubo",
        "objectKey": "videos/abc.mp4",
        "cameraId": "CAM-01",
        "originalFileName": "Camera_ch6_20261016100358.mp4",
        "createdAt": "2026-10-16T15:10:00Z",
    }
    body.update(overrides)
    body = {k: v for k, v in body.items() if v is not None}
    return {
        "MessageId": "m-1",
        "ReceiptHandle": "recibo-1",
        "Body": json.dumps(body),
        "Attributes": {"ApproximateReceiveCount": str(receive_count)},
    }


def test_send_event_builds_the_envelope():
    sqs = FakeSqs()

    worker.send_event(sqs, worker.EVENT_FAILED, "job-1", reason="sin video")

    queue, body = sqs.sent[0]
    assert queue == RESULTS_QUEUE
    assert body["eventType"] == "VIDEO_PROCESSING_FAILED"
    assert body["eventVersion"] == "1"
    assert body["jobId"] == "job-1"
    assert body["reason"] == "sin video"
    assert body["sentAt"].endswith("Z") and "T" in body["sentAt"]


def test_send_event_rejects_oversized_results():
    appearances = [{"faceId": "x" * 1000}] * 400

    with pytest.raises(worker.JobError):
        worker.send_event(FakeSqs(), worker.EVENT_COMPLETED, "job-1",
                          appearances=appearances)


def test_successful_job_reports_started_and_completed(pipeline):
    sqs, s3 = FakeSqs(), FakeS3()

    worker.handle_message(sqs, s3, job_message())

    assert sqs.events() == ["VIDEO_PROCESSING_STARTED", "VIDEO_PROCESSING_COMPLETED"]
    appearance = sqs.sent[1][1]["appearances"][0]
    assert appearance == {
        "faceId": "job-1_4",
        "timestamp": "00:01:23",
        "frameUrl": "https://cubo.s3.us-east-2.amazonaws.com/faces/job-1/4.jpg",
        "capturedAt": "2026-10-16T10:05:21-05:00",
    }
    assert s3.uploaded == [("cubo", "faces/job-1/4.jpg")]
    assert sqs.deleted == [(JOBS_QUEUE, "recibo-1")]
    assert not os.path.exists(os.path.join(worker.JOBS_DIR, "job-1"))


def test_missing_video_reports_failure_and_drops_the_message(pipeline):
    sqs = FakeSqs()

    worker.handle_message(sqs, FakeS3(missing=True), job_message())

    assert sqs.events() == ["VIDEO_PROCESSING_STARTED", "VIDEO_PROCESSING_FAILED"]
    assert "videos/abc.mp4" in sqs.sent[1][1]["reason"]
    assert sqs.deleted == [(JOBS_QUEUE, "recibo-1")]
    assert pipeline == []


def test_transient_failure_leaves_the_message_for_a_retry(monkeypatch):
    def broken(video_path, job_dir):
        raise worker.StageError("tracking.py terminó con código 1")

    monkeypatch.setattr(worker, "run_pipeline", broken)
    sqs = FakeSqs()

    worker.handle_message(sqs, FakeS3(), job_message(receive_count=1))

    assert sqs.events() == ["VIDEO_PROCESSING_STARTED"]
    assert sqs.deleted == []


def test_last_attempt_reports_failure_and_drops_the_message(monkeypatch):
    def broken(video_path, job_dir):
        raise worker.StageError("tracking.py terminó con código 1")

    monkeypatch.setattr(worker, "run_pipeline", broken)
    sqs = FakeSqs()

    worker.handle_message(sqs, FakeS3(), job_message(receive_count=3))

    assert sqs.events() == ["VIDEO_PROCESSING_STARTED", "VIDEO_PROCESSING_FAILED"]
    reason = sqs.sent[1][1]["reason"]
    assert "3 intentos" in reason and "tracking.py terminó con código 1" in reason
    assert sqs.deleted == [(JOBS_QUEUE, "recibo-1")]
    assert not os.path.exists(os.path.join(worker.JOBS_DIR, "job-1"))


def test_malformed_message_with_job_id_reports_failure():
    sqs = FakeSqs()

    worker.handle_message(sqs, FakeS3(), job_message(objectKey=None))

    assert sqs.events() == ["VIDEO_PROCESSING_FAILED"]
    assert sqs.sent[0][1]["jobId"] == "job-1"
    assert sqs.deleted == [(JOBS_QUEUE, "recibo-1")]


def test_malformed_message_without_job_id_is_dropped():
    sqs = FakeSqs()
    message = job_message()
    message["Body"] = "esto no es json"

    worker.handle_message(sqs, FakeS3(), message)

    assert sqs.sent == []
    assert sqs.deleted == [(JOBS_QUEUE, "recibo-1")]


def test_unsafe_job_id_is_not_reported():
    sqs = FakeSqs()

    worker.handle_message(sqs, FakeS3(), job_message(jobId="../etc"))

    assert sqs.sent == []
    assert sqs.deleted == [(JOBS_QUEUE, "recibo-1")]


def test_message_stays_when_the_failure_cannot_be_reported(pipeline):
    sqs = FakeSqs(fail_send=True)

    worker.handle_message(sqs, FakeS3(missing=True), job_message(receive_count=3))

    assert sqs.deleted == []


def test_results_are_kept_and_resent_when_sending_them_fails(pipeline):
    failing = FakeSqs(fail_send=True)

    worker.handle_message(failing, FakeS3(), job_message())

    job_dir = os.path.join(worker.JOBS_DIR, "job-1")
    assert os.listdir(job_dir) == [worker.PAYLOAD_NAME]
    assert failing.deleted == []

    sqs = FakeSqs()
    worker.handle_message(sqs, FakeS3(), job_message(receive_count=2))

    assert pipeline == [os.path.join(job_dir, "video.mp4")]
    assert sqs.events() == ["VIDEO_PROCESSING_STARTED", "VIDEO_PROCESSING_COMPLETED"]
    assert sqs.sent[1][1]["appearances"][0]["faceId"] == "job-1_4"
    assert sqs.deleted == [(JOBS_QUEUE, "recibo-1")]
    assert not os.path.exists(job_dir)


def test_heartbeat_extends_visibility_and_reports_progress():
    sqs = FakeSqs()

    worker.JobHeartbeat(sqs, "recibo-1", "job-1").beat()

    assert sqs.visibility == [(JOBS_QUEUE, "recibo-1", worker.VISIBILITY_TIMEOUT)]
    assert sqs.events() == ["VIDEO_PROCESSING_STARTED"]


def test_heartbeat_survives_a_failing_queue():
    worker.JobHeartbeat(FakeSqs(fail_send=True), "recibo-1", "job-1").beat()


@pytest.fixture
def stages(monkeypatch):
    calls = []

    def run_stage(python, script, *args):
        calls.append((python, script, list(args)))
        if "--faces-dir" in args:
            faces_dir = args[args.index("--faces-dir") + 1]
            os.makedirs(faces_dir, exist_ok=True)
            with open(os.path.join(faces_dir, "results.json"), "w") as f:
                json.dump({"tracks_checked": 1, "faces": FACES}, f)

    monkeypatch.setattr(worker, "run_stage", run_stage)
    return calls


def test_pipeline_without_gate_runs_only_face_identities(monkeypatch, tmp_path, stages):
    monkeypatch.setattr(worker, "FACE_GATING", False)
    job_dir = str(tmp_path / "job-1")

    faces_dir, faces = worker.run_pipeline("/videos/v.mp4", job_dir)

    assert stages == [(worker.FACES_PYTHON, "face_identities.py",
                       ["--video", "/videos/v.mp4", "--faces-dir", os.path.join(job_dir, "faces")])]
    assert faces_dir == os.path.join(job_dir, "faces")
    assert faces == FACES


def test_pipeline_always_runs_gate_and_face_identities(monkeypatch, tmp_path, stages):
    monkeypatch.setattr(worker, "FACE_GATING", True)
    job_dir = str(tmp_path / "job-1")
    presence = os.path.join(job_dir, "presence.json")

    faces_dir, faces = worker.run_pipeline("/videos/v.mp4", job_dir)

    assert stages == [
        (worker.TRACK_PYTHON, "presence_gate.py",
         ["--video", "/videos/v.mp4", "--out", presence]),
        (worker.FACES_PYTHON, "face_identities.py",
         ["--video", "/videos/v.mp4", "--faces-dir", os.path.join(job_dir, "faces"),
          "--presence", presence]),
    ]
    assert faces == FACES
    assert not {"tracking.py", "face_recognition.py"} & {script for _, script, _ in stages}


def test_a_failing_gate_stops_the_job_before_the_face_pass(monkeypatch, tmp_path):
    calls = []

    def run_stage(python, script, *args):
        calls.append(script)
        raise worker.StageError(f"{script} terminó con código 1")

    monkeypatch.setattr(worker, "FACE_GATING", True)
    monkeypatch.setattr(worker, "run_stage", run_stage)

    with pytest.raises(worker.StageError, match="presence_gate.py"):
        worker.run_pipeline("/videos/v.mp4", str(tmp_path / "job-1"))
    assert calls == ["presence_gate.py"]


def gating_at_start(value, adaptive=None, extra=None):
    env = {k: v for k, v in os.environ.items()
           if k not in ("FACE_GATING", "FACE_ADAPTIVE")
           and not k.startswith("GATE_")}
    env.update(extra or {})
    if value is not None:
        env["FACE_GATING"] = value
    if adaptive is not None:
        env["FACE_ADAPTIVE"] = adaptive
    return subprocess.run([sys.executable, "-c", "import worker; print(worker.FACE_GATING)"],
                          cwd=PIPELINE_DIR, env=env, capture_output=True, text=True)


def test_gating_is_on_by_default():
    assert gating_at_start(None).stdout.strip() == "True"
    assert gating_at_start("").stdout.strip() == "True"


def test_gating_is_read_from_the_environment():
    assert gating_at_start("1").stdout.strip() == "True"
    assert gating_at_start("0").stdout.strip() == "False"


def test_invalid_gating_value_stops_the_worker_with_a_clear_message():
    result = gating_at_start("si")

    assert result.returncode != 0
    assert "FACE_GATING debe ser 0 o 1, no 'si'" in result.stderr


def test_adaptive_sampling_with_gating_stops_the_worker_at_start():
    result = gating_at_start(None, adaptive="1")

    assert result.returncode != 0
    assert "FACE_ADAPTIVE=1" in result.stderr and "FACE_GATING" in result.stderr


def test_adaptive_with_gate_refuses_start():
    for gating in (None, "1"):
        result = gating_at_start(gating, adaptive="1")
        assert result.returncode != 0
        assert "FACE_ADAPTIVE=1" in result.stderr


def test_pipeline_mode_variable_is_ignored():
    env = dict(os.environ, PIPELINE_MODE="tracks")
    out = subprocess.run([sys.executable, "-c",
                          "import worker; print(hasattr(worker, 'PIPELINE_MODE'))"],
                         cwd=PIPELINE_DIR, env=env, capture_output=True, text=True)
    assert out.returncode == 0
    assert out.stdout.strip() == "False"


def test_adaptive_sampling_is_fine_without_gating():
    assert gating_at_start("0", adaptive="1").stdout.strip() == "False"
    assert gating_at_start("1", adaptive="0").stdout.strip() == "True"


@pytest.mark.parametrize("name,value,message", [
    ("GATE_CONF", "1.5", "GATE_CONF debe estar entre 0 y 1, no 1.5"),
    ("GATE_CONF", "abc", "GATE_CONF debe ser un número positivo, no 'abc'"),
    ("GATE_CHECKS_PER_S", "0", "GATE_CHECKS_PER_S debe ser un número positivo"),
    ("GATE_REWIND_S", "-1", "GATE_REWIND_S debe ser un número no negativo"),
    ("GATE_EXIT_S", "nan", "GATE_EXIT_S debe ser un número no negativo"),
    ("GATE_IMGSZ", "640.5", "GATE_IMGSZ debe ser un entero positivo"),
])
def test_invalid_gate_setting_stops_the_worker_at_start(name, value, message):
    result = gating_at_start(None, extra={name: value})

    assert result.returncode != 0
    assert message in result.stderr


def test_gate_settings_are_not_checked_when_the_gate_does_not_run():
    assert gating_at_start("0", extra={"GATE_CONF": "1.5"}).stdout.strip() == "False"


def test_valid_gate_settings_let_the_worker_start():
    extra = {"GATE_CONF": "0.4", "GATE_CHECKS_PER_S": "4", "GATE_IMGSZ": "320"}
    assert gating_at_start(None, extra=extra).stdout.strip() == "True"


def test_worker_imports_no_framework():
    code = ("import sys, worker; "
            "print([m for m in ('numpy', 'cv2', 'torch', 'tensorflow', 'ultralytics') "
            "if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], cwd=PIPELINE_DIR, capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "[]"


def embedding_text(seed):
    import base64
    import numpy as np
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=512).astype("<f4")
    vec = vec / np.linalg.norm(vec)
    return base64.b64encode(vec.tobytes()).decode()


def write_embeddings(faces_dir, track_ids):
    os.makedirs(faces_dir, exist_ok=True)
    with open(os.path.join(faces_dir, "embeddings.json"), "w") as f:
        json.dump({"model": "GhostFaceNet", "dimensions": 512,
                   "embeddings": {t: embedding_text(int(t)) for t in track_ids}}, f)


@pytest.fixture
def embedding_pipeline(monkeypatch):
    def run_pipeline(video_path, job_dir):
        faces_dir = os.path.join(job_dir, "faces")
        write_embeddings(faces_dir, ["4"])
        return faces_dir, FACES

    monkeypatch.setattr(worker, "run_pipeline", run_pipeline)


def fake_appearances(count):
    return [{"faceId": f"job-1_{i}", "timestamp": "00:00:01", "frameUrl": "https://x/" + str(i),
             "capturedAt": "2026-10-16T10:05:21-05:00", "embedding": "A" * 2732}
            for i in range(count)]


def test_completed_carries_embeddings(embedding_pipeline):
    sqs = FakeSqs()

    worker.handle_message(sqs, FakeS3(), job_message())

    body = sqs.sent[1][1]
    assert body["embeddingModel"] == "GhostFaceNet"
    assert body["embeddingDimensions"] == 512
    assert body["part"] == 1 and body["parts"] == 1
    assert body["appearances"][0]["embedding"] == embedding_text(4)


def test_split_parts_respects_limit():
    appearances = fake_appearances(300)

    parts = worker.split_parts(appearances, 400)

    assert len(parts) > 1
    assert [a for part in parts for a in part] == appearances
    for part in parts:
        body = json.dumps({"appearances": part})
        assert 400 + len(body.encode()) < worker.PART_BYTES


def test_parts_share_attempt_and_total():
    sqs = FakeSqs()

    worker.send_completed(sqs, "job-1", "intento-1", fake_appearances(300),
                          "GhostFaceNet", 512)

    bodies = [body for _, body in sqs.sent]
    assert len(bodies) > 1
    assert [b["part"] for b in bodies] == list(range(1, len(bodies) + 1))
    assert {b["parts"] for b in bodies} == {len(bodies)}
    assert {b["attemptId"] for b in bodies} == {"intento-1"}
    assert {b["embeddingModel"] for b in bodies} == {"GhostFaceNet"}
    assert sum(len(b["appearances"]) for b in bodies) == 300


def test_resend_reuses_attempt_id(embedding_pipeline):
    failing = FakeSqs(fail_send=True)
    worker.handle_message(failing, FakeS3(), job_message())
    job_dir = os.path.join(worker.JOBS_DIR, "job-1")
    with open(os.path.join(job_dir, worker.PAYLOAD_NAME)) as f:
        saved = json.load(f)
    assert saved["embeddingModel"] == "GhostFaceNet"
    assert saved["embeddingDimensions"] == 512
    assert saved["appearances"][0]["faceId"] == "job-1_4"

    sqs = FakeSqs()
    worker.handle_message(sqs, FakeS3(), job_message(receive_count=2))

    completed = sqs.sent[1][1]
    assert completed["attemptId"] == saved["attemptId"]
    assert completed["appearances"][0]["embedding"] == embedding_text(4)


def test_legacy_payload_list_still_resent(pipeline):
    job_dir = os.path.join(worker.JOBS_DIR, "job-1")
    os.makedirs(job_dir)
    legacy = [{"faceId": "job-1_4", "timestamp": "00:01:23",
               "frameUrl": "https://cubo.s3.us-east-2.amazonaws.com/faces/job-1/4.jpg",
               "capturedAt": "2026-10-16T10:05:21-05:00"}]
    with open(os.path.join(job_dir, worker.PAYLOAD_NAME), "w") as f:
        json.dump(legacy, f)
    sqs = FakeSqs()

    worker.handle_message(sqs, FakeS3(), job_message(receive_count=2))

    body = sqs.sent[1][1]
    assert body["appearances"] == legacy
    assert body["attemptId"]
    assert body["part"] == 1 and body["parts"] == 1
    assert "embeddingModel" not in body
    assert pipeline == []


def test_earlier_dict_payload_still_resent(pipeline):
    job_dir = os.path.join(worker.JOBS_DIR, "job-1")
    os.makedirs(job_dir)
    with open(os.path.join(job_dir, worker.PAYLOAD_NAME), "w") as f:
        json.dump({"appearances": [{"faceId": "job-1_4"}]}, f)
    sqs = FakeSqs()

    worker.handle_message(sqs, FakeS3(), job_message(receive_count=2))

    body = sqs.sent[1][1]
    assert body["appearances"] == [{"faceId": "job-1_4"}]
    assert body["attemptId"]


def test_missing_embeddings_file_sends_no_embeddings(monkeypatch, stages):
    monkeypatch.setattr(worker, "FACE_GATING", False)
    sqs = FakeSqs()

    worker.handle_message(sqs, FakeS3(), job_message())

    body = sqs.sent[1][1]
    assert "embeddingModel" not in body
    assert "embeddingDimensions" not in body
    assert "embedding" not in body["appearances"][0]
    assert body["attemptId"]


def test_oversized_appearance_fails_in_spanish():
    huge = [{"faceId": "job-1_1", "embedding": "A" * worker.PART_BYTES}]

    with pytest.raises(worker.JobError, match="no cabe"):
        worker.split_parts(huge, 400)


def part_bytes_at_start(value):
    env = {k: v for k, v in os.environ.items() if k != "RESULT_PART_BYTES"}
    if value is not None:
        env["RESULT_PART_BYTES"] = value
    return subprocess.run([sys.executable, "-c", "import worker; print(worker.PART_BYTES)"],
                          cwd=PIPELINE_DIR, env=env, capture_output=True, text=True)


def test_result_part_bytes_default():
    assert part_bytes_at_start(None).stdout.strip() == "245760"
    assert part_bytes_at_start("").stdout.strip() == "245760"
    assert part_bytes_at_start("245760").stdout.strip() == "245760"


def test_result_part_bytes_small_value_splits_more(monkeypatch):
    assert part_bytes_at_start("50000").stdout.strip() == "50000"
    default_parts = worker.split_parts(fake_appearances(300), 400)

    monkeypatch.setattr(worker, "PART_BYTES", 50000)
    small_parts = worker.split_parts(fake_appearances(300), 400)

    assert len(small_parts) > len(default_parts)
    for part in small_parts:
        assert 400 + len(json.dumps({"appearances": part}).encode()) < 50000


@pytest.mark.parametrize("value", ["abc", "0", "-5", "245761", "1.5"])
def test_result_part_bytes_invalid_refuses_start(value):
    result = part_bytes_at_start(value)

    assert result.returncode != 0
    assert "RESULT_PART_BYTES debe ser un entero entre 1 y 245760" in result.stderr
