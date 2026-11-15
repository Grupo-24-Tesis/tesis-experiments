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


def test_faces_mode_runs_face_identities_with_the_faces_interpreter(monkeypatch, tmp_path, stages):
    monkeypatch.setattr(worker, "PIPELINE_MODE", "faces")
    job_dir = str(tmp_path / "job-1")

    faces_dir, faces = worker.run_pipeline("/videos/v.mp4", job_dir)

    assert stages == [(worker.FACES_PYTHON, "face_identities.py",
                       ["--video", "/videos/v.mp4", "--faces-dir", os.path.join(job_dir, "faces")])]
    assert faces_dir == os.path.join(job_dir, "faces")
    assert faces == FACES


def test_tracks_mode_runs_the_two_stages(monkeypatch, tmp_path, stages):
    monkeypatch.setattr(worker, "PIPELINE_MODE", "tracks")
    job_dir = str(tmp_path / "job-1")
    candidates_dir = os.path.join(job_dir, "candidates")
    faces_dir = os.path.join(job_dir, "faces")

    result = worker.run_pipeline("/videos/v.mp4", job_dir)

    assert stages == [
        (worker.TRACK_PYTHON, "tracking.py",
         ["--video", "/videos/v.mp4", "--candidates-dir", candidates_dir]),
        (worker.FACES_PYTHON, "face_recognition.py",
         ["--candidates-dir", candidates_dir, "--faces-dir", faces_dir]),
    ]
    assert result == (faces_dir, FACES)


def mode_at_start(value):
    env = {k: v for k, v in os.environ.items() if k != "PIPELINE_MODE"}
    if value is not None:
        env["PIPELINE_MODE"] = value
    out = subprocess.run([sys.executable, "-c", "import worker; print(worker.PIPELINE_MODE)"],
                         cwd=PIPELINE_DIR, env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_faces_is_the_default_mode():
    assert mode_at_start(None) == "faces"


def test_empty_mode_falls_back_to_faces():
    assert mode_at_start("") == "faces"


def test_mode_is_read_from_the_environment():
    assert mode_at_start("tracks") == "tracks"


def test_unknown_mode_stops_the_worker_before_any_work(monkeypatch, stages):
    monkeypatch.setattr(worker, "PIPELINE_MODE", "personas")

    with pytest.raises(SystemExit, match="PIPELINE_MODE desconocido 'personas'"):
        worker.run_local("/videos/v.mp4", "job-1")
    with pytest.raises(SystemExit, match="PIPELINE_MODE desconocido 'personas'"):
        worker.poll_forever()
    assert stages == []
    assert not os.path.exists(os.path.join(worker.JOBS_DIR, "job-1"))
