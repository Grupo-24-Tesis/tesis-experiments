import hashlib
import json
import os
import platform
import resource
import socket
import subprocess
import time
from datetime import datetime, timezone
from importlib import metadata

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.abspath(os.environ.get(
    "TESIS_PIPELINE_DIR", os.path.join(EXP_DIR, "..", "tesis_face_recognition_pipeline")))
RUNS_DIR = os.path.join(EXP_DIR, "runs")
CACHE_DIR = os.path.join(EXP_DIR, ".cache")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
GT_ROOT = os.path.join(EXP_DIR, "ground_truth")
IDENTITIES = os.path.join(GT_ROOT, "identities.json")
VIDEOS_META = os.path.join(GT_ROOT, "videos.json")
CONFIG_DIR = os.path.join(EXP_DIR, "configs")
KEY_PACKAGES = ["torch", "torchvision", "ultralytics", "onnxruntime-gpu", "onnxruntime",
                "opencv-python-headless", "opencv-python", "numpy", "tensorflow", "tf_keras",
                "retina-face", "deepface", "mtcnn", "scikit-learn", "pandas", "boto3"]


def isolated_env(extra=None):
    env = dict(os.environ)
    env.update({
        "DEEPFACE_HOME": os.path.join(CACHE_DIR, "deepface"),
        "YOLO_CONFIG_DIR": os.path.join(CACHE_DIR, "ultralytics"),
        "MPLCONFIGDIR": os.path.join(CACHE_DIR, "matplotlib"),
        "PIP_CACHE_DIR": os.path.join(CACHE_DIR, "pip"),
        "XDG_CACHE_HOME": os.path.join(CACHE_DIR, "xdg"),
        "TORCH_HOME": os.path.join(CACHE_DIR, "torch"),
        "KERAS_HOME": os.path.join(CACHE_DIR, "keras"),
        "YOLO_AUTOINSTALL": "false",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": EXP_DIR,
        "TESIS_PIPELINE_DIR": PIPELINE_DIR,
    })
    for key in ("DEEPFACE_HOME", "YOLO_CONFIG_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME", "TORCH_HOME",
                "KERAS_HOME"):
        os.makedirs(env[key], exist_ok=True)
    if extra:
        env.update(extra)
    return env


def exp_path(path):
    return path if os.path.isabs(path) else os.path.join(EXP_DIR, path)


def rel_exp(path):
    return os.path.relpath(os.path.abspath(path), EXP_DIR)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def config_hash(cfg) -> str:
    canon = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canon.encode()).hexdigest()


def new_run_id(cfg) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{config_hash(cfg)[:8]}"


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=False, default=str)
    os.replace(tmp, path)


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


class EventLog:

    def __init__(self, path, stage):
        self.path = path
        self.stage = stage
        self._f = open(path, "a", buffering=1)

    def emit(self, event, **fields):
        rec = {"ts": utc_now(), "stage": self.stage, "event": event, **fields}
        self._f.write(json.dumps(rec, default=str) + "\n")

    def close(self):
        self._f.close()


def file_sha256(path) -> str:
    st = os.stat(path)
    cache_path = os.path.join(CACHE_DIR, "sha256.json")
    cache = read_json(cache_path, {})
    key = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"
    if key not in cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        cache[key] = h.hexdigest()
        os.makedirs(CACHE_DIR, exist_ok=True)
        write_json(cache_path, cache)
    return cache[key]


def video_info(path):
    path = exp_path(path)
    info = {"path": rel_exp(path), "sha256": file_sha256(path)}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_read_packets",
             "-show_entries", "format=duration", "-of", "json", path],
            capture_output=True, text=True, timeout=120, check=True)
        probe = json.loads(out.stdout)
        st = probe["streams"][0]
        num, den = st["r_frame_rate"].split("/")
        info.update({"codec": st.get("codec_name"), "width": st.get("width"),
                     "height": st.get("height"), "fps": float(num) / float(den),
                     "frames": int(st.get("nb_read_packets", 0)),
                     "duration_s": float(probe["format"]["duration"])})
    except (OSError, subprocess.SubprocessError, KeyError, ValueError) as e:
        info["probe_error"] = str(e)
    return info


def git_info():
    def repo(path):
        def git(*args):
            return subprocess.run(["git", "-C", path, *args], capture_output=True,
                                  text=True).stdout.strip()
        return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    return {"pipeline": repo(PIPELINE_DIR), "experiments": repo(EXP_DIR)}


def gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=20, check=True).stdout.strip()
        name, driver, mem = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"gpu": name, "driver": driver, "gpu_memory": mem}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"gpu": None}


def package_versions():
    versions = {}
    for name in KEY_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def env_info():
    return {"host": socket.gethostname(), "python": platform.python_version(),
            "packages": package_versions(), **gpu_info()}


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
