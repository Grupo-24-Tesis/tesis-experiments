import argparse
import copy
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

import yaml

from harness import runlog

CONFIG_DIR = runlog.CONFIG_DIR
CACHE_ROOT = os.path.join(runlog.RUNS_DIR, "_cache")
PY = {"track": os.path.join(runlog.PIPELINE_DIR, "venvs", "track", "bin", "python"),
      "faces": os.path.join(runlog.PIPELINE_DIR, "venvs", "faces", "bin", "python"),
      "eval": os.path.join(runlog.EXP_DIR, "venvs", "eval", "bin", "python")}
STAGE_TIMEOUT_S = 7200
HARNESS_VERSION = 2


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def get_dotted(cfg, dotted):
    node = cfg
    for k in dotted.split("."):
        node = node[k]
    return node


def expand(suite, base):
    points = []
    for p in suite.get("points", []):
        p = dict(p)
        points.append((p.pop("label", "point"), p))
    for key, values in (suite.get("one_at_a_time") or {}).items():
        for v in values:
            points.append((f"{key}={v}", {key: v}))
    groups = [{"label": "", "set": {}, "grid": suite["grid"]}] if suite.get("grid") else []
    groups += suite.get("groups") or []
    for g in groups:
        keys = list(g.get("grid") or {})
        for combo in itertools.product(*(g["grid"][k] for k in keys)):
            over = dict(g.get("set") or {}, **dict(zip(keys, combo)))
            name = ",".join(f"{k.rsplit('.', 1)[-1]}={v}" for k, v in zip(keys, combo))
            points.append((f"{g['label']} {name}".strip(), over))
    if not points:
        points.append(("baseline", {}))
    resolved, seen = [], set()
    for label, over in points:
        cfg = deep_merge(base, suite.get("fixed"))
        for k, v in over.items():
            set_dotted(cfg, k, v)
        h = runlog.config_hash(cfg)
        if h in seen:
            continue
        seen.add(h)
        resolved.append((label, over, cfg))
    return resolved


def run_logged(cmd, log_prefix, env, timeout=STAGE_TIMEOUT_S):
    t0 = time.perf_counter()
    with open(log_prefix + ".stdout.log", "w") as out, open(log_prefix + ".stderr.log", "w") as err:
        try:
            rc = subprocess.run(cmd, stdout=out, stderr=err, env=env, cwd=runlog.PIPELINE_DIR,
                                timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    return rc, round(time.perf_counter() - t0, 2)


def stderr_flags(path):
    flags = {}
    if os.path.exists(path):
        text = open(path, errors="replace").read()
        if "Failed to create CUDAExecutionProvider" in text:
            flags["reid_device"] = "CPU (no cargó el proveedor CUDA de ONNX)"
        elif "CUDAExecutionProvider" in text:
            flags["reid_device"] = "CUDA"
        if "Skipping registering GPU devices" in text:
            flags["tf_device"] = "CPU (faltan las librerías GPU de TensorFlow)"
        if "Traceback (most recent call last)" in text:
            flags["traceback"] = True
    return flags


def cached_stage(kind, key_cfg, build, env):
    h = runlog.config_hash(key_cfg)[:16]
    d = os.path.join(CACHE_ROOT, kind, h)
    done = os.path.join(d, "done.json")
    if os.path.exists(done):
        info = runlog.read_json(done)
        if info.get("status") == "ok":
            return d, h, info, True
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    runlog.write_json(os.path.join(d, "config.json"), key_cfg)
    info = build(d, env)
    info["created_utc"] = runlog.utc_now()
    runlog.write_json(done, info)
    return d, h, info, False


def build_stage1(cfg_path):
    def build(d, env):
        rc, t = run_logged([PY["track"], "-m", "harness.stage1_runner", "--config", cfg_path,
                            "--run-dir", d, "--out", os.path.join(d, "candidates")],
                           os.path.join(d, "stage1"), env)
        return {"status": "ok" if rc == 0 else "failed", "exit_code": rc, "elapsed_s": t,
                **stderr_flags(os.path.join(d, "stage1.stderr.log"))}
    return build


def build_stage2(cfg_path, candidates):
    def build(d, env):
        rc, t = run_logged([PY["faces"], "-m", "harness.stage2_runner", "--config", cfg_path,
                            "--run-dir", d, "--candidates", candidates, "--out", os.path.join(d, "faces")],
                           os.path.join(d, "stage2"), env)
        return {"status": "ok" if rc == 0 else "failed", "exit_code": rc, "elapsed_s": t,
                **stderr_flags(os.path.join(d, "stage2.stderr.log"))}
    return build


def copy_events(src_dir, dst_events, cached):
    src = os.path.join(src_dir, "events.jsonl")
    if not os.path.exists(src):
        return
    with open(src) as s, open(dst_events, "a") as d:
        for line in s:
            rec = json.loads(line)
            rec["cached"] = cached
            d.write(json.dumps(rec) + "\n")


def link(target, name):
    if os.path.lexists(name):
        os.remove(name)
    os.symlink(os.path.relpath(target, os.path.dirname(name)), name)


def run_pipeline_point(test_id, label, over, cfg, repeat):
    key = dict(cfg, _repeat=repeat) if repeat else cfg
    run_id = runlog.new_run_id(key)
    run_dir = os.path.join(runlog.RUNS_DIR, test_id, run_id)
    os.makedirs(os.path.join(run_dir, "outputs"), exist_ok=True)
    cfg_path = os.path.join(run_dir, "config.json")
    runlog.write_json(cfg_path, cfg)
    env = runlog.isolated_env()
    video = runlog.video_info(cfg["video"])
    meta = {"run_id": run_id, "test_id": test_id, "kind": "pipeline", "label": label, "overrides": over,
            "repeat": repeat, "status": "running", "started_utc": runlog.utc_now(),
            "config_hash": runlog.config_hash(key), "git": runlog.git_info(), "video": video,
            "env": runlog.env_info(), "parents": {}, "timing_s": {}, "exit_codes": {},
            "devices_used": {}, "cache_hits": {}}
    runlog.write_json(os.path.join(run_dir, "run.json"), meta)
    events = os.path.join(run_dir, "events.jsonl")
    try:
        k1 = {"video_sha256": video["sha256"], "stage1": cfg["stage1"], "tracker": cfg["tracker"],
              "_repeat": repeat, "_harness": HARNESS_VERSION}
        d1, h1, i1, hit1 = cached_stage("stage1", k1, build_stage1(cfg_path), env)
        meta["parents"]["stage1"] = h1
        meta["cache_hits"]["stage1"] = hit1
        meta["exit_codes"]["stage1"] = i1["exit_code"]
        meta["timing_s"]["stage1"] = i1["elapsed_s"]
        meta["devices_used"]["reid"] = i1.get("reid_device")
        s1 = runlog.read_json(os.path.join(d1, "candidates", "stage1_summary.json"), {})
        meta["devices_used"]["yolo"] = s1.get("device")
        copy_events(d1, events, hit1)
        link(os.path.join(d1, "candidates"), os.path.join(run_dir, "outputs", "candidates"))
        for ext in ("stdout", "stderr"):
            link(os.path.join(d1, f"stage1.{ext}.log"), os.path.join(run_dir, f"stage1.{ext}.log"))
        if i1["status"] != "ok":
            raise RuntimeError(f"stage1 falló (código {i1['exit_code']})")

        k2 = {"stage1": h1, "stage2": cfg["stage2"], "_harness": HARNESS_VERSION}
        d2, h2, i2, hit2 = cached_stage("stage2", k2, build_stage2(cfg_path, os.path.join(d1, "candidates")), env)
        meta["parents"]["stage2"] = h2
        meta["cache_hits"]["stage2"] = hit2
        meta["exit_codes"]["stage2"] = i2["exit_code"]
        meta["timing_s"]["stage2"] = i2["elapsed_s"]
        meta["devices_used"]["retinaface"] = i2.get("tf_device", "GPU")
        copy_events(d2, events, hit2)
        link(os.path.join(d2, "faces"), os.path.join(run_dir, "outputs", "faces"))
        for ext in ("stdout", "stderr"):
            link(os.path.join(d2, f"stage2.{ext}.log"), os.path.join(run_dir, f"stage2.{ext}.log"))
        if i2["status"] != "ok":
            raise RuntimeError(f"stage2 falló (código {i2['exit_code']})")
        meta["status"] = "ok"
    except Exception as e:
        meta["status"] = "timeout" if "timeout" in str(e) else "failed"
        meta["error"] = traceback.format_exc()
    meta["timing_s"]["total"] = round(sum(v for v in meta["timing_s"].values() if v), 2)
    meta["finished_utc"] = runlog.utc_now()
    runlog.write_json(os.path.join(run_dir, "run.json"), meta)
    return run_dir, meta


def run_detect_point(test_id, label, over, cfg, repeat):
    gt_path = os.path.join(runlog.GT_ROOT,
                           os.path.splitext(os.path.basename(cfg["video"]))[0], "frames.json")
    run_id = runlog.new_run_id(dict(cfg, _repeat=repeat) if repeat else cfg)
    run_dir = os.path.join(runlog.RUNS_DIR, test_id, run_id)
    os.makedirs(os.path.join(run_dir, "outputs"), exist_ok=True)
    cfg_path = os.path.join(run_dir, "config.json")
    runlog.write_json(cfg_path, cfg)
    d = cfg["detect"]
    video = runlog.video_info(cfg["video"])
    meta = {"run_id": run_id, "test_id": test_id, "kind": "detect", "label": label, "overrides": over,
            "repeat": repeat, "status": "running", "started_utc": runlog.utc_now(),
            "config_hash": runlog.config_hash(cfg), "git": runlog.git_info(), "video": video,
            "env": runlog.env_info(), "parents": {}, "timing_s": {}, "exit_codes": {}, "cache_hits": {},
            "devices_used": {}}
    env = runlog.isolated_env()
    try:
        frame_ids = sorted(int(k) for k in runlog.read_json(gt_path)["frames"])
        base_key = {"video_sha256": video["sha256"], "frames": runlog.config_hash(frame_ids),
                    "detect": d, "_harness": HARNESS_VERSION}
        extra = []
        if d["detector"] == "pipeline":
            if d["frame_upscale"] != 1:
                raise ValueError("detector=pipeline trabaja sobre cajas de persona en píxeles del fotograma: frame_upscale debe ser 1")
            yolo = {k: cfg["stage1"][k] for k in ("YOLO_WEIGHTS", "YOLO_CONF", "YOLO_IOU", "YOLO_IMGSZ")}

            def build_boxes(cd, env):
                rc, t = run_logged([PY["track"], "-m", "harness.person_boxes", "--config", cfg_path,
                                    "--gt", gt_path, "--out", os.path.join(cd, "person_boxes.json")],
                                   os.path.join(cd, "person_boxes"), env)
                return {"status": "ok" if rc == 0 else "failed", "exit_code": rc, "elapsed_s": t}
            pd_, ph, pi, phit = cached_stage("person_boxes", dict(base_key, detect=None, yolo=yolo),
                                             build_boxes, env)
            meta["parents"]["person_boxes"] = ph
            meta["cache_hits"]["person_boxes"] = phit
            meta["timing_s"]["person_boxes"] = pi["elapsed_s"]
            if pi["status"] != "ok":
                raise RuntimeError(f"person_boxes falló (código {pi['exit_code']})")
            base_key.update(yolo=yolo, stage2=cfg["stage2"])
            extra = ["--config", cfg_path, "--person-boxes", os.path.join(pd_, "person_boxes.json")]

        def build_dets(cd, env):
            cmd = [PY["faces"], "-m", "harness.detect_frames", "--video", runlog.exp_path(cfg["video"]),
                   "--out", cd, "--gt", gt_path, "--detector", d["detector"],
                   "--threshold", str(d["threshold"]), "--allow-upscaling", str(d["allow_upscaling"]).lower(),
                   "--upscale", str(d["frame_upscale"]), "--mtcnn-min-face", str(d.get("mtcnn_min_face", 20))] + extra
            rc, t = run_logged(cmd, os.path.join(cd, "detect"), env)
            return {"status": "ok" if rc == 0 else ("timeout" if rc == "timeout" else "failed"),
                    "exit_code": rc, "elapsed_s": t, **stderr_flags(os.path.join(cd, "detect.stderr.log"))}
        cd, h, info, hit = cached_stage("detect", base_key, build_dets, env)
        meta["parents"]["detect"] = h
        meta["cache_hits"]["detect"] = hit
        meta["exit_codes"]["detect"] = info["exit_code"]
        meta["timing_s"]["detect"] = info["elapsed_s"]
        meta["devices_used"]["detector"] = info.get("tf_device", "GPU")
        copy_events(cd, os.path.join(run_dir, "events.jsonl"), hit)
        link(os.path.join(cd, "outputs", "detections.json"), os.path.join(run_dir, "outputs", "detections.json"))
        for ext in ("stdout", "stderr"):
            link(os.path.join(cd, f"detect.{ext}.log"), os.path.join(run_dir, f"detect.{ext}.log"))
        if info["status"] != "ok":
            raise RuntimeError(f"detección {info['status']} (código {info['exit_code']})")
        meta["status"] = "ok"
    except Exception:
        meta["status"] = "failed"
        meta["error"] = traceback.format_exc()
    meta["timing_s"]["total"] = round(sum(v for v in meta["timing_s"].values() if v), 2)
    meta["finished_utc"] = runlog.utc_now()
    runlog.write_json(os.path.join(run_dir, "run.json"), meta)
    return run_dir, meta


def build_embed(items_path, e):
    def build(d, env):
        rc, t = run_logged([PY["faces"], "-m", "harness.embed", "--items", items_path,
                            "--out", d, "--model", e["model"], "--align", str(e["align"]).lower(),
                            "--normalization", e["normalization"]],
                           os.path.join(d, "embed"), env)
        return {"status": "ok" if rc == 0 else "failed", "exit_code": rc, "elapsed_s": t,
                **stderr_flags(os.path.join(d, "embed.stderr.log"))}
    return build


def run_embed_point(test_id, label, over, cfg, repeat, kind="embed"):
    from harness import items as items_mod
    run_id = runlog.new_run_id(cfg)
    run_dir = os.path.join(runlog.RUNS_DIR, test_id, run_id)
    os.makedirs(os.path.join(run_dir, "outputs"), exist_ok=True)
    runlog.write_json(os.path.join(run_dir, "config.json"), cfg)
    video = runlog.video_info(cfg["video"])
    meta = {"run_id": run_id, "test_id": test_id, "kind": kind, "label": label, "overrides": over,
            "repeat": repeat, "status": "running", "started_utc": runlog.utc_now(),
            "config_hash": runlog.config_hash(cfg), "git": runlog.git_info(), "video": video,
            "env": runlog.env_info(), "parents": {}, "timing_s": {}, "exit_codes": {}, "cache_hits": {}}
    try:
        its, source = items_mod.build(cfg)
        if not its:
            raise RuntimeError(f"no hay rostros para la fuente {source!r}: primero etiquete el ground truth")
        items_path = os.path.join(run_dir, "outputs", "items.json")
        runlog.write_json(items_path, its)
        meta["n_items"] = len(its)
        key = {"items": runlog.config_hash(its), "embed": {k: cfg["embed"][k] for k in ("model", "align", "normalization")},
               "_harness": HARNESS_VERSION}
        d, h, info, hit = cached_stage("embed", key, build_embed(items_path, cfg["embed"]), runlog.isolated_env())
        meta["parents"]["embed"] = h
        meta["cache_hits"]["embed"] = hit
        meta["exit_codes"]["embed"] = info["exit_code"]
        meta["timing_s"]["embed"] = info["elapsed_s"]
        meta["devices_used"] = {"deepface": info.get("tf_device", "GPU")}
        copy_events(d, os.path.join(run_dir, "events.jsonl"), hit)
        link(d, os.path.join(run_dir, "outputs", "embeddings"))
        if info["status"] != "ok":
            raise RuntimeError(f"embed falló (código {info['exit_code']})")
        meta["status"] = "ok"
    except Exception:
        meta["status"] = "failed"
        meta["error"] = traceback.format_exc()
    meta["timing_s"]["total"] = round(sum(v for v in meta["timing_s"].values() if v), 2)
    meta["finished_utc"] = runlog.utc_now()
    runlog.write_json(os.path.join(run_dir, "run.json"), meta)
    return run_dir, meta


def run_cluster_point(test_id, label, over, cfg, repeat):
    return run_embed_point(test_id, label, over, cfg, repeat, kind="cluster")


def run_xvideo_point(test_id, label, over, cfg, repeat):
    run_id = runlog.new_run_id(cfg)
    run_dir = os.path.join(runlog.RUNS_DIR, test_id, run_id)
    os.makedirs(os.path.join(run_dir, "outputs"), exist_ok=True)
    runlog.write_json(os.path.join(run_dir, "config.json"), cfg)
    meta = {"run_id": run_id, "test_id": test_id, "kind": "xvideo", "label": label, "overrides": over,
            "repeat": repeat, "status": "running", "started_utc": runlog.utc_now(),
            "config_hash": runlog.config_hash(cfg), "git": runlog.git_info(),
            "video": runlog.video_info(cfg["video"]), "env": runlog.env_info(), "parents": {},
            "timing_s": {}, "exit_codes": {}, "cache_hits": {}, "videos": []}
    events = runlog.EventLog(os.path.join(run_dir, "events.jsonl"), "xvideo")
    try:
        for ref in cfg["xvideo"]["runs"]:
            src = runlog.read_json(os.path.join(runlog.exp_path(ref), "run.json"))
            if not src or src.get("status") != "ok" or src.get("kind") != "cluster":
                raise RuntimeError(f"{ref} no es una ejecución de agrupación con estado ok")
            meta["parents"][ref] = src["run_id"]
            meta["videos"].append(src["video"]["path"])
            events.emit("source", run=ref, video=src["video"]["path"], label=src.get("label"))
        meta["status"] = "ok"
    except Exception:
        meta["status"] = "failed"
        meta["error"] = traceback.format_exc()
    events.close()
    meta["timing_s"]["total"] = 0.0
    meta["finished_utc"] = runlog.utc_now()
    runlog.write_json(os.path.join(run_dir, "run.json"), meta)
    return run_dir, meta


RUNNERS = {"pipeline": run_pipeline_point, "detect": run_detect_point,
           "embed": run_embed_point, "cluster": run_cluster_point, "xvideo": run_xvideo_point}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", help="ejecuta solo los puntos cuya etiqueta contiene este texto")
    args = ap.parse_args()
    with open(os.path.join(CONFIG_DIR, "base.yaml")) as f:
        base = yaml.safe_load(f)
    with open(args.suite) as f:
        suite = yaml.safe_load(f)
    test_id, kind = suite["test_id"], suite.get("kind", "pipeline")
    points = expand(suite, base)
    if args.only:
        points = [p for p in points if args.only in p[0]]
    repeats = int(suite.get("repeats", 1))
    print(f"{test_id}: {len(points)} puntos x {repeats} repeticiones ({kind})", flush=True)
    if args.dry_run:
        for label, over, _ in points:
            print(f"  {label}: {over}")
        return
    os.makedirs(CACHE_ROOT, exist_ok=True)
    for label, over, cfg in points:
        for r in range(repeats):
            run_dir, meta = RUNNERS[kind](test_id, label, over, cfg, r)
            print(f"  [{meta['status']}] {label} r{r} -> {runlog.rel_exp(run_dir)} "
                  f"({meta['timing_s'].get('total')}s)", flush=True)
    subprocess.run([sys.executable, "-m", "harness.evaluate", "--test-id", test_id],
                   cwd=runlog.EXP_DIR)


if __name__ == "__main__":
    main()
