import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

from harness import runlog

STATIC_DIR = os.path.dirname(os.path.abspath(__file__))
FRAME_RE = re.compile(r"^/frames/([\w\-]+)/(\d{6})\.jpg$")
CROP_RE = re.compile(r"^/crops/([\w\-]+)/([\w\-]+)\.jpg$")
STATE_RE = re.compile(r"^/api/state/([\w\-]+)$")
POST_RE = re.compile(r"^/api/(frame|crop)/([\w\-]+)/([\w\-]+)$")
REF_POSES = ("frontal", "semi")
REFS_PER_VIDEO = 2
POSES = {"frontal", "semi", "profile", "back"}
OCCLUSIONS = {"none", "partial", "total"}
lock = Lock()


def validate_frame(frame):
    problems = []
    for i, face in enumerate(frame["faces"]):
        if len(face.get("box", [])) != 4:
            problems.append(f"rostro {i}: caja inválida")
        if face.get("ignore"):
            continue
        if not face.get("identity"):
            problems.append(f"rostro {i}: sin identidad")
        if face.get("pose") not in POSES:
            problems.append(f"rostro {i}: sin pose")
        if face.get("occlusion") not in OCCLUSIONS:
            problems.append(f"rostro {i}: sin oclusión")
    return problems


def videos_meta():
    return runlog.read_json(runlog.VIDEOS_META, {"videos": {}, "cameras": {}})


def videos():
    meta = videos_meta()
    out = []
    for stem in sorted(os.listdir(runlog.GT_ROOT)):
        doc = runlog.read_json(os.path.join(runlog.GT_ROOT, stem, "frames.json"))
        if not doc:
            continue
        crops = runlog.read_json(os.path.join(runlog.GT_ROOT, stem, "crop_labels.json"), {"items": {}})["items"]
        info = meta["videos"].get(stem, {})
        camera = info.get("camera", stem)
        out.append({"stem": stem, "label": info.get("label", stem), "camera": camera,
                    "id_prefix": meta["cameras"].get(camera, {}).get("id_prefix", "P"),
                    "frames": len(doc["frames"]),
                    "reviewed": sum(1 for f in doc["frames"].values() if f.get("reviewed")),
                    "crops": len(crops), "crops_labeled": sum(1 for c in crops.values() if c.get("label"))})
    return out


def references():
    refs = {}
    for v in videos():
        doc = runlog.read_json(os.path.join(runlog.GT_ROOT, v["stem"], "frames.json"))
        best = {}
        for key, fr in doc["frames"].items():
            if not fr.get("reviewed"):
                continue
            for f in fr["faces"]:
                if f.get("ignore") or not f.get("identity"):
                    continue
                x1, y1, x2, y2 = f["box"]
                best.setdefault(f["identity"], []).append(
                    (f.get("pose") in REF_POSES, (x2 - x1) * (y2 - y1), int(key), f["box"]))
        for pid, cands in best.items():
            cands.sort(reverse=True)
            picked = []
            for _, area, key, box in cands:
                if all(abs(key - k) > 30 for k, _ in picked):
                    picked.append((key, box))
                if len(picked) == REFS_PER_VIDEO:
                    break
            refs.setdefault(pid, []).extend({"video": v["stem"], "frame": k, "box": b} for k, b in picked)
    return refs


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path, ctype):
        if not os.path.isfile(path):
            return self._send(404, {"error": "no encontrado"})
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    @staticmethod
    def _paths(stem):
        d = os.path.join(runlog.GT_ROOT, stem)
        if not os.path.isfile(os.path.join(d, "frames.json")):
            return None
        return (os.path.join(d, "frames.json"), os.path.join(d, "crop_labels.json"),
                os.path.join(d, "labels_audit.jsonl"))

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._file(os.path.join(STATIC_DIR, "index.html"), "text/html; charset=utf-8")
        if path == "/api/videos":
            return self._send(200, {"videos": videos()})
        if path == "/api/identities":
            return self._send(200, {"identities": runlog.read_json(runlog.IDENTITIES, {"identities": []}),
                                    "references": references()})
        m = STATE_RE.match(path)
        if m:
            paths = self._paths(m.group(1))
            if paths is None:
                return self._send(404, {"error": f"video desconocido {m.group(1)}"})
            return self._send(200, {
                "frames": runlog.read_json(paths[0], {"frames": {}}),
                "crops": runlog.read_json(paths[1], {"items": {}}),
            })
        m = FRAME_RE.match(path)
        if m:
            return self._file(os.path.join(runlog.GT_ROOT, m.group(1), "frames", m.group(2) + ".jpg"), "image/jpeg")
        m = CROP_RE.match(path)
        if m:
            return self._file(os.path.join(runlog.GT_ROOT, m.group(1), "crops", m.group(2) + ".jpg"), "image/jpeg")
        self._send(404, {"error": "no encontrado"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        with lock:
            if self.path == "/api/identities":
                before = runlog.read_json(runlog.IDENTITIES, {"identities": []})
                runlog.write_json(runlog.IDENTITIES, body)
                self._audit(os.path.join(runlog.GT_ROOT, "identities_audit.jsonl"), "identities", None, before, body)
                return self._send(200, {"ok": True})
            m = POST_RE.match(self.path)
            paths = self._paths(m.group(2)) if m else None
            if paths is None:
                return self._send(404, {"error": "no encontrado"})
            frames_p, crops_p, audit_p = paths
            kind, key = m.group(1), m.group(3)
            if kind == "frame":
                doc = runlog.read_json(frames_p)
                if doc is None or key not in doc["frames"]:
                    return self._send(404, {"error": f"fotograma desconocido {key}"})
                frame = body["frame"]
                if frame.get("reviewed"):
                    problems = validate_frame(frame)
                    if problems:
                        return self._send(400, {"error": "; ".join(problems)})
                    for face in frame["faces"]:
                        face["source"] = "human"
                        face.pop("suggested", None)
                    frame["reviewed_utc"] = runlog.utc_now()
                before = doc["frames"][key]
                doc["frames"][key] = frame
                runlog.write_json(frames_p, doc)
                self._audit(audit_p, "frame", key, before, frame)
                return self._send(200, {"ok": True, "frame": frame})
            if kind == "crop":
                doc = runlog.read_json(crops_p, {"items": {}})
                if key not in doc["items"]:
                    return self._send(404, {"error": f"recorte desconocido {key}"})
                before = dict(doc["items"][key])
                doc["items"][key].update(label=body["label"], labeled_utc=runlog.utc_now())
                runlog.write_json(crops_p, doc)
                self._audit(audit_p, "crop", key, before, doc["items"][key])
                return self._send(200, {"ok": True})
        self._send(404, {"error": "no encontrado"})

    @staticmethod
    def _audit(path, kind, key, before, after):
        with open(path, "a") as f:
            f.write(json.dumps({"ts": runlog.utc_now(), "kind": kind, "key": key,
                                "before": before, "after": after}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"Etiquetando {[v['stem'] for v in videos()]} en http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
