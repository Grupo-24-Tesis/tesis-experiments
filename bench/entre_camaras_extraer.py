import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import numpy as np

from bench import entre_camaras

DEFAULT_CHOKEPOINT = "/data/TesisFaceRecognition/chokepoint"
DEFAULT_PIPELINE = "/data/TesisFaceRecognition/tesis_face_recognition_pipeline"
SOURCE_RUNS = ("compuerta_defecto", "compuerta_nuevos")
SOURCE_FILES = ("presence.json", "identities.json")


def chokepoint_videos(chokepoint_dir):
    videos = {name[:-4] for name in os.listdir(os.path.join(chokepoint_dir, "videos"))
              if name.endswith(".mp4")}
    groundtruth = {name[:-4] for name in os.listdir(os.path.join(chokepoint_dir, "groundtruth"))
                   if name.endswith(".xml")}
    return sorted(videos & groundtruth)


def source_run(name, gating_root):
    for run in SOURCE_RUNS:
        folder = os.path.join(gating_root, run, name)
        if all(os.path.exists(os.path.join(folder, file)) for file in SOURCE_FILES):
            return folder
    return None


def plan_videos(names, gating_root, out_root):
    sources = {name: source_run(name, gating_root) for name in names}
    missing = [name for name, source in sources.items() if source is None]
    if missing:
        raise SystemExit("Faltan presence.json o identities.json en las ejecuciones con compuerta "
                         f"({', '.join(SOURCE_RUNS)}) de {len(missing)} videos: "
                         f"{', '.join(missing)}")
    return [(name, sources[name]) for name in names
            if not os.path.exists(os.path.join(out_root, name, entre_camaras.EXTRACTION_NAME))]


def check_settings(adaptive, refusal):
    message = refusal(True, adaptive)
    if message is not None:
        raise SystemExit(message)


def face_signature(faces):
    return [[face["frame"], [int(value) for value in face["box"]]] for face in faces]


def compare_identities(stored, fresh):
    stored_by_id = {item["id"]: item for item in stored["identities"]}
    fresh_by_id = {item["id"]: item for item in fresh}
    different = sorted(identity_id for identity_id in stored_by_id.keys() & fresh_by_id.keys()
                       if face_signature(stored_by_id[identity_id]["faces"])
                       != face_signature(fresh_by_id[identity_id]["faces"]))
    same_ids = stored_by_id.keys() == fresh_by_id.keys()
    return {"coincide": same_ids and not different, "guardadas": len(stored_by_id),
            "nuevas": len(fresh_by_id), "distintas": different}


def identity_arrays(identities, best_rows):
    faces = [face for group in identities for face in group]
    identity = [identity_id for identity_id, group in enumerate(identities, start=1)
                for _ in group]
    return {
        "embeddings": np.stack([face["embedding"] for face in faces]).astype(np.float32),
        "identity": np.array(identity, dtype=np.int64),
        "frame": np.array([face["frame"] for face in faces], dtype=np.int64),
        "box": np.array([[int(value) for value in face["box"]] for face in faces],
                        dtype=np.int64).reshape(-1, 4),
        "best": np.array(best_rows, dtype=np.int64),
    }


def write_json(path, document):
    with open(path, "w") as f:
        json.dump(document, f, indent=2)


def write_video(out_dir, arrays, identities_doc, summary):
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, entre_camaras.EMBEDDINGS_NAME), **arrays)
    write_json(os.path.join(out_dir, entre_camaras.IDENTITIES_NAME), identities_doc)
    write_json(os.path.join(out_dir, entre_camaras.EXTRACTION_NAME), summary)


def best_rows_of(identities, ranking):
    rows, offset = [], 0
    for group in identities:
        best = max(range(len(group)), key=lambda index: ranking(group[index]))
        rows.append(offset + best)
        offset += len(group)
    return rows


def best_frames_match(best_frames, results_path):
    with open(results_path) as f:
        stored = {item["track_id"]: item["frame_idx"] for item in json.load(f)["faces"]}
    return all(stored.get(str(identity_id)) == frame
               for identity_id, frame in enumerate(best_frames, start=1))


def run_video(name, source, chokepoint_dir, out_root):
    import cv2
    import face_identities
    import identity_linking
    import presence_intervals

    started = time.monotonic()
    cap = face_identities.open_video(os.path.join(chokepoint_dir, "videos", f"{name}.mp4"))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        presence = presence_intervals.load_presence(os.path.join(source, "presence.json"), fps)
        detections, _, _ = face_identities.analyse_video(cap, fps, presence["intervals"])
    finally:
        cap.release()
    identities = identity_linking.link_identities(detections, by_time=False)
    best_rows = best_rows_of(identities, face_identities.ranking)
    arrays = identity_arrays(identities, best_rows)
    records = [face_identities.identity_record(identity_id, group, fps)
               for identity_id, group in enumerate(identities, start=1)]
    with open(os.path.join(source, "identities.json")) as f:
        comparison = compare_identities(json.load(f), records)
    best_frames = [int(arrays["frame"][row]) for row in best_rows]
    summary = {
        "video": name,
        "fuente": source,
        "identidades": len(identities),
        "rostros": int(len(arrays["identity"])),
        "coincide": comparison["coincide"],
        "mejor_coincide": best_frames_match(best_frames, os.path.join(source, "results.json")),
        "tiempo_s": round(time.monotonic() - started, 1),
        "comparacion": comparison,
    }
    write_video(os.path.join(out_root, name), arrays, {"identities": [
        {"id": record["id"], "faces": [{"frame": face["frame"], "box": face["box"]}
                                       for face in record["faces"]]}
        for record in records]}, summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrae embeddings por identidad de los videos de ChokePoint")
    parser.add_argument("--chokepoint", default=DEFAULT_CHOKEPOINT)
    parser.add_argument("--out", default=None)
    parser.add_argument("--videos", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline_dir = os.environ.get("TESIS_PIPELINE_DIR", DEFAULT_PIPELINE)
    sys.path.insert(0, pipeline_dir)
    import face_sampling
    import presence_intervals

    check_settings(face_sampling.ADAPTIVE, presence_intervals.gating_refusal)
    out_root = args.out or os.path.join(args.chokepoint, "runs", "entre_camaras")
    names = args.videos if args.videos else chokepoint_videos(args.chokepoint)
    gating_root = os.path.join(args.chokepoint, "runs", "gating")
    plan = plan_videos(names, gating_root, out_root)
    print(f"Videos por extraer: {len(plan)} de {len(names)}")
    for name, source in plan:
        summary = run_video(name, source, args.chokepoint, out_root)
        print(f"{name}: {summary['identidades']} identidades, {summary['rostros']} rostros, "
              f"coincide {summary['coincide']}, mejor_coincide {summary['mejor_coincide']}, "
              f"{summary['tiempo_s']} s", flush=True)


if __name__ == "__main__":
    main()
