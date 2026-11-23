import argparse
import base64
import json
import os
import shutil

import numpy as np

from bench import bench_compuerta, entre_camaras

SOURCE_EMBEDDINGS = "embeddings.json"
ORIGIN = "embeddings.json de la corrida con compuerta (un vector por detección enviada)"


def decode(text):
    return np.frombuffer(base64.b64decode(text), dtype="<f4").astype(np.float32)


def chokepoint_names(run_dir):
    names = []
    for name in sorted(os.listdir(run_dir)):
        if not os.path.exists(os.path.join(run_dir, name, SOURCE_EMBEDDINGS)):
            continue
        if entre_camaras.VIDEO_PATTERN.match(name) is None:
            continue
        kind, _, _ = bench_compuerta.find_ground_truth(name)
        if kind == bench_compuerta.KIND_CHOKEPOINT:
            names.append(name)
    return names


def identity_arrays(document):
    ids = sorted(int(key) for key in document["embeddings"])
    dimensions = int(document.get("dimensions") or 0)
    vectors = (np.stack([decode(document["embeddings"][str(i)]) for i in ids]) if ids
               else np.zeros((0, dimensions), dtype=np.float32))
    best = np.zeros(max(ids) if ids else 0, dtype=np.int64)
    for row, identity_id in enumerate(ids):
        best[identity_id - 1] = row
    return {"embeddings": vectors, "identity": np.array(ids, dtype=np.int64), "best": best}


def convert_video(source, out_dir, name):
    with open(os.path.join(source, SOURCE_EMBEDDINGS)) as f:
        arrays = identity_arrays(json.load(f))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, entre_camaras.EMBEDDINGS_NAME), **arrays)
    shutil.copyfile(os.path.join(source, entre_camaras.IDENTITIES_NAME),
                    os.path.join(out_dir, entre_camaras.IDENTITIES_NAME))
    summary = {"video": name, "fuente": source, "origen": ORIGIN,
               "identidades": int(len(arrays["identity"])), "rostros": int(len(arrays["identity"])),
               "coincide": True, "mejor_coincide": True}
    with open(os.path.join(out_dir, entre_camaras.EXTRACTION_NAME), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepara la búsqueda entre cámaras con los embeddings que escribió "
                    "face_identities.py en una corrida con compuerta")
    parser.add_argument("--run", required=True, help="carpeta de la corrida (DIR/ETIQUETA)")
    parser.add_argument("--out", required=True, help="carpeta para bench.entre_camaras --runs")
    return parser.parse_args()


def main():
    args = parse_args()
    names = chokepoint_names(args.run)
    if not names:
        raise SystemExit(f"No hay videos de ChokePoint con embeddings en {args.run}")
    for name in names:
        summary = convert_video(os.path.join(args.run, name), os.path.join(args.out, name), name)
        print(f"{name}: {summary['identidades']} detecciones con embedding", flush=True)


if __name__ == "__main__":
    main()
