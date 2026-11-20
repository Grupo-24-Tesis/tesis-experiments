import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import numpy as np

from bench import bench_compuerta, metrics

DEFAULT_RUNS = "/data/TesisFaceRecognition/chokepoint/runs/entre_camaras"
EMBEDDINGS_NAME = "embeddings.npz"
IDENTITIES_NAME = "identities.json"
EXTRACTION_NAME = "extraccion.json"
VARIANTS = ("media", "mejor", "par_maximo")
CASES = ("todas", "misma_pasada", "otra_pasada", "otro_portal")
THRESHOLDS = tuple(round(0.05 * step, 2) for step in range(19))
TOP_K = (1, 5, 10)
VIDEO_PATTERN = re.compile(r"^P([12])([EL])_S(\d)_C([123])(?:\.(\d))?$")


def parse_video_name(name):
    match = VIDEO_PATTERN.match(name)
    if match is None:
        raise ValueError(f"nombre de video no válido: {name}")
    portal, direction, session, camera, part = match.groups()
    return {"portal": int(portal), "direction": direction, "session": int(session),
            "camera": int(camera), "part": None if part is None else int(part)}


def relation(a, b):
    if a == b:
        return None
    first = parse_video_name(a)
    second = parse_video_name(b)
    if first["portal"] != second["portal"]:
        return "otro_portal"
    same_pass = all(first[key] == second[key] for key in ("direction", "session", "part"))
    if same_pass and first["camera"] != second["camera"]:
        return "misma_pasada"
    return "otra_pasada"


def person_key(pid):
    return pid.rsplit(":", 1)[-1]


def identity_votes(identity, gt, frame_map):
    counter = Counter()
    for face in identity["faces"]:
        people = metrics.people_at(gt, frame_map, face["frame"])
        if people is None:
            continue
        hits = [pid for pid, point in people if metrics.inside(face["box"], point)]
        if len(hits) == 1:
            counter[hits[0]] += 1
    return counter


def vote_outcome(counter, mixed_share):
    if not counter:
        return None, "sin_persona"
    total = sum(counter.values())
    if sum(1 for count in counter.values() if count >= mixed_share * total) >= 2:
        return None, "mezclada"
    return person_key(counter.most_common(1)[0][0]), "etiquetada"


def identity_outcomes(identities, gt, frame_map, mixed_share=metrics.MIXED_SHARE):
    return {identity["id"]: vote_outcome(identity_votes(identity, gt, frame_map), mixed_share)
            for identity in identities}


def identity_labels(identities, gt, frame_map, mixed_share=metrics.MIXED_SHARE):
    return {identity_id: label for identity_id, (label, _) in
            identity_outcomes(identities, gt, frame_map, mixed_share).items()}


def load_video_embeddings(folder):
    data = np.load(os.path.join(folder, EMBEDDINGS_NAME))
    embeddings = data["embeddings"]
    identity = data["identity"]
    best = data["best"]
    return [{"id": int(identity_id), "faces": embeddings[identity == identity_id],
             "best": embeddings[int(best[int(identity_id) - 1])]}
            for identity_id in sorted(set(identity.tolist()))]


def unit(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def build_gallery(items):
    sizes = [len(item["faces"]) for item in items]
    return {
        "means": np.stack([unit(item["faces"].mean(axis=0)) for item in items]),
        "bests": np.stack([item["best"] for item in items]),
        "faces": np.concatenate([item["faces"] for item in items]),
        "starts": np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int),
    }


def query_scores(gallery, item, variant):
    if variant == "media":
        return gallery["means"] @ unit(item["faces"].mean(axis=0))
    if variant == "mejor":
        return gallery["bests"] @ item["best"]
    if variant == "par_maximo":
        pairs = gallery["faces"] @ item["faces"].T
        return np.maximum.reduceat(pairs.max(axis=1), gallery["starts"])
    raise ValueError(f"variante no válida: {variant}")


def ranked_mask(scores, mask):
    order = np.lexsort((mask.astype(int), -np.asarray(scores)))
    return np.asarray(mask)[order]


def chance_hit(gallery_size, targets, k):
    miss = 1.0
    for i in range(min(k, gallery_size)):
        factor = (gallery_size - targets - i) / (gallery_size - i)
        if factor <= 0:
            return 1.0
        miss *= factor
    return 1.0 - miss


def search_metrics(queries):
    keys = ("acierto_1", "acierto_5", "acierto_10", "precision_10", "precision_10_max",
            "rango_medio", "ap_media")
    if not queries:
        return {"consultas": 0, **{key: None for key in keys},
                "azar": {key: None for key in ("acierto_1", "acierto_5", "acierto_10",
                                               "precision_10", "rango_medio")}}
    ranks, precisions, maxima, aps = [], [], [], []
    chance = {key: [] for key in ("acierto_1", "acierto_5", "acierto_10", "precision_10",
                                  "rango_medio")}
    for scores, mask in queries:
        mask = np.asarray(mask, dtype=bool)
        ranked = ranked_mask(scores, mask)
        positions = np.flatnonzero(ranked) + 1
        size = len(ranked)
        targets = int(mask.sum())
        ranks.append(int(positions[0]))
        precisions.append(ranked[:10].sum() / 10)
        maxima.append(min(10, targets) / 10)
        aps.append(float(np.mean(np.arange(1, targets + 1) / positions)))
        for k in TOP_K:
            chance[f"acierto_{k}"].append(chance_hit(size, targets, k))
        chance["precision_10"].append(targets * min(10, size) / size / 10)
        chance["rango_medio"].append((size + 1) / (targets + 1))
    ranks = np.array(ranks)
    result = {"consultas": len(queries)}
    for k in TOP_K:
        result[f"acierto_{k}"] = float(np.mean(ranks <= k))
    result["precision_10"] = float(np.mean(precisions))
    result["precision_10_max"] = float(np.mean(maxima))
    result["rango_medio"] = float(np.mean(ranks))
    result["ap_media"] = float(np.mean(aps))
    result["azar"] = {key: float(np.mean(values)) for key, values in chance.items()}
    return result


def threshold_table(queries, thresholds=THRESHOLDS):
    table = []
    total_targets = sum(int(np.asarray(mask).sum()) for _, mask in queries)
    for threshold in thresholds:
        passing = hits = 0
        for scores, mask in queries:
            passed = np.asarray(scores) >= threshold
            passing += int(passed.sum())
            hits += int((passed & np.asarray(mask, dtype=bool)).sum())
        table.append({
            "umbral": threshold,
            "precision": hits / passing if passing else None,
            "exhaustividad": hits / total_targets if total_targets else None,
            "resultados_por_consulta": passing / len(queries) if queries else None,
        })
    return table


def evaluate(items):
    gallery = build_gallery(items)
    videos = [item["video"] for item in items]
    labels = np.array([item["label"] for item in items], dtype=object)
    relations = {}
    for video in set(videos):
        for other in set(videos):
            relations[(video, other)] = relation(video, other)
    collected = {variant: {case: {"queries": [], "sin_objetivo": 0} for case in CASES}
                 for variant in VARIANTS}
    for index, item in enumerate(items):
        if item["label"] is None:
            continue
        same_label = labels == item["label"]
        case_masks = {}
        for case in CASES:
            gallery_mask = np.array([
                relations[(item["video"], other)] is not None and
                (case == "todas" or relations[(item["video"], other)] == case)
                for other in videos])
            case_masks[case] = (gallery_mask, gallery_mask & same_label)
        for variant in VARIANTS:
            scores = query_scores(gallery, item, variant)
            for case in CASES:
                gallery_mask, target_mask = case_masks[case]
                if not target_mask.any():
                    collected[variant][case]["sin_objetivo"] += 1
                    continue
                collected[variant][case]["queries"].append(
                    (scores[gallery_mask], target_mask[gallery_mask]))
    return {variant: {case: {"metricas": search_metrics(data["queries"]),
                             "umbrales": threshold_table(data["queries"]),
                             "sin_objetivo": data["sin_objetivo"]}
                      for case, data in cases.items()}
            for variant, cases in collected.items()}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def label_video(folder, name):
    kind, gt, frame_map = bench_compuerta.find_ground_truth(name)
    if kind != bench_compuerta.KIND_CHOKEPOINT:
        raise ValueError(f"El video {name} no tiene etiquetas de ChokePoint")
    identities = load_json(os.path.join(folder, IDENTITIES_NAME))["identities"]
    return identity_outcomes(identities, gt, frame_map)


def format_value(value):
    return "-" if value is None else f"{value:.3f}"


def print_results(results):
    for variant in VARIANTS:
        print(f"Variante {variant}")
        print(f"{'caso':<14}{'consultas':>10}{'sin objetivo':>14}{'acierto@1':>11}"
              f"{'acierto@5':>11}{'acierto@10':>12}{'precision@10':>14}{'rango medio':>13}"
              f"{'AP media':>10}")
        for case in CASES:
            data = results[variant][case]
            m = data["metricas"]
            print(f"{case:<14}{m['consultas']:>10}{data['sin_objetivo']:>14}"
                  f"{format_value(m['acierto_1']):>11}{format_value(m['acierto_5']):>11}"
                  f"{format_value(m['acierto_10']):>12}{format_value(m['precision_10']):>14}"
                  f"{format_value(m['rango_medio']):>13}{format_value(m['ap_media']):>10}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Puntúa la búsqueda de rostros entre cámaras en ChokePoint")
    parser.add_argument("--runs", default=DEFAULT_RUNS)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = args.out or os.path.join(args.runs, "resultados.json")
    names = sorted(name for name in os.listdir(args.runs)
                   if os.path.exists(os.path.join(args.runs, name, EXTRACTION_NAME)))
    if not names:
        raise SystemExit(f"No hay videos extraídos en {args.runs}")
    items = []
    statuses = Counter()
    different = []
    for name in names:
        folder = os.path.join(args.runs, name)
        extraction = load_json(os.path.join(folder, EXTRACTION_NAME))
        if not extraction["coincide"]:
            different.append(name)
        outcomes = label_video(folder, name)
        for entry in load_video_embeddings(folder):
            label, status = outcomes[entry["id"]]
            statuses[status] += 1
            items.append({"video": name, "id": entry["id"], "label": label,
                          "faces": entry["faces"], "best": entry["best"]})
    results = evaluate(items)
    document = {"extraccion": {"videos": len(names), "identidades": len(items),
                               "etiquetadas": statuses["etiquetada"],
                               "sin_persona": statuses["sin_persona"],
                               "mezcladas": statuses["mezclada"],
                               "videos_distintos": different},
                "resultados": results}
    with open(out, "w") as f:
        json.dump(document, f, indent=2, ensure_ascii=False)
    print(f"Videos: {len(names)}  identidades: {len(items)}  etiquetadas: "
          f"{statuses['etiquetada']}  sin persona: {statuses['sin_persona']}  mezcladas: "
          f"{statuses['mezclada']}")
    print()
    print_results(results)
    print(f"Resultados escritos en {out}")


if __name__ == "__main__":
    main()
