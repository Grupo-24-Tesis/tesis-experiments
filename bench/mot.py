import csv
from collections import Counter
from statistics import median

HEAD_REGION = 0.35
HEAD_POINT = 0.12
MIXED_SHARE = 0.2
MIN_TRACK_IOU = 0.5
MIN_ASSIGNED_SHARE = 0.5


def load_mot_ground_truth(path):
    ground_truth = {}
    with open(path, newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 8:
                continue
            if int(float(row[6])) != 1 or int(float(row[7])) != 1:
                continue
            frame = int(row[0]) - 1
            x, y, w, h = (float(v) for v in row[2:6])
            ground_truth.setdefault(frame, []).append((int(row[1]), [x, y, x + w, y + h]))
    return ground_truth


def tracks_to_identities(manifest, results):
    identities = []
    for face in results.get("faces", []):
        track_id = str(face["track_id"])
        origin = manifest["boxes"][track_id][face["candidate"]]
        fx1, fy1, fx2, fy2 = face["face_box"]
        identities.append({
            "id": track_id,
            "faces": [{
                "frame": face["frame_idx"],
                "box": [origin[0] + fx1, origin[1] + fy1, origin[0] + fx2, origin[1] + fy2],
                "score": face["score"],
            }],
        })
    return identities


def assign_face(face_box, people):
    cx = (face_box[0] + face_box[2]) / 2
    cy = (face_box[1] + face_box[3]) / 2
    best = None
    best_distance = None
    for person_id, box in people:
        height = box[3] - box[1]
        if not (box[0] <= cx <= box[2] and box[1] <= cy <= box[1] + HEAD_REGION * height):
            continue
        head_x = (box[0] + box[2]) / 2
        head_y = box[1] + HEAD_POINT * height
        distance = (head_x - cx) ** 2 + (head_y - cy) ** 2
        if best_distance is None or distance < best_distance:
            best, best_distance = person_id, distance
    return best


def identity_metrics(identities, ground_truth):
    labelled = {pid for people in ground_truth.values() for pid, _ in people}
    counters = {}
    for identity in identities:
        counter = Counter()
        faces = identity["faces"]
        for face in faces:
            person = assign_face(face["box"], ground_truth.get(face["frame"], []))
            if person is not None:
                counter[person] += 1
        if not faces or sum(counter.values()) < MIN_ASSIGNED_SHARE * len(faces):
            counter = Counter()
        counters[identity["id"]] = counter
    summary = summarise(counters)
    return {
        "personas_etiquetadas": len(labelled),
        "identidades": len(counters),
        "encontradas": summary["found"],
        "duplicados": summary["duplicates"],
        "mezcladas": summary["mixed"],
        "sin_asignar": summary["unassigned"],
        "pureza": summary["purity"],
        "correctos": summary["found"] - summary["duplicates"] - summary["mixed"],
    }


def summarise(counters):
    main = {k: c.most_common(1)[0][0] for k, c in counters.items() if c}
    per_person = Counter(main.values())
    assigned = sum(sum(c.values()) for c in counters.values())
    pure = sum(c[main[k]] for k, c in counters.items() if c)
    return {
        "found": len(per_person),
        "duplicates": sum(n - 1 for n in per_person.values()),
        "mixed": sum(
            1 for c in counters.values()
            if c and sum(1 for v in c.values() if v >= MIXED_SHARE * sum(c.values())) >= 2
        ),
        "unassigned": sum(1 for c in counters.values() if not c),
        "purity": pure / assigned if assigned else 0.0,
    }


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def tracking_metrics(manifest, ground_truth):
    labelled = {pid for people in ground_truth.values() for pid, _ in people}
    counters = {}
    for track_id, candidates in manifest["tracks"].items():
        counter = Counter()
        for candidate, frame in candidates.items():
            box = manifest["boxes"][track_id][candidate]
            best_iou, best_person = 0.0, None
            for person_id, gt_box in ground_truth.get(frame, []):
                value = box_iou(box, gt_box)
                if value > best_iou:
                    best_iou, best_person = value, person_id
            if best_person is not None and best_iou >= MIN_TRACK_IOU:
                counter[best_person] += 1
        if not candidates or sum(counter.values()) < MIN_ASSIGNED_SHARE * len(candidates):
            counter = Counter()
        counters[track_id] = counter
    summary = summarise(counters)
    return {
        "personas_etiquetadas": len(labelled),
        "pistas": len(counters),
        "personas_cubiertas": summary["found"],
        "pistas_duplicadas": summary["duplicates"],
        "pistas_mezcladas": summary["mixed"],
        "pistas_sin_asignar": summary["unassigned"],
        "pureza": summary["purity"],
    }


def footage_statistics(ground_truth, frame_height, detections=None):
    heights = [box[3] - box[1] for people in ground_truth.values() for _, box in people]
    per_frame = [len(people) for people in ground_truth.values() if people]
    median_height = median(heights) if heights else 0.0
    stats = {
        "altura_mediana_px": median_height,
        "altura_relativa": median_height / frame_height if frame_height else 0.0,
        "personas_por_fotograma_mediana": median(per_frame) if per_frame else 0.0,
    }
    if detections is not None:
        faces = detections.get("detections", [])
        counts = Counter(d["frame"] for d in faces)
        analysed = detections.get("frames_analysed", [])
        stats["rostro_px_mediana"] = median(d["px"] for d in faces) if faces else 0.0
        stats["detecciones_por_fotograma_mediana"] = (
            median(counts.get(f, 0) for f in analysed) if analysed else 0.0
        )
    return stats
