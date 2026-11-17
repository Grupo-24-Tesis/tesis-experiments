import re
from collections import Counter

MIXED_SHARE = 0.2
CCTV_MAX_DT_S = 0.2
CCTV_MIN_IOU = 0.3
SAME_DETECTION_IOU = 0.9


def parse_chokepoint_gt(text):
    gt = {}
    for frame, body in re.findall(r'<frame number="(\d+)"\s*>(.*?)</frame>', text, re.S):
        people = []
        for pid, eyes in re.findall(r'<person id="([^"]+)"\s*>(.*?)</person>', body, re.S):
            pts = [(int(x), int(y)) for x, y in re.findall(r'x="(\d+)" y="(\d+)"', eyes)]
            if len(pts) == 2:
                people.append((pid, ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)))
        gt[int(frame)] = people
    return gt


def parse_cctv_gt(document):
    frames = []
    for frame in document["frames"].values():
        faces = [(x["box"], x["identity"]) for x in frame.get("faces", [])
                 if x.get("identity") and not x.get("ignore")
                 and not str(x["identity"]).startswith("?")]
        frames.append((float(frame["time_s"]), faces))
    frames.sort(key=lambda item: item[0])
    return frames


def inside(box, point):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def summarise_votes(votes, labelled, identities_count, faces_total, faces_matched,
                    mixed_share=MIXED_SHARE):
    majority = {i: v.most_common(1)[0][0] for i, v in votes.items() if v}
    per_person = Counter(majority.values())
    duplicates = sum(n - 1 for n in per_person.values())
    mixed = [i for i, v in votes.items()
             if v and sum(1 for c in v.values() if c >= mixed_share * sum(v.values())) >= 2]
    without_person = [i for i, v in votes.items() if not v]
    pure = sum(v[majority[i]] for i, v in votes.items() if v)
    return {
        "labelled": len(labelled),
        "identities": identities_count,
        "found": len(set(majority.values())),
        "duplicates": duplicates,
        "mixed": len(mixed),
        "without_person": len(without_person),
        "purity": pure / max(1, faces_matched),
        "faces_total": faces_total,
        "faces_matched": faces_matched,
    }


def chokepoint_identity_metrics(identities, gt, frame_map, mixed_share=MIXED_SHARE):
    labelled = {pid for people in gt.values() for pid, _ in people}
    votes = {}
    faces_total = faces_matched = 0
    for identity in identities:
        counter = Counter()
        for face in identity["faces"]:
            frame = face["frame"]
            if frame_map is not None:
                if frame >= len(frame_map):
                    continue
                frame = frame_map[frame]
            people = gt.get(frame)
            if people is None:
                continue
            faces_total += 1
            hits = [pid for pid, point in people if inside(face["box"], point)]
            if len(hits) == 1:
                counter[hits[0]] += 1
                faces_matched += 1
        votes[identity["id"]] = counter
    return summarise_votes(votes, labelled, len(identities), faces_total, faces_matched,
                           mixed_share)


def nearest_index(times, t):
    return min(range(len(times)), key=lambda i: abs(times[i] - t))


def cctv_identity_metrics(identities, fps, gt, max_dt=CCTV_MAX_DT_S, min_iou=CCTV_MIN_IOU,
                          mixed_share=MIXED_SHARE):
    labelled = {pid for _, faces in gt for _, pid in faces}
    times = [t for t, _ in gt]
    votes = {}
    faces_total = faces_matched = 0
    for identity in identities:
        counter = Counter()
        for face in identity["faces"]:
            if not times:
                continue
            t = face["frame"] / fps
            k = nearest_index(times, t)
            if abs(times[k] - t) > max_dt:
                continue
            faces_total += 1
            best = max(((iou(face["box"], b), pid) for b, pid in gt[k][1]), default=(0, None))
            if best[0] >= min_iou:
                counter[best[1]] += 1
                faces_matched += 1
        votes[identity["id"]] = counter
    return summarise_votes(votes, labelled, len(identities), faces_total, faces_matched,
                           mixed_share)


def chokepoint_detection_metrics(detections, frames_analysed, gt, frame_map):
    by_frame = {}
    for detection in detections:
        by_frame.setdefault(detection["frame"], []).append(detection["box"])
    tp = fp = fn = 0
    for frame in frames_analysed:
        if frame_map is not None:
            if frame >= len(frame_map):
                continue
            mapped = frame_map[frame]
        else:
            mapped = frame
        people = gt.get(mapped)
        if people is None:
            continue
        boxes = by_frame.get(frame, [])
        for _, point in people:
            if sum(1 for box in boxes if inside(box, point)) == 1:
                tp += 1
            else:
                fn += 1
        for box in boxes:
            if not any(inside(box, point) for _, point in people):
                fp += 1
    return precision_recall_f1(tp, fp, fn)


def match_boxes(gt_boxes, boxes, min_iou):
    pairs = sorted(((iou(g, b), gi, bi) for gi, g in enumerate(gt_boxes)
                    for bi, b in enumerate(boxes)), reverse=True)
    used_gt, used_boxes, matched = set(), set(), 0
    for value, gi, bi in pairs:
        if value < min_iou:
            break
        if gi in used_gt or bi in used_boxes:
            continue
        used_gt.add(gi)
        used_boxes.add(bi)
        matched += 1
    return matched


def cctv_detection_metrics(detections, frames_analysed, fps, gt, max_dt=CCTV_MAX_DT_S,
                           min_iou=CCTV_MIN_IOU):
    by_frame = {}
    for detection in detections:
        by_frame.setdefault(detection["frame"], []).append(detection["box"])
    analysed = sorted(frames_analysed)
    analysed_times = [frame / fps for frame in analysed]
    tp = fn = 0
    unmatched_boxes = {}
    for t, faces in gt:
        gt_boxes = [box for box, _ in faces]
        if not analysed_times:
            fn += len(gt_boxes)
            continue
        k = nearest_index(analysed_times, t)
        if abs(analysed_times[k] - t) > max_dt:
            fn += len(gt_boxes)
            continue
        boxes = by_frame.get(analysed[k], [])
        matched = match_boxes(gt_boxes, boxes, min_iou)
        tp += matched
        fn += len(gt_boxes) - matched
        unmatched = len(boxes) - matched
        unmatched_boxes[k] = min(unmatched, unmatched_boxes.get(k, unmatched))
    return precision_recall_f1(tp, sum(unmatched_boxes.values()), fn)


def missing_detections(baseline, candidate, min_iou=SAME_DETECTION_IOU):
    by_frame = {}
    for detection in candidate:
        by_frame.setdefault(detection["frame"], []).append(detection["box"])
    return [d for d in baseline
            if not any(iou(d["box"], box) >= min_iou for box in by_frame.get(d["frame"], []))]


def format_time(seconds):
    whole = int(seconds)
    tenth = int((seconds - whole) * 10 + 1e-9)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}.{tenth}"
