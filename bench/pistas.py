from collections import Counter

from bench import metrics, mot

VOTE_RULE = "pistas: votos de las cajas de persona"
CCTV_VOTE_MAX_DT_S = 0.5
UNVERIFIED = object()


def track_candidates(manifest):
    for track_id, frames in manifest["tracks"].items():
        boxes = manifest["boxes"].get(track_id, {})
        items = []
        for name, frame in frames.items():
            if name not in boxes:
                raise ValueError(f"El manifest no tiene la caja del candidato {name} de la pista "
                                 f"{track_id}")
            items.append((frame, boxes[name]))
        yield track_id, items


def candidate_counts(manifest):
    return {track_id: len(frames) for track_id, frames in manifest["tracks"].items()}


def person_in_head(box, labelled_points):
    best = None
    best_distance = None
    for person_id, point in labelled_points:
        if not mot.in_head_region(box, point):
            continue
        distance = mot.head_distance(box, point)
        if best_distance is None or distance < best_distance:
            best, best_distance = person_id, distance
    return best


def track_votes(manifest, person_at):
    votes = {}
    for track_id, items in track_candidates(manifest):
        counter = Counter()
        for frame, box in items:
            person = person_at(frame, box)
            if person is not None:
                counter[person] += 1
        votes[track_id] = dict(counter)
    return votes


def mot_track_votes(manifest, ground_truth):
    return track_votes(manifest, lambda frame, box: mot.match_person_box(
        box, ground_truth.get(frame, [])))


def chokepoint_track_votes(manifest, gt, frame_map):
    def person_at(frame, box):
        people = metrics.people_at(gt, frame_map, frame)
        return None if people is None else person_in_head(box, people)
    return track_votes(manifest, person_at)


def labelled_faces_near(gt, times, t, max_dt):
    if not times:
        return None
    k = metrics.nearest_index(times, t)
    if abs(times[k] - t) > max_dt:
        return None
    return gt[k][1]


def cctv_track_votes(manifest, fps, gt, max_dt=CCTV_VOTE_MAX_DT_S):
    times = [t for t, _ in gt]

    def person_at(frame, box):
        faces = labelled_faces_near(gt, times, frame / fps, max_dt)
        if faces is None:
            return None
        return person_in_head(box, [(pid, mot.box_centre(face_box)) for face_box, pid in faces])
    return track_votes(manifest, person_at)


def face_verdicts(identities, face_person):
    found = {}
    for identity in identities:
        found.setdefault(identity["id"], []).extend(
            face_person(face["frame"], face["box"]) for face in identity["faces"])
    return found


def mot_saved_faces(identities, ground_truth):
    def face_person(frame, box):
        if frame not in ground_truth:
            return UNVERIFIED
        return mot.assign_face(box, ground_truth[frame])
    return face_verdicts(identities, face_person)


def chokepoint_saved_faces(identities, gt, frame_map):
    def face_person(frame, box):
        people = metrics.people_at(gt, frame_map, frame)
        if people is None:
            return UNVERIFIED
        hits = [pid for pid, point in people if metrics.inside(box, point)]
        return hits[0] if len(hits) == 1 else None
    return face_verdicts(identities, face_person)


def cctv_saved_faces(identities, fps, gt, max_dt=metrics.CCTV_MAX_DT_S,
                     min_iou=metrics.CCTV_MIN_IOU):
    times = [t for t, _ in gt]

    def face_person(frame, box):
        faces = labelled_faces_near(gt, times, frame / fps, max_dt)
        if faces is None:
            return UNVERIFIED
        best = max(((metrics.iou(box, face_box), pid) for face_box, pid in faces),
                   default=(0, None))
        return best[1] if best[0] >= min_iou else None
    return face_verdicts(identities, face_person)


def track_identity_metrics(votes, saved_faces, labelled_people, candidates=None):
    counters = []
    correct = unverified = 0
    for track_id, verdicts in saved_faces.items():
        counter = Counter(votes.get(track_id, {}))
        items = sum(counter.values()) if candidates is None else candidates.get(track_id, 0)
        if not mot.has_main_person(counter, items):
            counters.append(None)
            continue
        counters.append(counter)
        person = mot.main_person(counter)
        correct += sum(1 for verdict in verdicts if verdict == person)
        unverified += sum(1 for verdict in verdicts if verdict is UNVERIFIED)
    summary = mot.summarise(counters)
    return {
        "personas_etiquetadas": len(labelled_people),
        "identidades": len(counters),
        "encontradas": summary["found"],
        "duplicados": summary["duplicates"],
        "mezcladas": summary["mixed"],
        "sin_asignar": summary["unassigned"],
        "pureza": summary["purity"],
        "correctos": summary["found"] - summary["duplicates"] - summary["mixed"],
        "cara_correcta": correct,
        "cara_sin_verificar": unverified,
    }


def mot_identity(manifest, identities, ground_truth):
    labelled = {pid for people in ground_truth.values() for pid, _ in people}
    return track_identity_metrics(mot_track_votes(manifest, ground_truth),
                                  mot_saved_faces(identities, ground_truth), labelled,
                                  candidate_counts(manifest))


def chokepoint_identity(manifest, identities, gt, frame_map):
    return track_identity_metrics(chokepoint_track_votes(manifest, gt, frame_map),
                                  chokepoint_saved_faces(identities, gt, frame_map),
                                  metrics.labelled_people(gt, frame_map))


def cctv_identity(manifest, identities, fps, gt):
    labelled = {pid for _, faces in gt for _, pid in faces}
    return track_identity_metrics(cctv_track_votes(manifest, fps, gt),
                                  cctv_saved_faces(identities, fps, gt), labelled)
