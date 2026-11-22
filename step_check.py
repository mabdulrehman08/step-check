"""Verify standard work from an overhead video calibrated to this station."""

import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

import cv2
import numpy as np

from station import HEIGHT, HOME, HOME_HOLD_S, MIN_DWELL_S, TEMPLATE, WIDTH, ZONES, caption


class HandTracker:
    def __init__(self):
        self.point = None
        self.missed = 0
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def update(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (5, 55, 105), (28, 180, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            moments = cv2.moments(contour)
            area = moments["m00"]
            if 220 <= area <= 1900:
                center = np.array([moments["m10"], moments["m01"]]) / area
                candidates.append((area, center))

        if self.point is None:
            if candidates:
                self.point = max(candidates, key=lambda item: item[0])[1]
        elif candidates:
            _, closest = min(candidates, key=lambda item: np.linalg.norm(item[1] - self.point))
            if np.linalg.norm(closest - self.point) <= 45 + 15 * self.missed:
                self.point = 0.78 * closest + 0.22 * self.point
            else:
                candidates = []
        if not candidates:
            self.missed += 1
            if self.missed >= 4:
                self.point = None
            return None
        self.missed = 0
        return tuple(np.rint(self.point).astype(int))


@dataclass
class Visit:
    step: str
    start_frame: int
    end_frame: int
    dwell_s: float
    accepted: bool


class Dwell:
    def __init__(self, fps):
        self.fps = fps
        self.zone = None

    def update(self, index, point):
        visit = None
        if self.zone is not None:
            if self.zone.contains(point, margin=5):
                self.last_inside = index
                self.observations += 1
                return None
            if index - self.last_inside <= 2:
                return None
            visit = self.close()
        for zone in ZONES:
            if zone.contains(point, margin=-5):
                self.zone = zone
                self.start = self.last_inside = index
                self.observations = 1
                break
        return visit

    def close(self):
        if self.zone is None:
            return None
        seconds = self.observations / self.fps
        visit = Visit(self.zone.name, self.start, self.last_inside + 1,
                      round(seconds, 3), seconds >= MIN_DWELL_S)
        self.zone = None
        return visit if seconds >= 0.20 else None


@dataclass
class Cycle:
    start_frame: int
    end_frame: int = 0
    cycle_time_s: float = 0.0
    visits: list[Visit] = field(default_factory=list)
    sequence: list[str] = field(default_factory=list)
    verdict: str = ""
    reason: str = ""
    reason_code: str = ""

    def add(self, visit):
        if visit is not None:
            self.visits.append(visit)
            if visit.accepted:
                self.sequence.append(visit.step)

    def finish(self, end, fps, complete):
        self.end_frame = end
        self.cycle_time_s = round((end - self.start_frame) / fps, 2)
        self.verdict = "FAIL"
        if not complete:
            self.reason_code, self.reason = "incomplete", "no confirmed return to home"
        elif tuple(self.sequence) == TEMPLATE:
            self.verdict = "PASS"
            self.reason_code, self.reason = "correct", "standard work matched"
        else:
            missing = Counter(TEMPLATE) - Counter(self.sequence)
            extra = Counter(self.sequence) - Counter(TEMPLATE)
            problems = []
            for step in missing:
                rejected = [v.dwell_s for v in self.visits if v.step == step and not v.accepted]
                if rejected:
                    problems.append(("short_dwell",
                                     f"{step} dwell {max(rejected):.2f}s < {MIN_DWELL_S:.2f}s"))
                else:
                    problems.append(("skipped", f"missing {step}"))
            if extra:
                problems.append(("unexpected", "extra " + ", ".join(extra.elements())))
            if not problems:
                mismatch = next(i for i, step in enumerate(TEMPLATE) if step != self.sequence[i])
                position = "first" if mismatch == 0 else f"at step {mismatch + 1}"
                problems.append((
                    "out_of_order",
                    f"out of order: expected {TEMPLATE[mismatch]} {position}, saw {self.sequence[mismatch]}",
                ))
            self.reason_code = problems[0][0]
            self.reason = " / ".join(detail for _, detail in problems)


class CycleTracker:
    def __init__(self, fps):
        self.fps = fps
        self.home_frames = math.ceil(HOME_HOLD_S * fps)
        self.home_run = 0
        self.leave_run = 0
        self.armed = False
        self.current = None
        self.completed = []
        self.dwell = Dwell(fps)

    def update(self, index, point):
        self.home_run = self.home_run + 1 if HOME.contains(point) else 0
        if self.current is None:
            if self.home_run >= self.home_frames:
                self.armed = True
            away = point is not None and not HOME.contains(point, margin=5)
            self.leave_run = self.leave_run + 1 if self.armed and away else 0
            if self.leave_run >= 3:
                self.current = Cycle(index - self.leave_run + 1)
                self.leave_run = 0
                self.armed = False
        if self.current is not None:
            self.current.add(self.dwell.update(index, point))
            if self.home_run >= self.home_frames:
                self.finish(index - self.home_run + 1)
                self.armed = True

    def finish(self, end, complete=True):
        if self.current is not None:
            self.current.add(self.dwell.close())
            self.current.finish(end, self.fps, complete)
            self.completed.append(self.current)
            self.current = None


GREEN, RED, AMBER, MUTED = (163, 227, 107), (128, 136, 252), (93, 205, 251), (180, 166, 146)
PANEL = 300


def wrap_caption(panel, text, y, size=0.44):
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, size, 1)[0][0] > PANEL - 38:
            caption(panel, line, (19, y), size)
            y += 16
            line = word
        else:
            line = candidate
    caption(panel, line, (19, y), size)


def annotate(frame, tracker, point, trail, index):
    cycle = tracker.current
    active = tracker.dwell.zone
    for zone in ZONES:
        color = MUTED
        if cycle and zone.name in cycle.sequence:
            color = GREEN
        if zone == active:
            color = AMBER
        cv2.rectangle(frame, *zone.bounds(), color, 2, cv2.LINE_AA)
    if len(trail) > 1:
        cv2.polylines(frame, [np.array(trail)], False, (161, 205, 214), 1, cv2.LINE_AA)
    if point is not None:
        cv2.circle(frame, point, 20, AMBER, 1, cv2.LINE_AA)
        cv2.drawMarker(frame, point, (240, 250, 250), cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
    panel = np.full((HEIGHT, PANEL, 3), (43, 34, 28), dtype=np.uint8)
    cv2.line(panel, (0, 0), (0, HEIGHT - 1), (117, 98, 77), 1)
    caption(panel, "SEQUENCE MONITOR", (19, 31), 0.68, weight=2)
    caption(panel, f"{index / tracker.fps:05.1f}s  /  MIN DWELL {MIN_DWELL_S:.2f}s", (19, 54), 0.44, MUTED)
    cv2.line(panel, (19, 69), (PANEL - 19, 69), (106, 84, 65), 1)
    if cycle:
        caption(panel, f"CYCLE {len(tracker.completed) + 1:02d}  /  RUNNING", (19, 96), 0.55, AMBER)
        steps = " > ".join(cycle.sequence[-4:]) or "Waiting for first step"
        if len(cycle.sequence) > 4:
            steps = "... " + steps
        caption(panel, steps, (19, 121), 0.45)
        if active:
            caption(panel, f"{active.label}: {tracker.dwell.observations / tracker.fps:.2f}s",
                    (19, 145), 0.46, AMBER)
    elif tracker.completed:
        last = tracker.completed[-1]
        color = GREEN if last.verdict == "PASS" else RED
        caption(panel, f"CYCLE {len(tracker.completed):02d}  /  {last.verdict}", (19, 96), 0.55, color)
        wrap_caption(panel, last.reason, 122)
    else:
        caption(panel, "Waiting at home", (19, 96), 0.55, MUTED)

    caption(panel, "STANDARD WORK", (19, 183), 0.43, MUTED)
    caption(panel, " > ".join(TEMPLATE), (19, 206), 0.45)
    caption(panel, "COMPLETED CYCLES", (19, 245), 0.43, MUTED)
    offset = max(0, len(tracker.completed) - 4)
    for row, previous in enumerate(tracker.completed[offset:]):
        y = 272 + row * 28
        caption(panel, f"{offset + row + 1:02d}   {previous.verdict}", (19, y), 0.49,
                GREEN if previous.verdict == "PASS" else RED)
        caption(panel, f"{previous.cycle_time_s:5.2f}s", (215, y), 0.49)
    return np.hstack((frame, panel))


def match_intervals(predicted, expected, threshold=0.30):
    """Greedily pair same-label intervals by descending temporal IoU."""
    candidates = []
    for i, a in enumerate(predicted):
        for j, b in enumerate(expected):
            if a.get("step") != b.get("step"):
                continue
            overlap = min(a["end_frame"], b["end_frame"]) - max(a["start_frame"], b["start_frame"])
            span = max(a["end_frame"], b["end_frame"]) - min(a["start_frame"], b["start_frame"])
            if span > 0 and overlap / span >= threshold:
                candidates.append((overlap / span, i, j))
    pairs, used_pred, used_truth = [], set(), set()
    for _, i, j in sorted(candidates, reverse=True):
        if i not in used_pred and j not in used_truth:
            pairs.append((i, j))
            used_pred.add(i)
            used_truth.add(j)
    return pairs


def evaluate(predicted, truth):
    actual = truth["cycles"]
    pairs = match_intervals(predicted, actual, threshold=0.50)
    pred_visits = [[v for v in cycle["visits"] if v["accepted"]] for cycle in predicted]
    truth_visits = [[v for v in cycle["visits"] if v["accepted"]] for cycle in actual]
    tp = sum(len(match_intervals(pred_visits[i], truth_visits[j])) for i, j in pairs)
    pred_count = sum(map(len, pred_visits))
    truth_count = sum(map(len, truth_visits))
    correct = sum(predicted[i]["verdict"] == actual[j]["verdict"] for i, j in pairs)
    reasons_correct = sum(predicted[i]["reason_code"] == actual[j]["scenario"] for i, j in pairs)
    cycle_count = len(predicted) + len(actual) - len(pairs)
    return {
        "event_true_positives": tp,
        "event_false_positives": pred_count - tp,
        "event_false_negatives": truth_count - tp,
        "event_precision": tp / pred_count if pred_count else 0.0,
        "event_recall": tp / truth_count if truth_count else 0.0,
        "matched_cycles": len(pairs),
        "cycles_scored": cycle_count,
        "correct_cycle_verdicts": correct,
        "cycle_accuracy": correct / cycle_count if cycle_count else 0.0,
        "correct_cycle_reasons": reasons_correct,
    }


def verify(source, output, truth_path=None):
    video = cv2.VideoCapture(str(source))
    if not video.isOpened():
        raise ValueError(f"Cannot open {source}")
    writer = None
    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        shape = tuple(int(video.get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        if shape != (WIDTH, HEIGHT) or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Expected calibrated {WIDTH}x{HEIGHT} video with valid fps, got {shape} at {fps} fps")
        truth = json.loads(truth_path.read_text(encoding="utf-8")) if truth_path else None
        if truth is not None and not math.isclose(truth["fps"], fps, abs_tol=0.01):
            raise ValueError("Ground-truth and video frame rates differ")

        output.mkdir(parents=True, exist_ok=True)
        annotated = output / "annotated.mp4"
        if annotated.resolve() == source.resolve():
            raise ValueError("Choose an output directory that will not overwrite the source")
        writer = cv2.VideoWriter(str(annotated), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (WIDTH + PANEL, HEIGHT))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for {annotated}")
        hand = HandTracker()
        tracker = CycleTracker(fps)
        trail = deque(maxlen=10)
        index = 0
        # Buffer one frame so EOF can show the verdict for an unfinished cycle.
        pending = None
        while True:
            decoded, frame = video.read()
            if not decoded:
                break
            if pending:
                writer.write(annotate(*pending))
            point = hand.update(frame)
            tracker.update(index, point)
            if point is None:
                trail.clear()
            else:
                trail.append(point)
            pending = (frame, tracker, point, trail, index)
            index += 1
        if pending is None:
            raise ValueError("No frames could be decoded")
        tracker.finish(index, complete=False)
        writer.write(annotate(*pending))
    finally:
        if writer is not None:
            writer.release()
        video.release()

    cycles = [asdict(cycle) for cycle in tracker.completed]
    report = {"video": source.name, "fps": fps, "frames": index, "template": TEMPLATE,
              "min_dwell_s": MIN_DWELL_S, "cycles": cycles}
    for i, cycle in enumerate(tracker.completed, 1):
        steps = " > ".join(cycle.sequence) or "(none)"
        print(f"Cycle {i:02d}  {cycle.verdict}  {cycle.cycle_time_s:5.2f}s  {steps} | {cycle.reason}")
    if truth is not None:
        metrics = report["evaluation"] = evaluate(cycles, truth)
        print(f"Events: precision {metrics['event_precision']:.1%}, recall {metrics['event_recall']:.1%} "
              f"(TP={metrics['event_true_positives']}, FP={metrics['event_false_positives']}, "
              f"FN={metrics['event_false_negatives']})")
        print(f"Cycle accuracy: {metrics['cycle_accuracy']:.1%} "
              f"({metrics['correct_cycle_verdicts']}/{metrics['cycles_scored']}) | "
              f"reason codes {metrics['correct_cycle_reasons']}/{metrics['cycles_scored']}")
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(f"Wrote {annotated} and {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--truth", type=Path, help="Optional synthetic labels for evaluation")
    args = parser.parse_args()
    try:
        verify(args.video, args.out, args.truth)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"step-check: {error}\n")
