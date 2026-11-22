"""Render a synthetic overhead workstation and write its scripted ground truth."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from station import FPS, HEIGHT, HOME, MIN_DWELL_S, WIDTH, ZONES, caption


SCENARIOS = (
    ("correct", "PASS", ("bolt", "bracket", "fixture", "torque")),
    ("skipped", "FAIL", ("bolt", "fixture", "torque")),
    ("out_of_order", "FAIL", ("bracket", "bolt", "fixture", "torque")),
    ("short_dwell", "FAIL", ("bolt", "bracket", "fixture", "torque")),
)


def bench(rng):
    # Cool hues keep the synthetic hand the only warm object in the source.
    texture = rng.normal(0, 1.6, (HEIGHT, WIDTH, 1))
    image = np.clip(np.array([56, 48, 42]) + texture, 0, 255).astype(np.uint8)
    cv2.rectangle(image, (38, 61), (522, 386), (75, 67, 59), -1)
    cv2.rectangle(image, (38, 61), (522, 386), (125, 111, 92), 1, cv2.LINE_AA)
    for y in range(74, 385, 12):
        for x in range(50, 522, 12):
            cv2.circle(image, (x, y), 1, (91, 80, 68), -1)
    caption(image, "STEP-CHECK  /  OVERHEAD STATION", (21, 28), 0.61)
    caption(image, "SYNTHETIC VIDEO", (21, 47), 0.36, (185, 172, 152))

    for i, zone in enumerate(ZONES, 1):
        x, y = zone.center
        (left, top), (right, bottom) = zone.bounds()
        cv2.rectangle(image, (left - 8, top - 7), (right + 8, bottom + 7), (41, 36, 32), -1)
        cv2.rectangle(image, (left - 6, top - 6), (right + 6, bottom + 5), (131, 101, 76), 2)
        cv2.rectangle(image, (left, top), (right, bottom), (61, 54, 48), -1)
        caption(image, f"{i}  {zone.label.upper()}", (left - 5, bottom + 24), 0.43)
        metal = (178, 170, 155)
        if zone.name == "bolt":
            for dx, dy in [(-22, -15), (8, -16), (-12, 8), (22, 9)]:
                cv2.line(image, (x + dx, y + dy), (x + dx + 10, y + dy + 9), metal, 3)
                cv2.circle(image, (x + dx, y + dy), 4, metal, -1, cv2.LINE_AA)
        elif zone.name == "bracket":
            for dx, dy in [(-22, -14), (9, -3)]:
                corners = np.array([
                    (x + dx, y + dy), (x + dx, y + dy + 14), (x + dx + 15, y + dy + 14),
                ])
                cv2.polylines(image, [corners], False, metal, 5, cv2.LINE_AA)
        elif zone.name == "fixture":
            cv2.rectangle(image, (x - 24, y - 16), (x + 24, y + 16), metal, 2, cv2.LINE_AA)
            for dx in (-20, 20):
                cv2.circle(image, (x + dx, y), 4, metal, -1, cv2.LINE_AA)
        else:
            cv2.line(image, (x - 24, y + 10), (x + 19, y - 9), metal, 5, cv2.LINE_AA)
            cv2.circle(image, (x + 20, y - 10), 8, metal, 3, cv2.LINE_AA)
            cv2.line(image, (x - 24, y + 10), (x - 11, y + 4), (166, 124, 80), 7)

    cv2.rectangle(image, *HOME.bounds(), (114, 124, 108), 1, cv2.LINE_AA)
    caption(image, "HOME", (260, 329), 0.4)
    return image


def trajectory(rng):
    points = []
    cycles = []
    centers = {zone.name: zone.center for zone in ZONES}
    current = np.array(HOME.center)

    def hold(frames):
        points.extend(current + rng.normal(0, 0.9, (frames, 2)))

    def move(destination):
        nonlocal current
        # Travel through the center aisle so transits avoid unrelated bins.
        for waypoint in ((280, 183), destination):
            target = np.array(waypoint)
            frames = max(1, math.ceil(np.linalg.norm(target - current) / 16))
            points.extend(current + (target - current) * i / frames for i in range(1, frames + 1))
            current = target

    hold(FPS)
    for name, verdict, steps in SCENARIOS:
        # Boundaries come from the programmed motion. The verifier finds them
        # independently after decoding the video.
        start = len(points)
        visits = []
        for step in steps:
            move(centers[step])
            duration = 3 if name == "short_dwell" and step == "fixture" else int(rng.integers(17, 24))
            arrival = len(points)
            hold(duration)
            visits.append({
                "step": step,
                "start_frame": arrival,
                "end_frame": len(points),
                "accepted": duration / FPS >= MIN_DWELL_S,
            })
        move(HOME.center)
        cycles.append({
            "scenario": name, "verdict": verdict, "start_frame": start,
            "end_frame": len(points), "visits": visits,
        })
        hold(round(FPS * 1.55))
    return points, cycles


def generate(output, seed):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    backdrop = bench(rng)
    points, cycles = trajectory(rng)
    path = output / "synthetic.mp4"
    video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not video.isOpened():
        raise RuntimeError(f"Cannot open video writer for {path}")

    previous = points[0]
    try:
        for index, point in enumerate(points):
            light = 1 + 0.11 * math.sin(index * 0.29) + 0.035 * math.sin(index * 1.18)
            noise = rng.integers(-3, 4, (HEIGHT, WIDTH, 1), dtype=np.int16)
            frame = np.clip(backdrop * light + noise, 0, 255).astype(np.uint8)
            velocity = point - previous
            angle = math.degrees(math.atan2(velocity[1], velocity[0]))
            shadow = tuple(np.rint(point + (6, 6)).astype(int))
            cv2.ellipse(frame, shadow, (21, 11), angle, 0, 360, (44, 40, 35), -1, cv2.LINE_AA)
            hand_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
            for fraction in (0.45, 0.25, 0.0):
                center = tuple(np.rint(point - fraction * velocity).astype(int))
                cv2.ellipse(hand_mask, center, (20, 11), angle, 0, 360, 255, -1, cv2.LINE_AA)
            hand_mask = cv2.GaussianBlur(hand_mask, (7, 7), 1.6)
            x, y, w, h = cv2.boundingRect(hand_mask)
            region = frame[y:y + h, x:x + w]
            alpha = hand_mask[y:y + h, x:x + w, None].astype(np.float32) / 255
            color = np.clip(np.array([132, 176, 229]) * light + noise[y:y + h, x:x + w], 0, 255)
            region[:] = (1 - alpha) * region + alpha * color
            video.write(frame)
            previous = point
    finally:
        video.release()
    (output / "truth.json").write_text(json.dumps({
        "synthetic": True, "seed": seed, "fps": FPS, "cycles": cycles,
    }, indent=2) + "\n", encoding="ascii")
    print(f"Wrote {path} ({len(points)} frames at {FPS} fps) and {output / 'truth.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    generate(args.out, args.seed)
