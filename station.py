"""Station geometry and standard work shared by the renderer and verifier."""

from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class Zone:
    name: str
    label: str
    center: tuple[int, int]
    half_size: tuple[int, int] = (41, 32)

    def bounds(self, margin=0):
        x, y = self.center
        w, h = self.half_size
        return (x - w - margin, y - h - margin), (x + w + margin, y + h + margin)

    def contains(self, point, margin=0):
        if point is None:
            return False
        (left, top), (right, bottom) = self.bounds(margin)
        x, y = point
        return left <= x <= right and top <= y <= bottom


WIDTH, HEIGHT = 560, 408
FPS = 20
MIN_DWELL_S = 0.70
HOME_HOLD_S = 0.45
ZONES = (
    Zone("bolt", "Bolt bin", (142, 117)),
    Zone("bracket", "Bracket bin", (418, 117)),
    Zone("fixture", "Part fixture", (142, 252)),
    Zone("torque", "Torque tool", (418, 252)),
)
HOME = Zone("home", "Home", (280, 357), (39, 22))
TEMPLATE = tuple(zone.name for zone in ZONES)


def caption(image, text, xy, size=0.5, color=(226, 234, 232), weight=1):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, weight, cv2.LINE_AA)
