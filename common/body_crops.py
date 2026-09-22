"""Правило кропов тела, общее для клиента и хаба (ТЗ F-202).

The client decides WHEN to cut a crop and cuts it; the hub decides whether to
keep it. Both sides must agree on the numbers - height, interval, how much the
aspect has to change to count as a new view - or one of them would quietly
reject the other's work. The numbers and the schedule therefore live here, in
the shared contract, and both sides import them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: ТЗ F-202: JPEG высотой до 640 px.
MAX_HEIGHT = 640
#: ТЗ F-202: a fresh crop of a track every two seconds.
INTERVAL_S = 2.0
#: ТЗ F-202: "смена ракурса" is read as a change of the box's aspect ratio.
ASPECT_TOLERANCE = 0.15
#: A crop narrower or shorter than this is not a person.
MIN_SIDE_PX = 24
#: Sanity cap: no JPEG of a 640-px crop should be anywhere near this.
MAX_BYTES = 512 * 1024

_JPEG_MAGIC = b"\xff\xd8\xff"


@dataclass
class CropSchedule:
    """When a track needs a new body crop (ТЗ F-202), with no camera involved."""

    interval_s: float = INTERVAL_S
    tolerance: float = ASPECT_TOLERANCE
    _open: dict[str, dict[str, float]] = field(default_factory=dict)

    def should_send(self, track_id: str, aspect: float, *, now: float) -> bool:
        """True on appearance, every ``interval_s``, and on an aspect change."""
        key = str(track_id)
        state = self._open.get(key)
        if state is None:
            self._open[key] = {"at": float(now), "aspect": float(aspect)}
            return True
        if float(now) - state["at"] >= self.interval_s:
            state["at"], state["aspect"] = float(now), float(aspect)
            return True
        reference = state["aspect"]
        if reference > 0 and abs(float(aspect) - reference) >= self.tolerance:
            # The person turned: that is a new view of them, and the hub wants
            # one of those (ТЗ F-204 spreads one face over the whole track).
            state["at"], state["aspect"] = float(now), float(aspect)
            return True
        return False

    def forget(self, track_id: str) -> None:
        """Drop a track that left: its next appearance is a fresh crop anyway."""
        self._open.pop(str(track_id), None)

    def live(self) -> list[str]:
        return list(self._open)


def crop_box(bbox: tuple[float, float, float, float], width: int, height: int,
             *, margin: float = 0.04) -> tuple[int, int, int, int]:
    """The pixel rectangle of a normalized box, with a little air around it."""
    x1, y1, x2, y2 = (float(value) for value in bbox)
    span_x, span_y = abs(x2 - x1), abs(y2 - y1)
    x1, x2 = min(x1, x2) - margin * span_x, max(x1, x2) + margin * span_x
    y1, y2 = min(y1, y2) - margin * span_y, max(y1, y2) + margin * span_y
    left = max(0, min(int(width), int(round(x1 * width))))
    top = max(0, min(int(height), int(round(y1 * height))))
    right = max(left, min(int(width), int(round(x2 * width))))
    bottom = max(top, min(int(height), int(round(y2 * height))))
    return left, top, right, bottom


def crop_is_large_enough(box: tuple[int, int, int, int], *, min_side: int = MIN_SIDE_PX) -> bool:
    left, top, right, bottom = box
    return (right - left) >= int(min_side) and (bottom - top) >= int(min_side)


def scaled_size(width: int, height: int, *, max_height: int = MAX_HEIGHT) -> tuple[int, int]:
    """The crop's size with its height capped, keeping the aspect ratio."""
    width, height = max(1, int(width)), max(1, int(height))
    if height <= int(max_height):
        return width, height
    scale = int(max_height) / height
    return max(1, int(round(width * scale))), int(max_height)


def jpeg_height(jpeg: bytes) -> int:
    """The height of a JPEG, read from its SOF marker (0 when unreadable)."""
    data = bytes(jpeg or b"")
    if len(data) < 4 or not data.startswith(_JPEG_MAGIC):
        return 0
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        length = int.from_bytes(data[index + 2:index + 4], "big")
        if length < 2:
            return 0
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            return int.from_bytes(data[index + 5:index + 7], "big")
        index += 2 + length
    return 0


def crop_is_valid(jpeg: bytes, *, max_height: int = MAX_HEIGHT) -> tuple[bool, str]:
    """``(ok, reason)`` for one incoming crop."""
    data = bytes(jpeg or b"")
    if not data:
        return False, "the crop is empty"
    if not data.startswith(_JPEG_MAGIC):
        return False, "the crop is not a JPEG"
    if len(data) > MAX_BYTES:
        return False, f"the crop is {len(data)} bytes, over the {MAX_BYTES} limit"
    height = jpeg_height(data)
    if height <= 0:
        return False, "the crop has no readable JPEG size"
    if height > int(max_height):
        return False, f"the crop is {height} px tall, over the {int(max_height)} px limit"
    return True, ""


__all__ = [
    "ASPECT_TOLERANCE",
    "INTERVAL_S",
    "MAX_BYTES",
    "MAX_HEIGHT",
    "MIN_SIDE_PX",
    "CropSchedule",
    "crop_box",
    "crop_is_large_enough",
    "crop_is_valid",
    "jpeg_height",
    "scaled_size",
]
