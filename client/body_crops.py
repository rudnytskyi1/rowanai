"""Кроп тела, вырезанный на клиенте (ТЗ F-202).

The hub never sees the room's full frames unless it asks for one: what travels
for identification is a full-height JPEG of each person, at most 640 px tall,
cut on the room PC and sent on appearance, every two seconds and whenever the
person turns. This module owns the cutting; the rule of WHEN to cut lives in
:mod:`common.body_crops`, shared with the hub.

``cv2`` is imported by the camera service and passed in, so this module works
with whatever OpenCV the client has (and can be tested with a stub).
"""
from __future__ import annotations

import logging
from typing import Any

from common.body_crops import (
    INTERVAL_S,
    MAX_HEIGHT,
    MIN_SIDE_PX,
    CropSchedule,
    crop_box,
    crop_is_large_enough,
    crop_is_valid,
    jpeg_height,
    scaled_size,
)

log = logging.getLogger(__name__)

#: JPEG quality of a body crop: the hub runs a ReID model on it (F-203), so it
#: deserves more than the pushed preview frames but still travels cheaply.
CROP_JPEG_QUALITY = 90


def encode_crop(frame: Any, bbox: Any, cv2_module: Any, *,
                max_height: int = MAX_HEIGHT,
                quality: int = CROP_JPEG_QUALITY) -> tuple[bytes, int, int] | None:
    """Cut one person out of ``frame`` and encode it as a JPEG.

    ``frame`` is an OpenCV BGR image, ``bbox`` a normalized ``(x1, y1, x2, y2)``.
    Returns ``(jpeg, width, height)`` or ``None`` when there is nothing worth
    sending (a box of a few pixels, a frame that will not crop, a JPEG the hub
    would refuse anyway).
    """
    if frame is None or cv2_module is None:
        return None
    try:
        height, width = int(frame.shape[0]), int(frame.shape[1])
    except (AttributeError, IndexError, TypeError):
        return None
    box = crop_box(tuple(bbox), width, height)
    if not crop_is_large_enough(box, min_side=MIN_SIDE_PX):
        return None
    left, top, right, bottom = box
    target = scaled_size(right - left, bottom - top, max_height=max_height)
    try:
        patch = frame[top:bottom, left:right]
        if patch is None or getattr(patch, "size", 0) == 0:
            return None
        if (target[0], target[1]) != (right - left, bottom - top):
            patch = cv2_module.resize(patch, target, interpolation=cv2_module.INTER_AREA)
        ok, encoded = cv2_module.imencode(
            ".jpg", patch, [int(cv2_module.IMWRITE_JPEG_QUALITY), int(quality)])
    except Exception as exc:  # noqa: BLE001 - a broken crop must not stop the camera
        log.debug("Could not cut a body crop (%s)", exc)
        return None
    if not ok:
        return None
    data = encoded.tobytes()
    valid, reason = crop_is_valid(data, max_height=max_height)
    if not valid:
        # Our own crop that the hub would refuse is not sent at all: a bad
        # request costs a round trip and teaches nobody anything.
        log.debug("Refusing our own body crop (%s)", reason)
        return None
    return data, target[0], target[1]


__all__ = [
    "CROP_JPEG_QUALITY",
    "INTERVAL_S",
    "MAX_HEIGHT",
    "MIN_SIDE_PX",
    "CropSchedule",
    "crop_box",
    "crop_is_valid",
    "encode_crop",
    "jpeg_height",
    "scaled_size",
]
