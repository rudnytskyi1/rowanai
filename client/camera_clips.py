"""Short silent MP4 clips from the existing capture thread, with bounded work."""
from __future__ import annotations

import asyncio
import logging
import math
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from client.camera import PREROLL_SECONDS
from common.protocol import CAMERA_CLIP_MAX_BYTES, MSG_CAMERA_CLIP, MSG_CAMERA_CLIP_ERROR

log = logging.getLogger("client.camera_clips")

#: Владелец 2026-09-24: «можешь чтобы оно с bounding box видео записывало и
#: идентификацией человека над ним (label)». One colour per box, chosen by the
#: order of the tracks in the frame, so two people are told apart at a glance.
TRACK_COLOURS = ((0, 200, 255), (0, 255, 120), (255, 170, 0), (120, 120, 255), (255, 120, 200))
#: How much of a clip may come from the pre-roll. A 5 s alert video is then 3 s
#: of the moment the person was seen plus 2 s of what happens next.
PREROLL_SHARE = 0.6


def _label_font(size: int):
    """A TrueType font that can draw Cyrillic names, or ``None`` without PIL."""
    try:
        from PIL import ImageFont
    except Exception:  # pragma: no cover - PIL ships with the client
        return None
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return None


def draw_tracks(frame, tracks, names=None, *, cv2=None, copy=True):
    """Draw the person boxes of one frame with the names the hub knows.

    ``tracks`` is the client's own shape - ``{"id": ..., "box": [x1, y1, x2, y2]}``
    with 0..1 coordinates - and ``names`` maps a track id to the display name the
    identity layer gave it. A track nobody was identified in gets its box only:
    the video must not invent a name (AGENTS.md). Text is drawn through PIL when
    it is there, because OpenCV cannot draw Cyrillic.
    """
    if cv2 is None or frame is None or not tracks:
        return frame
    if copy:
        # The capture thread's frame is shared with detection, crops and photo
        # requests: never draw into it. (A pre-roll frame is already ours.)
        frame = frame.copy()
    height, width = frame.shape[:2]
    if min(height, width) < 2:
        return frame
    thickness = max(1, int(round(max(height, width) / 480)))
    pending_labels: list[tuple[str, int, int, tuple[int, int, int], int]] = []
    for index, track in enumerate(tracks):
        if not isinstance(track, dict):
            continue
        box = track.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (max(0.0, min(1.0, float(value))) for value in box)
        except (TypeError, ValueError):
            continue
        if x2 - x1 < 0.01 or y2 - y1 < 0.01:
            continue
        left, top = int(x1 * width), int(y1 * height)
        right, bottom = int(x2 * width), int(y2 * height)
        colour = TRACK_COLOURS[index % len(TRACK_COLOURS)]
        cv2.rectangle(frame, (left, top), (right, bottom), colour, thickness)
        name = ""
        if isinstance(names, dict):
            name = str(names.get(str(track.get("id"))) or "").strip()
        if name:
            pending_labels.append((name, left, top, colour, max(11, int(height / 28))))
    if not pending_labels:
        return frame
    font = _label_font(pending_labels[0][4])
    if font is not None:
        try:
            import numpy as np
            from PIL import Image, ImageDraw

            # ``frame[:, :, ::-1]`` is a view with a negative stride: PIL needs a
            # contiguous array, otherwise it raises and the name silently fell
            # back to the ASCII path (which cannot draw Cyrillic at all).
            image = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))
            painter = ImageDraw.Draw(image)
            for text, left, top, colour, _size in pending_labels:
                box = painter.textbbox((0, 0), text, font=font)
                text_width, text_height = box[2] - box[0], box[3] - box[1]
                y = top - text_height - 8
                if y < 0:
                    y = top + 2
                painter.rectangle((left, y, left + text_width + 10, y + text_height + 6),
                                  fill=tuple(int(value) for value in colour))
                painter.text((left + 5, y + 2), text, font=font, fill=(16, 16, 16))
            frame[:, :, :] = np.asarray(image)[:, :, ::-1]
            return frame
        except Exception:  # noqa: BLE001 - a missing font must not stop the clip
            log.debug("Could not draw the names with PIL; using plain boxes", exc_info=True)
    for text, left, top, colour, size in pending_labels:
        # No PIL: ASCII only, so a Cyrillic name becomes its own letters rather
        # than a row of question marks.
        ascii_text = text.encode("ascii", "ignore").decode("ascii").strip()
        if not ascii_text:
            continue
        cv2.putText(frame, ascii_text, (left + 2, max(12, top - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, size / 28.0, colour, thickness, cv2.LINE_AA)
    return frame


def _cancelled(cancel: Any) -> bool:
    """Whether a cancellation was asked for; a plain stub counts as "no"."""
    checker = getattr(cancel, "is_set", None)
    return bool(checker()) if callable(checker) else False


def clip_settings(seconds=5, fps=8):
    # 60 s is the ceiling of ONE video of an alert episode (ТЗ F-702): a longer
    # visit is recorded as several of these, never as one oversized file.
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 3 <= seconds <= 60:
        raise ValueError('Clip duration must be between 3 and 60 seconds.')
    if type(fps) is not int or not 5 <= fps <= 10:
        raise ValueError('Clip frame rate must be between 5 and 10 FPS.')
    return float(seconds), fps


def record_clip(camera, seconds=5, fps=8, *, cancel=None, names=None, preroll=None):
    """Stream scaled frames to a temporary MP4; no frame queue.

    Called only off the event loop. Capture and YOLO continue independently.
    Stalled capture produces an error, not a fake video repeating one photo.
    The size check runs after every frame, so an unusually busy scene stops the
    recording honestly instead of producing a file the hub would refuse.

    Two owner asks of 2026-09-24 shape this loop. The video shows **who** was
    seen: every frame carries the person boxes and the names the hub sent with
    the request (``names``). And it starts **in the past**: the first part comes
    from the camera's own pre-roll (``camera.preroll_frames``), because a rule
    fires a second or two after somebody walked past - without that the clip was
    an empty room. ``preroll=0`` turns the pre-roll off (the later parts of one
    alert episode do not need the same seconds again).
    """
    seconds, fps = clip_settings(seconds, fps)
    cancel = cancel or threading.Event()
    cv2 = camera._cv2
    if not camera.enabled or cv2 is None:
        raise RuntimeError('The camera is unavailable.')
    if not camera._clip_lock.acquire(blocking=False):
        raise RuntimeError('Another camera clip is already being recorded.')
    writer = None
    try:
        with tempfile.TemporaryDirectory(prefix='rowan-camera-') as directory:
            path = Path(directory) / 'clip.mp4'
            started = time.monotonic()
            previous = started
            written = 0
            size = source_size = None
            total = math.ceil(seconds * fps)
            budget = PREROLL_SECONDS if preroll is None else float(preroll)
            history = []
            if budget > 0 and hasattr(camera, 'preroll_frames'):
                history = camera.preroll_frames(min(budget, seconds * PREROLL_SHARE))
            try:
                for _at, image, tracks in history[:total]:
                    if _cancelled(cancel) or camera._stop_event.is_set():
                        raise RuntimeError('Camera clip capture was cancelled.')
                    if writer is None:
                        size = (image.shape[1], image.shape[0])
                        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'),
                                                 fps, size)
                        if not writer.isOpened():
                            raise RuntimeError('MP4 encoding is unavailable on this client.')
                    writer.write(draw_tracks(image, tracks, names, cv2=cv2, copy=False))
                    written += 1
            except Exception:
                if writer is not None:
                    writer.release()
                    writer = None
                raise
            try:
                for index in range(total - written):
                    if time.monotonic() > started + seconds + 2:
                        raise RuntimeError('Camera clip encoding exceeded its time limit.')
                    delay = started + (index + 1) / fps - time.monotonic()
                    if cancel.wait(max(0., delay)) or camera._stop_event.is_set():
                        raise RuntimeError('Camera clip capture was cancelled.')
                    frame, captured_at = camera._latest_frame_ts()
                    if (frame is None or captured_at <= previous
                            or time.monotonic() - captured_at > 1):
                        continue
                    previous = captured_at
                    height, width = frame.shape[:2]
                    if min(width, height) < 2:
                        continue
                    if writer is None:
                        source_size = (width, height)
                        scale = min(1., 960 / max(width, height))
                        size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
                        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
                        if not writer.isOpened():
                            raise RuntimeError('MP4 encoding is unavailable on this client.')
                    if source_size is None:
                        source_size = (width, height)
                    if (width, height) != source_size:
                        raise RuntimeError('Camera resolution changed during the clip.')
                    frame = draw_tracks(frame, getattr(camera, '_tracks', None), names, cv2=cv2)
                    if (frame.shape[1], frame.shape[0]) != size:
                        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                    writer.write(frame)
                    written += 1
                    if path.exists() and path.stat().st_size > CAMERA_CLIP_MAX_BYTES:
                        raise RuntimeError('The camera clip exceeded its size limit.')
            finally:
                if writer is not None:
                    writer.release()
            if written < total * .8 or not path.exists():
                raise RuntimeError('The camera did not provide enough fresh frames for a video.')
            if not 0 < path.stat().st_size <= CAMERA_CLIP_MAX_BYTES:
                raise RuntimeError('The camera clip is empty or exceeds its size limit.')
            data = path.read_bytes()
            if b'ftyp' not in data[:64]:
                raise RuntimeError('The video encoder did not produce a valid MP4 container.')
            return dict(data=data, w=size[0], h=size[1], seconds=written / fps, fps=fps)
    finally:
        camera._clip_lock.release()


async def serve_clip(camera, request_id, seconds=5, fps=8, event_id='', names=None,
                     preroll=None):
    """Capture without holding the socket; hold its wire lock for header+MP4."""
    request_id = str(request_id or '')[:100]
    event_id = str(event_id or '')[:100]
    cancel = threading.Event()
    try:
        result = await asyncio.to_thread(record_clip, camera, seconds, fps, cancel=cancel,
                                        names=names, preroll=preroll)
        header = dict(type=MSG_CAMERA_CLIP, id=request_id, format='mp4', bytes=len(result['data']),
                      **{key: result[key] for key in ('w', 'h', 'seconds', 'fps')})
        if event_id:
            # ТЗ 4.5: the clip is the answer to one background camera event.
            header['event_id'] = event_id

        async def send():
            await camera._send_json(header)
            await camera._send_bytes(result['data'])

        if camera._send_lock is None:
            await send()
        else:
            async with camera._send_lock:
                await send()
    except asyncio.CancelledError:
        cancel.set()
        raise
    except Exception as exc:
        # Encoder/driver errors may expose device URLs. Return fixed safe text.
        error = str(exc) if isinstance(exc, (ValueError, RuntimeError)) and str(exc).startswith(
            ('Clip ', 'The camera ', 'The video ', 'Camera ', 'MP4 ', 'Another ')) else 'Camera clip capture failed.'
        if camera._send_json is not None:
            failure = dict(type=MSG_CAMERA_CLIP_ERROR, id=request_id, error=error)
            if event_id:
                failure['event_id'] = event_id
            await camera._send_json(failure)
