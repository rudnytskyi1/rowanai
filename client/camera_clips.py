"""Short silent MP4 clips from the existing capture thread, with bounded work."""
from __future__ import annotations

import asyncio
import math
import tempfile
import threading
import time
from pathlib import Path

from common.protocol import CAMERA_CLIP_MAX_BYTES, MSG_CAMERA_CLIP, MSG_CAMERA_CLIP_ERROR


def clip_settings(seconds=5, fps=8):
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 3 <= seconds <= 10:
        raise ValueError('Clip duration must be between 3 and 10 seconds.')
    if type(fps) is not int or not 5 <= fps <= 10:
        raise ValueError('Clip frame rate must be between 5 and 10 FPS.')
    return float(seconds), fps


def record_clip(camera, seconds=5, fps=8, *, cancel=None):
    """Stream at most 100 scaled frames to a temporary MP4; no frame queue.

    Called only off the event loop. Capture and YOLO continue independently.
    Stalled capture produces an error, not a fake video repeating one photo.
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
            try:
                for index in range(math.ceil(seconds * fps)):
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
                    if (width, height) != source_size:
                        raise RuntimeError('Camera resolution changed during the clip.')
                    if (width, height) != size:
                        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                    writer.write(frame)
                    written += 1
                    if path.exists() and path.stat().st_size > CAMERA_CLIP_MAX_BYTES:
                        raise RuntimeError('The camera clip exceeded its size limit.')
            finally:
                if writer is not None:
                    writer.release()
            if written < math.ceil(seconds * fps * .8) or not path.exists():
                raise RuntimeError('The camera did not provide enough fresh frames for a video.')
            if not 0 < path.stat().st_size <= CAMERA_CLIP_MAX_BYTES:
                raise RuntimeError('The camera clip is empty or exceeds its size limit.')
            data = path.read_bytes()
            if b'ftyp' not in data[:64]:
                raise RuntimeError('The video encoder did not produce a valid MP4 container.')
            return dict(data=data, w=size[0], h=size[1], seconds=written / fps, fps=fps)
    finally:
        camera._clip_lock.release()


async def serve_clip(camera, request_id, seconds=5, fps=8, event_id=''):
    """Capture without holding the socket; hold its wire lock for header+MP4."""
    request_id = str(request_id or '')[:100]
    event_id = str(event_id or '')[:100]
    cancel = threading.Event()
    try:
        result = await asyncio.to_thread(record_clip, camera, seconds, fps, cancel=cancel)
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
