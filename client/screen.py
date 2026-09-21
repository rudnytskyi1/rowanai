"""Screen capture for the room client (SPEC §7, v1.1/v1.2).

The server's ``look_at_screen`` and ``click_screen`` tools ask this machine for
a picture of its screen: the server sends ``screenshot_request``, and the client
answers with a ``screenshot`` header followed by exactly ONE binary JPEG frame
(SPEC §4).

Capture goes through Pillow's ``ImageGrab`` (Windows GDI) and is therefore
blocking, so callers run it off the event loop::

    from client.screen import capture_jpeg
    capture = await asyncio.to_thread(capture_jpeg)
    # capture.jpeg, capture.w/h (sent image), capture.screen_w/screen_h (desktop)

The image is downscaled to at most :data:`MAX_WIDTH_PX` pixels wide and encoded
as JPEG with quality :data:`JPEG_QUALITY` — enough detail for a vision model
without pushing megabytes over the WebSocket. v1.2 returns the dimensions
alongside the bytes: the vision model answers ``click_screen`` in image pixels,
so the server needs both the image size and the real desktop size to turn that
answer into normalized screen coordinates for the client's ``mouse_click``.

Any failure raises :class:`ScreenCaptureError` with a message the caller
forwards verbatim in ``screenshot_error``.
"""

from __future__ import annotations

import io
import logging
import sys
from dataclasses import dataclass
from typing import Any, Tuple

log = logging.getLogger(__name__)

#: Longest edge of the sent image along X; taller screens keep their aspect ratio.
# The room TV is 3840x2160, so 1024 px was a 3.75x linear (14x pixel) reduction
# and it cost real accuracy, not just sharpness: asked to click YouTube's search
# box, the vision model answered y=0.069 and the cursor landed in the browser's
# address bar one row above it, so the typed query became a Google search. The
# earlier 1024 px choice was made when images went to the 22 GB chat model and
# its prefill dominated the reply (8.7 s -> 1.0 s); screenshots now go to a
# separate 7B vision model, where ~1800 image tokens instead of ~780 costs a
# fraction of a second. 1600x900 also stays well inside that model's 4096-token
# context (~1836 image + ~220 prompt + 200 reserved for the answer).
MAX_WIDTH_PX = 1600
#: JPEG quality used for the encoded screenshot. Costs LAN bytes and nothing
#: else - the token cost of an image depends on its pixel size, not its file
#: size - and q80's chroma subsampling was smearing thin UI glyph strokes.
JPEG_QUALITY = 92
#: Value of the ``format`` field in the ``screenshot`` header (SPEC §4, C->S #6).
SCREENSHOT_FORMAT = "jpeg"

__all__ = [
    "MAX_WIDTH_PX",
    "JPEG_QUALITY",
    "SCREENSHOT_FORMAT",
    "Capture",
    "ScreenCaptureError",
    "capture_jpeg",
]


class ScreenCaptureError(RuntimeError):
    """The screen could not be grabbed or the image could not be encoded."""


@dataclass(frozen=True)
class Capture:
    """One encoded screenshot plus the sizes the ``screenshot`` header carries.

    :param jpeg: the encoded image bytes (the single binary frame).
    :param w: width of the encoded image, after the downscale.
    :param h: height of the encoded image, after the downscale.
    :param screen_w: width of the real desktop, before the downscale.
    :param screen_h: height of the real desktop, before the downscale.
    """

    jpeg: bytes
    w: int
    h: int
    screen_w: int
    screen_h: int


def _ensure_dpi_aware() -> None:
    """Declare this process per-monitor DPI aware, once, best-effort.

    A process Windows considers DPI-unaware is handed a VIRTUALIZED desktop by
    GDI: on this 3840x2160 TV at 300% scaling that would be a 1280x720 grab
    which the OS then stretches, and every pixel coordinate the vision model
    returns would be measured against the wrong geometry. CPython's own
    manifest already declares awareness today, so this changes nothing on the
    current interpreter - it is here so a future embedded/frozen build cannot
    silently start capturing a blurry, wrongly-scaled screen.
    """
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes  # noqa: PLC0415 - lazy, Windows only

        # PROCESS_PER_MONITOR_DPI_AWARE = 2. Fails harmlessly (E_ACCESSDENIED)
        # when awareness was already set by the manifest, which is the norm.
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:  # noqa: BLE001 - older Windows, or already set
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:  # noqa: BLE001 - nothing more to try
            pass


def _load_pillow() -> Tuple[Any, Any]:
    """Import Pillow lazily so a missing dependency cannot break client startup."""
    try:
        from PIL import Image, ImageGrab  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ScreenCaptureError(
            "Pillow is not installed on the room PC. Install the client "
            "dependencies: pip install -r client/requirements.txt"
        ) from exc
    return Image, ImageGrab


def _resample_filter(image_module: Any) -> Any:
    """LANCZOS constant across Pillow versions (moved to ``Image.Resampling`` in 9.1)."""
    resampling = getattr(image_module, "Resampling", None)
    if resampling is not None:
        return resampling.LANCZOS
    return image_module.LANCZOS  # pragma: no cover - Pillow < 9.1


def capture_jpeg(max_width: int = MAX_WIDTH_PX, quality: int = JPEG_QUALITY) -> Capture:
    """Grab the primary screen and return it as JPEG bytes plus its dimensions.

    :param max_width: downscale the grab so it is at most this many pixels wide.
    :param quality: JPEG quality (1..95).
    :returns: a :class:`Capture` with the encoded image and both sizes the
        ``screenshot`` header needs (SPEC §4, C->S #6).
    :raises ScreenCaptureError: capture or encoding failed — the message is
        meant to be sent to the server as ``screenshot_error.error``.
    """
    _ensure_dpi_aware()
    image_module, grab_module = _load_pillow()

    try:
        image = grab_module.grab()
    except Exception as exc:  # noqa: BLE001 - any GDI failure becomes one error
        raise ScreenCaptureError(f"screen capture failed: {exc}") from exc
    if image is None:  # pragma: no cover - depends on the display driver
        raise ScreenCaptureError("screen capture returned no image")

    try:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ScreenCaptureError("screen capture returned an empty image")
        if image.mode != "RGB":
            # JPEG cannot store an alpha channel (ImageGrab may return RGBA).
            image = image.convert("RGB")

        limit = max(1, int(max_width))
        if width > limit:
            scaled_height = max(1, int(round(height * limit / float(width))))
            image = image.resize((limit, scaled_height), _resample_filter(image_module))

        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=max(1, min(95, int(quality))),
            optimize=True,
        )
        encoded_size = image.size
    except ScreenCaptureError:
        raise
    except Exception as exc:  # noqa: BLE001 - encoding must not leak raw errors
        raise ScreenCaptureError(f"screenshot encoding failed: {exc}") from exc
    finally:
        close = getattr(image, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - defensive
                pass

    data = buffer.getvalue()
    if not data:
        raise ScreenCaptureError("screenshot encoding produced no data")
    log.debug(
        "Screenshot captured: %dx%d source, %dx%d sent, %d bytes JPEG",
        width,
        height,
        encoded_size[0],
        encoded_size[1],
        len(data),
    )
    return Capture(
        jpeg=data,
        w=int(encoded_size[0]),
        h=int(encoded_size[1]),
        screen_w=int(width),
        screen_h=int(height),
    )
