"""Live camera preview for the room screen (владелец, 2026-09-24).

Просьба владельца: «можешь на пк buro открыть камеру и чтобы оно показывало
видео с камеры на экране и все детекции?». До этого комната умела показать
только отдельный КАДР (``image_show``) и подписи поверх экрана (HUD) — живого
видео с боксами не было нигде.

Здесь одно окно OpenCV, которое живёт в своём потоке и рисует:

* кадр с той же камеры, что обслуживает комнату (поток захвата уже держит
  устройство: второй ``VideoCapture`` на том же ПК открыть нельзя);
* **все** боксы последнего вывода детектора — класс и уверенность, а не только
  люди и «объекты внимания»;
* подписи людей: имя, если хаб уже знает трек, иначе идентификатор трека.

Окно живёт ровно столько, сколько его держит хаб: ``camera_preview`` с
``on: false`` (или разрыв связи, или выход клиента) закрывает его. Ошибка окна
никогда не трогает голосовой контур — как и у остальных «экранных» частей
клиента, всё здесь best-effort.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from client.camera_clips import TRACK_COLOURS

log = logging.getLogger(__name__)

#: Сколько раз в секунду окно обновляется. Камера отдаёт 30 кадров/с, а
#: рисование боксов и ``imshow`` дешевле самого инференса; 20 кадров/с дают
#: живое видео без лишней нагрузки на слабый комнатный ПК.
PREVIEW_FPS = 20.0
#: Самая длинная сторона кадра, который рисуется и показывается. Комната отдаёт
#: 1920x1080; рисовать и перегонять в PIL полный кадр 20 раз в секунду на слабом
#: ПК слишком дорого (на buro выходило ~4 кадра/с вместо 20), а на экране всё
#: равно 960 px. Кадр камеры при этом не меняется - масштабируется копия.
PREVIEW_MAX_SIDE = 960
#: Пауза между попытками показать кадр, когда камера молчит.
IDLE_WAIT_S = 0.1
#: Сколько ждём, пока пользователь закроет окно, при выключении просмотра.
JOIN_TIMEOUT_S = 2.0
#: Цвета: люди — палитра клипов, остальные находки — один спокойный цвет.
OBJECT_COLOUR = (170, 170, 170)
HEADER_COLOUR = (32, 32, 32)
HEADER_TEXT_COLOUR = (240, 240, 240)


def _label_font(size: int):
    """Шрифт с кириллицей, как в клипах; без PIL остаются латинские буквы."""
    from client.camera_clips import _label_font as font

    return font(size)


class LivePreview:
    """One OpenCV window showing the room camera with every detection."""

    def __init__(self, camera: Any, label: str = '') -> None:
        self.camera = camera
        self.label = str(label or '')
        self._thread: threading.Thread | None = None
        self._closing = threading.Event()
        self._names: dict[str, str] = {}
        self._names_lock = threading.Lock()
        self._window = ''
        self._error = ''
        self._frames = 0
        self._started_at = 0.0

    # -- lifecycle --------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def update_names(self, names: Any) -> None:
        """The track→name map the hub knows; the window picks it up at once."""
        clean: dict[str, str] = {}
        if isinstance(names, dict):
            for key, value in names.items():
                if isinstance(key, str) and isinstance(value, str) and key and value:
                    clean[key[:100]] = value[:120]
        with self._names_lock:
            self._names = clean

    def start(self) -> bool:
        """Open the window; ``False`` when the client has no usable display."""
        if self.running:
            return True
        if getattr(self.camera, '_cv2', None) is None:
            self._error = 'the client has no OpenCV build'
            return False
        if not getattr(self.camera, 'enabled', False):
            self._error = 'the camera of this room is off'
            return False
        self._closing.clear()
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name='rowan-live-preview',
                                        daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._closing.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=JOIN_TIMEOUT_S)

    def status(self) -> dict[str, Any]:
        """Honest state for the hub: what the window is actually doing."""
        return {'running': self.running, 'label': self.label, 'frames': self._frames,
                'error': self._error}

    # -- the window thread -------------------------------------------------
    def _run(self) -> None:
        cv2 = getattr(self.camera, '_cv2', None)
        if cv2 is None:  # pragma: no cover - start() already checked
            return
        title = f"Rowan - camera {self.label}".strip()
        self._window = title
        interval = 1.0 / PREVIEW_FPS
        try:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 960, 540)
        except Exception as exc:  # noqa: BLE001 - a window is not a voice feature
            self._error = f'the window could not be opened ({type(exc).__name__})'
            log.warning('Live preview window is unavailable: %s', exc)
            return
        log.info('Live camera preview started (%s)', title)
        try:
            while not self._closing.is_set():
                started = time.monotonic()
                frame = self._next_frame()
                if frame is None:
                    self._closing.wait(IDLE_WAIT_S)
                    continue
                drawn = self._draw(cv2, frame)
                try:
                    cv2.imshow(title, drawn)
                    cv2.waitKey(1)
                except Exception as exc:  # noqa: BLE001 - the user closed the window
                    log.info('Live camera preview closed by the user (%s)', exc)
                    break
                self._frames += 1
                remaining = interval - (time.monotonic() - started)
                if remaining > 0:
                    self._closing.wait(remaining)
        finally:
            self._close_window(cv2, title)
            log.info('Live camera preview stopped (%s frames)', self._frames)

    def _close_window(self, cv2: Any, title: str) -> None:
        try:
            cv2.destroyWindow(title)
        except Exception:  # noqa: BLE001 - already gone
            pass
        try:
            # Одно нажатие клавиши, иначе окно остаётся «призраком» и следующий
            # показ открывается уже мёртвым (поведение HighGUI на Windows).
            cv2.waitKey(1)
        except Exception:  # noqa: BLE001
            pass

    def _next_frame(self) -> Any:
        """The newest captured frame, or ``None`` while the camera is quiet."""
        reader = getattr(self.camera, '_latest_frame_ts', None)
        if not callable(reader):
            return None
        try:
            frame, captured_at = reader()
        except Exception:  # noqa: BLE001 - a stalled camera is not an error here
            return None
        if frame is None or not getattr(frame, 'size', 0):
            return None
        if time.monotonic() - float(captured_at) > 1.0:
            return None
        return frame

    def _draw(self, cv2: Any, frame: Any) -> Any:
        """Every detection of the last inference plus the names of the people."""
        with self._names_lock:
            names = dict(self._names)
        tracks = list(getattr(self.camera, '_tracks', None) or ())
        image = self._fit(cv2, frame)
        height, width = image.shape[:2]
        thickness = max(1, int(round(max(height, width) / 480)))
        # Тексты копятся и рисуются ОДНИМ проходом PIL: каждый переход
        # numpy -> PIL -> numpy на кадре 960x540 стоит миллисекунды, и два
        # таких прохода (подписи людей + шапка) съедали половину кадров на buro.
        texts: list[tuple[str, int, int, tuple[int, int, int], tuple[int, int, int] | None]] = []
        for index, track in enumerate(tracks):
            box = track.get('box') if isinstance(track, dict) else None
            placed = self._box_pixels(box, width, height)
            if placed is None:
                continue
            left, top, right, bottom = placed
            colour = TRACK_COLOURS[index % len(TRACK_COLOURS)]
            cv2.rectangle(image, (left, top), (right, bottom), colour, thickness)
            name = ''
            if isinstance(names, dict):
                name = str(names.get(str(track.get('id'))) or '').strip()
            if name:
                texts.append((name, left, top, colour, None))
        person_boxes = [track.get('box') for track in tracks if isinstance(track, dict)]
        size = max(11, int(height / 32))
        for detection in list(getattr(self.camera, '_last_detections', None) or ()):
            if not isinstance(detection, dict):
                continue
            box = detection.get('box')
            if isinstance(box, (list, tuple)) and len(box) == 4 and \
                    detection.get('label') == 'person' and box in person_boxes:
                continue  # уже обведён вместе с именем
            placed = self._box_pixels(box, width, height)
            if placed is None:
                continue
            left, top, right, bottom = placed
            cv2.rectangle(image, (left, top), (right, bottom), OBJECT_COLOUR, thickness)
            label = str(detection.get('label') or '').strip()
            text = f'{label} {float(detection.get("conf") or 0.0):.2f}'.strip()
            if text:
                texts.append((text, left, top, OBJECT_COLOUR, None))
        bar = max(22, int(height / 24))
        cv2.rectangle(image, (0, 0), (width, bar), HEADER_COLOUR, -1)
        texts.append((self._header_text(len(tracks)), 8, max(2, bar // 6),
                      HEADER_TEXT_COLOUR, None))
        self._draw_texts(cv2, image, texts, size)
        return image

    @staticmethod
    def _fit(cv2: Any, frame: Any) -> Any:
        """A copy of the frame scaled to what the window actually shows."""
        height, width = frame.shape[:2]
        scale = min(1.0, PREVIEW_MAX_SIDE / float(max(width, height) or 1))
        if scale >= 1.0:
            return frame.copy()
        try:
            return cv2.resize(frame, (max(2, int(width * scale) // 2 * 2),
                                      max(2, int(height * scale) // 2 * 2)),
                              interpolation=cv2.INTER_AREA)
        except Exception:  # noqa: BLE001 - a plain copy still draws
            return frame.copy()

    @staticmethod
    def _box_pixels(box: Any, width: int, height: int):
        """Normalized box -> (left, top, right, bottom), or ``None`` when useless."""
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return None
        try:
            x1, y1, x2, y2 = (max(0.0, min(1.0, float(value))) for value in box)
        except (TypeError, ValueError):
            return None
        if x2 - x1 < 0.01 or y2 - y1 < 0.01:
            return None
        return (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height))

    def _header_text(self, people: int) -> str:
        """A one-line caption: which room, how live, how many people."""
        fps = 0.0
        span = time.monotonic() - self._started_at
        if span > 0:
            fps = self._frames / span
        return f'Rowan live {self.label} | {fps:.0f} fps | people {people}'.strip()

    def _draw_texts(self, cv2: Any, image: Any, texts: Any, size: int) -> None:
        """Labels in one PIL pass, or with OpenCV when PIL is missing."""
        if not texts:
            return
        height, width = image.shape[:2]
        font = _label_font(size)
        if font is None:
            for text, left, top, colour, _ in texts:
                ascii_text = str(text).encode('ascii', 'ignore').decode('ascii').strip()
                if ascii_text:
                    cv2.putText(image, ascii_text, (left + 2, max(12, top - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, size / 28.0, colour, 1, cv2.LINE_AA)
            return
        try:
            import numpy as np
            from PIL import Image, ImageDraw

            picture = Image.fromarray(np.ascontiguousarray(image[:, :, ::-1]))
            painter = ImageDraw.Draw(picture)
            for text, left, top, colour, _ in texts:
                box_at = painter.textbbox((0, 0), str(text), font=font)
                text_width = box_at[2] - box_at[0]
                text_height = box_at[3] - box_at[1]
                y = top - text_height - 8
                if y < 0:
                    y = top + 2
                painter.rectangle((left, y, min(width, left + text_width + 10),
                                   y + text_height + 6), fill=tuple(int(v) for v in colour))
                painter.text((left + 5, y + 2), str(text), font=font, fill=(16, 16, 16))
            image[:, :, :] = np.asarray(picture)[:, :, ::-1]
        except Exception:  # noqa: BLE001 - a missing font is not a missing preview
            log.debug('Could not draw the live-preview labels', exc_info=True)


__all__ = ['LivePreview', 'PREVIEW_FPS', 'TRACK_COLOURS']
