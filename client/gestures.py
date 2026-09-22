"""Жесты руки на клиенте (ТЗ F-306).

MediaPipe Hands живёт НА КОМНАТЕ и работает на CPU: кадр комнаты никуда не
уходит ради жеста. Модуль распознаёт три жеста ТЗ и держит «сколько жест
держится»:

* открытая ладонь дольше :data:`DEFAULT_HOLD_S` секунд к камере — остановить
  TTS (ладонь «стоп»);
* большой палец вверх — подтверждение вместо устного «да» (F-113, задача
  P5-09);
* указательный палец — «что это?» по направлению (F-306, задача P5-10).

Жесты включаются на дом отдельно: флаг приходит комнате настройками дома
(``homes[].settings.gestures``) или своим конфигом ``client.gestures``.
Пакет ``mediapipe`` импортируется ЛЕНИВО, на первом кадре, и его отсутствие —
это честное «жестов нет», а не молчащая функция: комната продолжает говорить
и слышать.

Распознавание считается по расстояниям до запястья, а не по «высоте пальца на
картинке»: человек может держать руку сбоку, и жест не должен зависеть от
того, как повёрнута кисть.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

log = logging.getLogger(__name__)

#: Жест «стоп»: сколько ладонь должна держаться, прежде чем он сработает.
DEFAULT_HOLD_S = 1.0
#: Не чаще одного распознавания в это время: MediaPipe на CPU не должен
#: съедать весь YOLO-поток комнаты.
DEFAULT_INTERVAL_S = 0.2
#: Какие жесты вообще понимает этот модуль.
GESTURES = ("palm", "thumb_up", "point")
#: Индексы точек MediaPipe Hands, на которые опирается распознавание.
WRIST, THUMB_MCP, THUMB_TIP = 0, 2, 4
FINGERS: dict[str, tuple[int, int]] = {
    # имя пальца: (кончик, сустав у ладони)
    "index": (8, 5),
    "middle": (12, 9),
    "ring": (16, 13),
    "pinky": (20, 17),
}


class GestureUnavailable(RuntimeError):
    """Жесты использовать нельзя: нет пакета, нет модели, нет OpenCV."""


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    value = obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)
    return default if value is None else value


def truthy(value: Any) -> bool:
    """Флаг из конфига дома: ``true``/``"true"``/``"on"``/``1`` — да, остальное нет.

    ``bool("false")`` в Python — это ``True``, а настройку дома пишет человек в
    yaml, поэтому строку нужно читать как слово, а не как непустую строку.
    """
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on", "да", "si", "sí"}
    return bool(value)


def _point(landmarks: Sequence[Any], index: int) -> tuple[float, float]:
    """Точка руки: объект MediaPipe (``.x``/``.y``), словарь или пара чисел."""
    item = landmarks[index]
    if isinstance(item, Mapping):
        return float(item.get("x", 0.0)), float(item.get("y", 0.0))
    x = getattr(item, "x", None)
    if x is None:
        return float(item[0]), float(item[1])
    return float(x), float(getattr(item, "y", 0.0))


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def finger_states(landmarks: Sequence[Any]) -> dict[str, bool]:
    """Какой палец выпрямлен: кончик дальше от запястья, чем сустав у ладони.

    Это расстояние не зависит от поворота кисти — «выпрямлен» остаётся
    выпрямленным и для руки сбоку, и для руки сверху.
    """
    states: dict[str, bool] = {}
    wrist = _point(landmarks, WRIST)
    for name, (tip, base) in FINGERS.items():
        states[name] = _distance(_point(landmarks, tip), wrist) > \
            _distance(_point(landmarks, base), wrist) * 1.15
    states["thumb"] = _distance(_point(landmarks, THUMB_TIP), wrist) > \
        _distance(_point(landmarks, THUMB_MCP), wrist) * 1.25
    return states


def recognize(landmarks: Any) -> str:
    """Один жест по 21 точке руки (``""`` — ничего понятного).

    Порядок разбора важен: ладонь — это ЧЕТЫРЕ выпрямленных пальца, поэтому
    «палец вверх» (четыре пальца сжаты) и «указание» (один палец) под ладонь
    не попадают. Сжатая рука — не жест: придумывать «наверное, он хотел»
    нельзя.
    """
    try:
        if landmarks is None or len(landmarks) < 21:
            return ""
    except TypeError:
        return ""
    try:
        states = finger_states(landmarks)
    except (TypeError, ValueError, IndexError):
        return ""
    four = [states["index"], states["middle"], states["ring"], states["pinky"]]
    if all(four):
        return "palm"
    if states["index"] and not any((states["middle"], states["ring"], states["pinky"])):
        return "point"
    # Большой палец вверх — и правда ВВЕРХ: кончик выше запястья, как в жесте
    # «ок, да». Иначе «рука расслаблена» читалась бы как подтверждение.
    if states["thumb"] and not any(four) and _point(landmarks, THUMB_TIP)[1] < \
            _point(landmarks, WRIST)[1] - 0.05:
        return "thumb_up"
    return ""


def point_hint(landmarks: Any, *, reach: float = 1.5) -> tuple[float, float] | None:
    """Куда показывает указательный палец: точка ЗА кончиком (ТЗ F-306).

    Человек показывает пальцем НА предмет, а сам кончик — ещё часть руки,
    поэтому точка продолжается по линии «сустав у ладони — кончик» на
    ``reach`` длин пальца. Точка вне кадра не выдумывается: ``None`` значит
    «направление неизвестно», и хаб тогда спросит про весь кадр.
    """
    try:
        if landmarks is None or len(landmarks) < 21:
            return None
        base = _point(landmarks, FINGERS["index"][1])
        tip = _point(landmarks, FINGERS["index"][0])
    except (TypeError, ValueError, IndexError):
        return None
    x = tip[0] + (tip[0] - base[0]) * float(reach)
    y = tip[1] + (tip[1] - base[1]) * float(reach)
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    return x, y


class GestureHold:
    """«Ладонь дольше секунды»: жест срабатывает один раз, пока не отпустят.

    Без этого правила «стоп» срабатывал бы каждые 200 мс, пока человек держит
    руку, а короткая вспышка ладони в кадре останавливала бы речь.
    """

    def __init__(self, hold_s: float = DEFAULT_HOLD_S) -> None:
        self.hold_s = float(hold_s)
        self._kind = ""
        self._since = 0.0
        self._fired = False

    def observe(self, kind: str, now: float) -> bool:
        """Returns ``True`` ровно один раз — когда жест продержался ``hold_s``."""
        kind = str(kind or "")
        if not kind:
            self._kind, self._since, self._fired = "", 0.0, False
            return False
        if kind != self._kind:
            self._kind, self._since, self._fired = kind, float(now), False
            return False
        if self._fired:
            return False
        if float(now) - self._since >= self.hold_s:
            self._fired = True
            return True
        return False


class MediaPipeHands:
    """MediaPipe Hands за ленивым импортом (ТЗ F-306: ``mediapipe`` на CPU)."""

    def __init__(self, *, max_hands: int = 1, min_detection_confidence: float = 0.5,
                 model_complexity: int = 0) -> None:
        self.max_hands = int(max_hands)
        self.min_detection_confidence = float(min_detection_confidence)
        self.model_complexity = int(model_complexity)
        self._hands: Any = None
        self._error = ""

    def _load(self) -> Any:
        if self._hands is not None:
            return self._hands
        if self._error:
            # Сломанный пакет не переимпортируется на каждом кадре.
            raise GestureUnavailable(self._error)
        try:
            import mediapipe as mp
        except Exception as exc:  # noqa: BLE001 - пакета может не быть вовсе
            self._error = f"mediapipe is not installed ({exc})"
            raise GestureUnavailable(self._error) from exc
        try:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=self.max_hands,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
            )
        except Exception as exc:  # noqa: BLE001 - модель может не загрузиться
            self._error = f"mediapipe Hands could not start ({exc})"
            raise GestureUnavailable(self._error) from exc
        log.info("MediaPipe Hands is up: %d hand(s), complexity %d",
                 self.max_hands, self.model_complexity)
        return self._hands

    def detect(self, frame: Any) -> list[Any]:
        """21 точка каждой руки в кадре (BGR, как его отдаёт OpenCV)."""
        hands = self._load()
        try:
            import cv2

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as exc:  # noqa: BLE001 - без OpenCV жестов нет
            raise GestureUnavailable(f"OpenCV could not prepare the frame ({exc})") from exc
        result = hands.process(rgb)
        found = getattr(result, "multi_hand_landmarks", None) or []
        return [getattr(hand, "landmark", hand) for hand in found]

    def close(self) -> None:
        hands, self._hands = self._hands, None
        if hands is not None:
            try:
                hands.close()
            except Exception as exc:  # noqa: BLE001 - закрытие не наша забота
                log.debug("Could not close MediaPipe Hands (%s)", exc)


class GestureService:
    """Жесты одной комнаты: кадр → жест → событие (и ничего лишнего).

    Сервис всегда конструируем: без флага он просто ничего не делает, а без
    ``mediapipe`` — один раз говорит об этом в лог и замолкает. Наружу он
    исключений не бросает: жесты не стоят ни кадра видео, ни хода комнаты.
    """

    def __init__(self, cfg: Any = None, detector: Any = None,
                 on_event: Callable[[str], None] | None = None) -> None:
        self._enabled = truthy(_attr(cfg, "enabled", False))
        self.hold = GestureHold(float(_attr(cfg, "hold_s", DEFAULT_HOLD_S) or DEFAULT_HOLD_S))
        self.interval_s = max(0.0, float(_attr(cfg, "interval_s", DEFAULT_INTERVAL_S)
                                         or DEFAULT_INTERVAL_S))
        self._detector = detector
        self._hooks: dict[str, list[Callable[[str], None]]] = {}
        if on_event is not None:
            for kind in GESTURES:
                self.on(kind, on_event)
        self._last_run = 0.0
        #: Последнее указание пальцем: ``{"x", "y", "at"}`` (ТЗ F-306). Хаб
        #: спрашивает про «вот это», поэтому точка должна быть свежей.
        self.last_point: dict[str, float] | None = None
        self._errors = 0
        self._warned = ""
        self.events = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, on: Any) -> bool:
        """Включить/выключить жесты дома (ТЗ F-306: «на дом отдельно»)."""
        value = truthy(on)
        if value == self._enabled:
            return False
        self._enabled = value
        self.hold = GestureHold(self.hold.hold_s)
        log.info("Hand gestures %s", "on" if value else "off")
        return True

    def on(self, kind: str, callback: Callable[[str], None]) -> None:
        """Подписаться на жест (``palm``, ``thumb_up``, ``point``)."""
        self._hooks.setdefault(str(kind), []).append(callback)

    def _detect_hands(self, frame: Any) -> list[Any]:
        detector = self._detector
        if detector is None:
            detector = self._detector = MediaPipeHands()
        return list(detector.detect(frame))

    def submit(self, frame: Any, *, now: float | None = None) -> list[str]:
        """Один кадр комнаты: какие жесты сработали на нём (обычно ни одного)."""
        if not self._enabled:
            return []
        moment = time.monotonic() if now is None else float(now)
        if self.interval_s and moment - self._last_run < self.interval_s:
            return []
        self._last_run = moment
        try:
            hands = self._detect_hands(frame)
        except GestureUnavailable as exc:
            if self._warned != str(exc):
                self._warned = str(exc)
                log.warning("Hand gestures are unavailable: %s", exc)
            return []
        except Exception as exc:  # noqa: BLE001 - кадр камеры не должен падать
            self._errors += 1
            if self._errors == 1:
                log.warning("Gesture recognition failed: %s", exc)
            return []
        seen: dict[str, Any] = {}
        for hand in hands:
            name = recognize(hand)
            if name and name not in seen:
                seen[name] = hand
        if "point" in seen:
            hint = point_hint(seen["point"])
            if hint is not None:  # ТЗ F-306: куда показывает палец
                self.last_point = {"x": hint[0], "y": hint[1], "at": moment}
        # Один жест на кадр: порядок рук MediaPipe не обещает, поэтому из
        # увиденных берётся старший по :data:`GESTURES` (ладонь «стоп» важнее
        # остальных), а пустой кадр сбрасывает отсчёт — руку убрали.
        kind = next((item for item in GESTURES if item in seen), "")
        if not self.hold.observe(kind, moment):
            return []
        self.events += 1
        self._fire(kind)
        return [kind]

    def _fire(self, kind: str) -> None:
        for callback in self._hooks.get(kind, ()):
            try:
                callback(kind)
            except Exception as exc:  # noqa: BLE001 - обработчик не ломает камеру
                log.warning("Gesture handler for %s failed (%s)", kind, exc)

    def close(self) -> None:
        detector = self._detector
        closer = getattr(detector, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception as exc:  # noqa: BLE001
                log.debug("Could not close the gesture detector (%s)", exc)


__all__ = [
    "DEFAULT_HOLD_S",
    "DEFAULT_INTERVAL_S",
    "FINGERS",
    "GESTURES",
    "GestureHold",
    "GestureService",
    "GestureUnavailable",
    "MediaPipeHands",
    "finger_states",
    "recognize",
    "truthy",
]
