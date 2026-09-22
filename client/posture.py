"""Поза и сон на клиенте (ТЗ F-307).

ТЗ F-307: «YOLO11-pose на клиенте с низкой частотой (1 кадр в 5 с): „лежит
неподвижно дольше 10 мин в тихие часы“ → режим сна (свет на минимум,
уведомления беззвучные); „встал утром“ → утренняя рутина».

Здесь живёт первая половина: комната сама решает, что человек ЛЕЖИТ и НЕ
ДВИГАЕТСЯ, и сама же замечает, что он ВСТАЛ. Наружу уходят только два
события (``sleep`` и ``awake``) — кадры остаются в комнате. Модель позы
грузится ЛЕНИВО: нет ``ultralytics`` или весов — это честное «позы нет», а не
молчащая функция.

Поза считается по ТЕЛУ (плечи и бёдра), а не по «высоте картинки»: лежащий
человек виден сбоку, и правило «прямоугольник выше, чем шире» врало бы на
каждом повороте. Движение — смещение центра тела между кадрами: 5 секунд
между замерами и есть та низкая частота, которую просит ТЗ.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

log = logging.getLogger(__name__)

#: ТЗ F-307: один кадр в 5 секунд — чаще поза не нужна и стоит CPU.
DEFAULT_INTERVAL_S = 5.0
#: ТЗ F-307: «лежит неподвижно дольше 10 мин».
DEFAULT_STILL_S = 600.0
#: Насколько центр тела должен сместиться между замерами, чтобы это было
#: движением (в долях кадра). Лежащий человек дышит — это не движение.
MOVE_THRESHOLD = 0.05
#: Минимальная уверенность точки, чтобы ей верить.
MIN_CONF = 0.3

LYING, SITTING, STANDING = "lying", "sitting", "standing"

#: COCO-точки YOLO-pose, на которые опирается разбор.
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16


class PoseUnavailable(RuntimeError):
    """Позу использовать нельзя: нет пакета, весов или кадра."""


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    value = obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)
    return default if value is None else value


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on", "да", "si", "sí"}
    return bool(value)


def _point(keypoints: Sequence[Any], index: int) -> tuple[float, float, float] | None:
    """Точка позы ``(x, y, conf)``: объект YOLO, словарь и пара чисел — все годятся."""
    try:
        item = keypoints[index]
    except (TypeError, IndexError):
        return None
    if item is None:
        return None
    if isinstance(item, Mapping):
        x, y, conf = item.get("x", 0.0), item.get("y", 0.0), item.get("conf", 1.0)
    elif hasattr(item, "x"):
        x = getattr(item, "x", 0.0)
        y = getattr(item, "y", 0.0)
        conf = getattr(item, "conf", getattr(item, "confidence", 1.0))
    else:
        x, y = item[0], item[1]
        conf = item[2] if len(item) > 2 else 1.0
    try:
        return float(x), float(y), float(conf)
    except (TypeError, ValueError):
        return None


def body_centre(keypoints: Any) -> tuple[float, float] | None:
    """Центр тела (середина плеч и бёдер) или ``None``, если точек нет."""
    try:
        if keypoints is None or len(keypoints) < 17:
            return None
    except TypeError:
        return None
    points = [_point(keypoints, index)
              for index in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)]
    good = [item for item in points if item is not None and item[2] >= MIN_CONF]
    if not good:
        return None
    return (sum(item[0] for item in good) / len(good),
            sum(item[1] for item in good) / len(good))


def posture_of(keypoints: Any, *, min_conf: float = MIN_CONF) -> str:
    """``"lying"`` / ``"sitting"`` / ``"standing"`` или ``""`` (непонятно).

    Лежит человек или стоит — решает ТУЛОВИЩЕ: вектор «бёдра минус плечи» почти
    горизонтален, значит тело горизонтально. Стоит или сидит — решает нога:
    видимая и не сложённая в колене — это «стоит». Не хватает точек — пустой
    ответ: придумывать позу нельзя.
    """
    try:
        if keypoints is None or len(keypoints) < 17:
            return ""
    except TypeError:
        return ""
    shoulders = [item for item in (_point(keypoints, index)
                                  for index in (LEFT_SHOULDER, RIGHT_SHOULDER))
                 if item is not None and item[2] >= min_conf]
    hips = [item for item in (_point(keypoints, index) for index in (LEFT_HIP, RIGHT_HIP))
            if item is not None and item[2] >= min_conf]
    if not shoulders or not hips:
        return ""
    shoulder = (sum(item[0] for item in shoulders) / len(shoulders),
                sum(item[1] for item in shoulders) / len(shoulders))
    hip = (sum(item[0] for item in hips) / len(hips), sum(item[1] for item in hips) / len(hips))
    dx, dy = hip[0] - shoulder[0], hip[1] - shoulder[1]
    if abs(dy) <= abs(dx) * 0.7:
        # Туловище ближе к горизонтали, чем к вертикали.
        return LYING
    knees = [item for item in (_point(keypoints, index) for index in (LEFT_KNEE, RIGHT_KNEE))
             if item is not None and item[2] >= min_conf]
    ankles = [item for item in (_point(keypoints, index) for index in (LEFT_ANKLE, RIGHT_ANKLE))
              if item is not None and item[2] >= min_conf]
    if not knees or not ankles:
        # Ноги не видны (человек за столом) — это «сидит», а не «стоит»:
        # утверждать «встал» без ног значило бы будить утреннюю рутину зря.
        return SITTING
    knee = (sum(item[0] for item in knees) / len(knees),
            sum(item[1] for item in knees) / len(knees))
    ankle = (sum(item[0] for item in ankles) / len(ankles),
             sum(item[1] for item in ankles) / len(ankles))
    leg = ((ankle[0] - hip[0]) ** 2 + (ankle[1] - hip[1]) ** 2) ** 0.5
    folded = ((knee[0] - hip[0]) ** 2 + (knee[1] - hip[1]) ** 2) ** 0.5 + \
        ((ankle[0] - knee[0]) ** 2 + (ankle[1] - knee[1]) ** 2) ** 0.5
    return STANDING if leg > 0 and folded <= leg * 1.25 else SITTING


class StillnessWatch:
    """«Лежит неподвижно дольше 10 минут» и «встал» (ТЗ F-307).

    Копит время лежания с последнего заметного движения и отдаёт ``sleep``
    ровно один раз; ``awake`` приходит, когда человек из лежачего положения
    оказался в сидячем или стоячем. Пропажа из кадра подъёмом НЕ считается:
    человек мог просто укрыться одеялом, и будить рутину по пустому кадру
    нельзя.
    """

    def __init__(self, still_s: float = DEFAULT_STILL_S,
                 move_threshold: float = MOVE_THRESHOLD) -> None:
        self.still_s = float(still_s)
        self.move_threshold = float(move_threshold)
        self._since = 0.0
        self._centre: tuple[float, float] | None = None
        self._asleep = False

    @property
    def asleep(self) -> bool:
        return self._asleep

    def reset(self) -> None:
        """Забыть наблюдение (жесты/поза выключены, событие не пригодилось)."""
        self._since, self._centre, self._asleep = 0.0, None, False

    def observe(self, posture: str, centre: tuple[float, float] | None, now: float) -> str:
        """Returns ``"sleep"``, ``"awake"`` or ``""``."""
        if posture == LYING:
            moved = (self._centre is None or centre is None or
                     abs(centre[0] - self._centre[0]) > self.move_threshold or
                     abs(centre[1] - self._centre[1]) > self.move_threshold)
            if moved:
                self._since = float(now)
            self._centre = centre
            if not self._asleep and float(now) - self._since >= self.still_s:
                self._asleep = True
                return "sleep"
            return ""
        # Из лежачего положения человек сел или встал — это и есть «встал».
        was_asleep = self._asleep
        self._asleep = False
        self._since = 0.0
        self._centre = centre
        if posture in (SITTING, STANDING) and was_asleep:
            return "awake"
        return ""


class PoseDetector:
    """YOLO11-pose за ленивым импортом (ТЗ F-307)."""

    def __init__(self, model: str = "yolo11n-pose.pt", *, confidence: float = 0.4) -> None:
        self.model_name = str(model or "yolo11n-pose.pt")
        self.confidence = float(confidence)
        self._model: Any = None
        self._error = ""

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if self._error:
            raise PoseUnavailable(self._error)
        try:
            from ultralytics import YOLO
        except Exception as exc:  # noqa: BLE001 - пакета может не быть
            self._error = f"ultralytics is not installed ({exc})"
            raise PoseUnavailable(self._error) from exc
        try:
            self._model = YOLO(self.model_name)
        except Exception as exc:  # noqa: BLE001 - веса могут не загрузиться
            self._error = f"YOLO pose could not load {self.model_name} ({exc})"
            raise PoseUnavailable(self._error) from exc
        log.info("YOLO pose is up: %s", self.model_name)
        return self._model

    def detect(self, frame: Any) -> list[dict[str, Any]]:
        """По одной записи на человека: ``{"keypoints": [...], "bbox": [...]}``."""
        model = self._load()
        results = model.predict(source=frame, conf=self.confidence, verbose=False)
        people: list[dict[str, Any]] = []
        for result in results or []:
            keypoints = getattr(result, "keypoints", None)
            if keypoints is None:
                continue
            data = getattr(keypoints, "data", None)
            rows = data.tolist() if hasattr(data, "tolist") else (data or [])
            for row in rows:
                people.append({"keypoints": list(row), "bbox": []})
        return people


class PostureService:
    """Поза комнаты: кадр → «лежит/сидит/стоит» → события сна и подъёма."""

    def __init__(self, cfg: Any = None, detector: Any = None,
                 on_event: Callable[[str], None] | None = None) -> None:
        self._enabled = _truthy(_attr(cfg, "enabled", False))
        self.interval_s = max(0.0, float(_attr(cfg, "interval_s", DEFAULT_INTERVAL_S)
                                         or DEFAULT_INTERVAL_S))
        self.watch = StillnessWatch(float(_attr(cfg, "still_s", DEFAULT_STILL_S)
                                          or DEFAULT_STILL_S))
        self._detector = detector
        self._hook = on_event
        self._last_run = 0.0
        self._errors = 0
        self._warned = ""
        self.posture = ""
        self.events = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, on: Any) -> bool:
        value = _truthy(on)
        if value == self._enabled:
            return False
        self._enabled = value
        self.watch = StillnessWatch(self.watch.still_s)
        log.info("Posture watching %s", "on" if value else "off")
        return True

    def _detect_people(self, frame: Any) -> list[dict[str, Any]]:
        detector = self._detector
        if detector is None:
            detector = self._detector = PoseDetector()
        return list(detector.detect(frame))

    def submit(self, frame: Any, *, now: float | None = None, quiet: bool = False) -> list[str]:
        """Один кадр комнаты. ``quiet`` — тихие часы дома (ТЗ F-307).

        Режим сна рождается ТОЛЬКО в тихие часы: днём человек, читающий лёжа,
        — не спящий. Подъём, наоборот, замечается всегда: утренняя рутина не
        должна зависеть от того, что тихий час ещё не кончился.
        """
        if not self._enabled:
            return []
        moment = time.monotonic() if now is None else float(now)
        if self.interval_s and moment - self._last_run < self.interval_s:
            return []
        self._last_run = moment
        try:
            people = self._detect_people(frame)
        except PoseUnavailable as exc:
            if self._warned != str(exc):
                self._warned = str(exc)
                log.warning("Posture watching is unavailable: %s", exc)
            return []
        except Exception as exc:  # noqa: BLE001 - кадр камеры не должен падать
            self._errors += 1
            if self._errors == 1:
                log.warning("Posture detection failed: %s", exc)
            return []
        posture, centre = "", None
        for person in people:
            found = posture_of(person.get("keypoints"))
            if found:
                posture, centre = found, body_centre(person.get("keypoints"))
                break
        self.posture = posture
        event = self.watch.observe(posture, centre, moment)
        if event == "sleep" and not quiet:
            # Днём «лежит неподвижно» — это отдых, а не сон (ТЗ F-307).
            self.watch.reset()
            return []
        if not event:
            return []
        self.events += 1
        if self._hook is not None:
            try:
                self._hook(event)
            except Exception as exc:  # noqa: BLE001 - обработчик не ломает камеру
                log.warning("Posture handler for %s failed (%s)", event, exc)
        return [event]


__all__ = [
    "DEFAULT_INTERVAL_S",
    "DEFAULT_STILL_S",
    "LYING",
    "MIN_CONF",
    "MOVE_THRESHOLD",
    "PoseDetector",
    "PoseUnavailable",
    "PostureService",
    "SITTING",
    "STANDING",
    "StillnessWatch",
    "body_centre",
    "posture_of",
]
