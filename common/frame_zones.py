"""Зоны кадра камеры и маска «не анализировать» (ТЗ F-309).

Владелец рисует в кадре «дверь», «стол», «кровать» и маску — область,
которую анализировать нельзя. Полигоны живут в НОРМАЛИЗОВАННЫХ координатах
(0…1), поэтому зона не зависит от разрешения камеры: и клиент, и хаб, и
админка считают одно и то же.

Маска вырезается НА КЛИЕНТЕ до отправки кадра (ТЗ F-309), и клиент помечает
кадр отпечатком своих масок (:func:`masks_rev`): хаб считает тот же отпечаток
по своему конфигу и отказывается анализировать кадр, который пришёл без
маски. Разошедшийся отпечаток означает, что клиент маскировал ДРУГИЕ
области, то есть мог прислать то, что владелец просил не смотреть, — а
«почти совпало» здесь не считается совпадением.

Модуль общий для хаба и комнаты и намеренно ни от чего не зависит, кроме
pydantic: ни базы, ни OpenCV, ни моделей, чтобы клиент мог его импортировать
без «мозга».
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

#: Одна точка кадра в нормализованных координатах.
Point = tuple[float, float]


class FrameZone(BaseModel):
    """Полигон в кадре камеры (ТЗ F-309): «дверь», «стол», «маска».

    Точки — в НОРМАЛИЗОВАННЫХ координатах кадра (0…1), как боксы треков в
    HUD: тогда зона не зависит от разрешения камеры. ``mask: true`` —
    «не анализировать»: такую область клиент закрашивает ДО отправки кадра, а
    хаб не индексирует то, что в неё попало, даже если кадр всё-таки пришёл.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str = Field(min_length=1, max_length=60)
    #: Не меньше трёх точек, каждая — ``[x, y]`` внутри кадра.
    points: list[list[float]] = Field(min_length=3, max_length=64)
    mask: bool = False

    @field_validator("points")
    @classmethod
    def _polygon(cls, value: list[list[float]]) -> list[list[float]]:
        for point in value:
            if len(point) != 2:
                raise ValueError("every zone point is [x, y]")
            if not all(0.0 <= float(axis) <= 1.0 for axis in point):
                raise ValueError("zone points use normalized frame coordinates 0..1")
        return value


def _as_mapping(item: Any) -> dict[str, Any] | None:
    """Одна зона в виде полей ``FrameZone`` — из модели, словаря или dataclass.

    Зону рисует владелец в конфиге хаба, но встречают её разные слои: pydantic
    ``FrameZone`` у клиента и ``HubConfig``, ``Zone`` (dataclass) у зон хаба,
    словарь ``describe()`` в админке. Все они — одно и то же, поэтому зона
    приводится к полям, а не к конкретному классу.
    """
    if isinstance(item, FrameZone):
        return item.model_dump()
    if isinstance(item, Mapping):
        return dict(item)
    name = getattr(item, "name", None)
    points = getattr(item, "points", None)
    if name is None or points is None:
        return None
    try:
        normalized = [[float(point[0]), float(point[1])] for point in points]
    except (TypeError, ValueError, IndexError):
        return None
    return {"name": str(name), "points": normalized,
            "mask": bool(getattr(item, "mask", False))}


def parse_zones(raw: Any) -> list[FrameZone]:
    """Список зон из конфига или патча; испорченная запись пропускается.

    Свой конфиг хаб проверяет строго (``HomeConfig.zones``), а патч с провода
    — это данные, где одна битая зона не должна уносить остальные. Придумать
    зону «на глаз» нельзя, поэтому непрошедшая проверку запись просто
    не применяется.
    """
    if raw is None:
        return []
    if isinstance(raw, FrameZone):
        # Одиночная зона — это одна зона, а не список её полей.
        return [raw]
    if isinstance(raw, dict):
        # Одна зона, записанная словарём, — это тоже зона, а не список ключей.
        raw = [raw]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Iterable):
        return []
    found: list[FrameZone] = []
    for item in raw:
        payload = _as_mapping(item)
        if payload is None:
            continue
        try:
            found.append(FrameZone.model_validate(payload))
        except ValidationError:
            continue
    return found


def _canonical(zones: Any, *, masks_only: bool) -> str:
    """Отпечаток зон: имена, точки (до 0,0001) и признак маски, отсортировано.

    Порядок зон в конфиге на смысл не влияет, поэтому отпечаток сортируется:
    иначе перестановка двух полигонов в yaml выглядела бы как новые маски.
    """
    seen: set[str] = set()
    entries: list[list[Any]] = []
    for zone in parse_zones(zones):
        if masks_only and not zone.mask:
            continue
        name = " ".join(str(zone.name).split())
        key = name.casefold()
        if not name or key in seen:
            continue
        seen.add(key)
        points = [[round(float(x), 4), round(float(y), 4)] for x, y in zone.points]
        entries.append([key, bool(zone.mask), points])
    if not entries:
        return ""
    entries.sort(key=lambda entry: (entry[0], entry[2]))
    raw = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def zones_rev(zones: Any) -> str:
    """Отпечаток ВСЕХ зон: по нему видно, что патч хаба что-то поменял."""
    return _canonical(zones, masks_only=False)


def masks_rev(zones: Any) -> str:
    """Отпечаток ТОЛЬКО масок — его сверяет хаб (ТЗ F-309); ``""`` = масок нет."""
    return _canonical(zones, masks_only=True)


def mask_polygons(zones: Any, width: int, height: int) -> list[list[tuple[int, int]]]:
    """Пиксельные полигоны масок для кадра ``width``×``height``.

    Пустой ответ — это «масок нет» или «размер кадра неизвестен»: без размера
    маскировать нечего, и клиент такую картинку не отправляет, а не делает вид,
    что она уже без маски.
    """
    try:
        frame_w, frame_h = int(width), int(height)
    except (TypeError, ValueError):
        return []
    if frame_w <= 0 or frame_h <= 0:
        return []
    found: list[list[tuple[int, int]]] = []
    for zone in parse_zones(zones):
        if not zone.mask:
            continue
        found.append([
            (int(round(float(x) * frame_w)), int(round(float(y) * frame_h)))
            for x, y in zone.points
        ])
    return found


def polygon_contains(points: Any, x: float, y: float) -> bool:
    """Настоящий ray casting: точка внутри полигона (ТЗ F-309).

    Границы зон не «размазываются» и не угадываются — луч из точки считает
    пересечения с рёбрами. Один алгоритм на хаб и на комнату: иначе зона
    «дверь» у клиента и у хаба оказалась бы разной.
    """
    try:
        polygon = [(float(point[0]), float(point[1])) for point in points or ()]
    except (TypeError, ValueError, IndexError):
        return False
    if len(polygon) < 3:
        return False
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        if (y1 > y) != (y2 > y):
            cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < cross:
                inside = not inside
    return inside


def zone_at(zones: Any, x: float, y: float) -> str:
    """Имя зоны, в которую попадает НОРМАЛИЗОВАННАЯ точка (``""`` — никуда).

    Маска отвечает не здесь: «не анализировать» — это не место, и смешивать
    эти два ответа нельзя (см. :func:`masked_at`).
    """
    for zone in parse_zones(zones):
        if not zone.mask and polygon_contains(zone.points, x, y):
            return zone.name
    return ""


def masked_at(zones: Any, x: float, y: float) -> bool:
    """Лежит ли нормализованная точка в области «не анализировать» (ТЗ F-309)."""
    return any(zone.mask and polygon_contains(zone.points, x, y)
               for zone in parse_zones(zones))


__all__ = [
    "FrameZone",
    "Point",
    "mask_polygons",
    "masked_at",
    "masks_rev",
    "parse_zones",
    "polygon_contains",
    "zone_at",
    "zones_rev",
]
