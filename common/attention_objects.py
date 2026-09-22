"""Классы объектов внимания и слова про них (ТЗ F-311).

ТЗ F-311: «Классы YOLO кроме person (кот, собака, посылка у двери) → события;
уведомление „посылка у двери“ при заданной зоне». Значит нужны три вещи,
которые одинаково понимают оба конца провода:

* какие классы считать вниманием (кот, собака, посылка и другие животные и
  вещи, о которых стоит сообщить) — остальное («стул», «чашка») в события не
  идёт, иначе правило срабатывало бы на каждую перестановку мебели;
* как назвать класс словами человека (ru/en/es), потому что детектор
  подписывает находки по-английски, а правило владелец пишет на своём языке;
* как сказать, ГДЕ объект: ТЗ хочет «посылка у двери», то есть имя зоны из
  конфига дома (F-309) с предлогом языка. Имя зоны не переводится — это слова
  самого владельца, как имя человека.
"""
from __future__ import annotations

from typing import Any

from common.object_labels import normalize_label

#: ТЗ F-311: классы объектов внимания и их слова на ru/en/es. Детектор зовёт
#: их по-английски (COCO: cat, dog, bird), а правило владельца — на своём
#: языке; `common.object_labels.normalize_label` сводит написание к группе.
ATTENTION_GROUPS: dict[str, dict[str, str]] = {
    "cat": {"ru": "кошка", "en": "cat", "es": "gato"},
    "dog": {"ru": "собака", "en": "dog", "es": "perro"},
    "package": {"ru": "посылка", "en": "package", "es": "paquete"},
}

#: Синонимы детектора: класс YOLO, который не совпадает с именем группы.
ATTENTION_ALIASES: dict[str, str] = {
    "kitten": "cat",
    "puppy": "dog",
    "parcel": "package",
    "suitcase": "package",
    "handbag": "package",
}

#: «ГДЕ объект» на языке человека. Имя зоны подставляется КАК ЕСТЬ: его
#: написал владелец, и переводить его значило бы переименовывать комнату.
PLACE_TEMPLATES: dict[str, str] = {
    "ru": "{word} у {zone}",
    "en": "{word} at {zone}",
    "es": "{word} en {zone}",
}

#: Языки, которые мы правда умеем; всё остальное — английский.
LANGUAGES = ("ru", "en", "es")


def attention_group(label: Any) -> str:
    """Группа объекта внимания или ``""``, когда это не объект внимания.

    «кот»/«cat»/«gato» — одна группа; «стул»/«chair» — пустая строка, потому
    что ТЗ перечисляет животных и вещи внимания, а не весь список YOLO.
    """
    word = normalize_label(label)
    if not word:
        return ""
    group = ATTENTION_ALIASES.get(word, word)
    return group if group in ATTENTION_GROUPS else ""


def language_of(code: Any) -> str:
    """Код языка из ТЗ (ru/en/es); неизвестный — английский."""
    value = str(code or "").strip().casefold().replace("_", "-")
    base = value.split("-")[0]
    return base if base in LANGUAGES else "en"


def attention_word(group: str, language: str = "en") -> str:
    """Слово объекта внимания на языке человека (пусто для чужой группы)."""
    words = ATTENTION_GROUPS.get(str(group or "").casefold(), {})
    if not words:
        return ""
    return words.get(language_of(language), words.get("en", ""))


def attention_line(group: str, zone: str = "", language: str = "en") -> str:
    """«посылка у двери» — объект и зона словами человека (ТЗ F-311).

    Без зоны строка — просто слово объекта: «посылка». Выдумывать «где-то в
    комнате» нельзя, потому что зону ставит владелец в конфиге дома (F-309).
    """
    word = attention_word(group, language)
    if not word:
        return ""
    place = " ".join(str(zone or "").split())
    if not place:
        return word
    template = PLACE_TEMPLATES.get(language_of(language), PLACE_TEMPLATES["en"])
    return template.format(word=word, zone=place)


__all__ = [
    "ATTENTION_ALIASES",
    "ATTENTION_GROUPS",
    "LANGUAGES",
    "PLACE_TEMPLATES",
    "attention_group",
    "attention_line",
    "attention_word",
    "language_of",
]
