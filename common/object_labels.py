"""Слова предметов на трёх языках ТЗ (F-305, F-311).

Детектор подписывает находки АНГЛИЙСКИМИ именами классов YOLO, а вопрос и
правило звучат на языке владельца: «где мои ключи?», «посылка у двери». Без
общей таблицы слов память объектов работала бы только по-английски, а
уведомление о посылке не совпадало бы с правилом, написанным по-русски.

Модуль общий для хаба и комнаты (клиент называет класс предмета хабу) и
намеренно ни от чего не зависит. Таблица короткая и честная: незнакомое слово
сравнивается как есть, а не притягивается к «похожему» предмету.
"""
from __future__ import annotations

import re
from typing import Any

#: Общие предметы на трёх языках ТЗ.
LABEL_GROUPS: dict[str, tuple[str, ...]] = {
    "key": ("key", "keys", "ключ", "ключи", "llave", "llaves"),
    "wallet": ("wallet", "wallets", "кошелёк", "кошелек", "кошельки", "cartera",
               "carteras", "billetera"),
    "phone": ("phone", "phones", "cell phone", "телефон", "телефоны", "móvil",
              "movil", "celular"),
    "laptop": ("laptop", "laptops", "ноутбук", "ноутбуки", "portátil", "portatil",
               "ordenador"),
    "cup": ("cup", "cups", "кружка", "кружки", "чашка", "чашки", "taza", "tazas"),
    "bottle": ("bottle", "bottles", "бутылка", "бутылки", "botella", "botellas"),
    "book": ("book", "books", "книга", "книги", "libro", "libros"),
    "backpack": ("backpack", "backpacks", "рюкзак", "рюкзаки", "mochila", "mochilas"),
    "umbrella": ("umbrella", "umbrellas", "зонт", "зонты", "paraguas"),
    "glasses": ("glasses", "очки", "gafas", "lentes"),
    "remote": ("remote", "remote control", "пульт", "пульты", "control remoto", "mando"),
    "charger": ("charger", "chargers", "зарядка", "зарядки", "cargador", "cargadores"),
    "bag": ("bag", "bags", "сумка", "сумки", "bolsa", "bolsas"),
    "cat": ("cat", "cats", "кошка", "кошки", "кот", "коты", "gato", "gatos"),
    "dog": ("dog", "dogs", "собака", "собаки", "пёс", "пес", "perro", "perros"),
    "package": ("package", "packages", "box", "boxes", "посылка", "посылки",
                "коробка", "коробки", "paquete", "paquetes", "caja", "cajas"),
}


def _group_aliases() -> dict[str, str]:
    return {alias: group for group, aliases in LABEL_GROUPS.items() for alias in aliases}


def normalize_label(text: Any) -> str:
    """«ключи» / «keys» / «llaves» — сравнимая форма (без числа и регистра)."""
    word = re.sub(r"[^\w\s]", " ", str(text or "").casefold())
    word = " ".join(word.split())
    aliases = _group_aliases()
    if word in aliases:
        return aliases[word]
    for suffix in ("ами", "ями", "ов", "ев", "ей", "es", "s", "ы", "и", "а", "я"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            word = word[: -len(suffix)].strip()
            break
    return aliases.get(word, word)


__all__ = ["LABEL_GROUPS", "normalize_label"]
