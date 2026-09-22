"""Computer-use под ограничениями: общие правила (ТЗ F-512).

ТЗ F-512: многошаговая задача («найди в Discord сообщение от Макса и ответь
„ок“») — это цикл «скриншот → vision-LLM → действие мыши и клавиатуры» с
ограничениями: allow-list приложений, запрет ввода паролей и платёжных данных,
максимум 15 шагов, стоп по слову «стоп» или ладони, видимый оверлей.

Правила живут в ``common``, потому что их обязаны соблюдать ОБА конца: хаб
решает, что вообще позволено, а клиент-исполнитель перепроверяет каждый шаг
перед тем, как пошевелить мышью. Второй экземпляр правил на клиенте означал бы
второй набор дыр.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: ТЗ F-512: «максимум 15 шагов». Больше не позволит ни конфиг, ни этот файл.
MAX_STEPS = 15

#: Что агент вообще умеет попросить у комнаты.
ACTIONS: tuple[str, ...] = ("click", "type", "key", "scroll", "app", "wait")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


#: ТЗ F-512: «запрет ввода паролей и платёжных данных». Слова трёх языков
#: пользователя (en/ru/es): человек диктует пароль на своём языке, и запрет не
#: должен зависеть от языка интерфейса. Отказ безопаснее пропуска — он виден
#: человеку, а секрет, набранный один раз, уже утёк.
SENSITIVE_RULES: tuple[tuple[str, str], ...] = (
    ("a password",
     r"(?i)(?:password|passwd|passphrase|пароль|пароля|пароли|contrase\w*|clave)"),
    ("a one-time code",
     r"(?i)(?:one[- ]time code|otp\b|2fa code|код подтверждения|код из смс|"
     r"c[oó]digo de verificaci[oó]n)"),
    ("payment card details",
     r"(?i)(?:card number|credit card|debit card|cvv2?|cvc2?\b|номер карты|"
     r"n[uú]mero de tarjeta|tarjeta de cr[eé]dito)"),
    ("bank details",
     r"(?i)(?:iban|bic\b|swift\b|routing number|bank account|реквизиты|"
     r"расч[её]тный сч[её]т|cuenta bancaria)"),
    ("a wallet seed phrase",
     r"(?i)(?:seed phrase|recovery phrase|mnemonic phrase|сид[- ]фраз\w*|"
     r"мнемоническ\w+ фраз\w*|frase semilla)"),
    ("a bank PIN",
     r"(?i)(?:pin[- ]?code|pin[- ]?код|пин[- ]?код|c[oó]digo pin)"),
)

_ACTION = Literal["click", "type", "key", "scroll", "app", "wait"]
_DIRECTION = Literal["", "up", "down"]
_BUTTON = Literal["left", "right", "double"]

#: ТЗ F-512: шаги, «меняющие систему», — те, что закрывают окно с несохранённой
#: работой, запирают ПК или открывают системные диалоги. Их спрашивает F-113,
#: как и любое необратимое действие: сочетания клавиш названы по-человечески,
#: потому что именно эту фразу произнесёт хаб.
SYSTEM_KEY_COMBOS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("alt", "f4"), "close the window with Alt+F4"),
    (("ctrl", "alt", "delete"), "open the security screen"),
    (("win", "l"), "lock the PC"),
    (("win", "r"), "open the Run dialog"),
    (("ctrl", "shift", "esc"), "open Task Manager"),
    (("ctrl", "alt", "esc"), "open Task Manager"),
)

#: Имена модификаторов, которые пишет модель: приводим к одному виду.
_KEY_ALIASES: dict[str, str] = {
    "control": "ctrl", "ctl": "ctrl", "windows": "win", "super": "win",
    "meta": "win", "cmd": "win", "escape": "esc", "del": "delete",
}


def sensitive_reason(text: Any) -> str:
    """Название запрета, который нарушает текст, или ``""``.

    Отказ срабатывает на УПОМИНАНИЕ секрета, а не на попытку отличить
    «настоящий» пароль от слова «пароль»: отказ можно объяснить человеку и
    попросить набрать секрет самому, а утёкший один раз — не вернуть.
    """
    value = str(text or "")
    if not value.strip():
        return ""
    for name, pattern in SENSITIVE_RULES:
        if re.search(pattern, value):
            return name
    return ""


def normalize_app(value: Any) -> str:
    """Имя приложения в одном виде для сравнения: без пути, ``.exe`` и регистра."""
    name = str(value or "").strip().strip('"').casefold()
    if not name:
        return ""
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    if name.endswith(".exe"):
        name = name[:-4]
    return name.strip()


def _key_parts(keys: Any) -> list[str]:
    parts = [str(part).strip().casefold() for part in str(keys or "").split("+")]
    return [_KEY_ALIASES.get(part, part) for part in parts if part]


def changes_system(step_fields: Any) -> str:
    """ТЗ F-512: описание шага, который меняет СИСТЕМУ, или ``""``.

    Читается до выполнения: такое действие хаб спрашивает голосом по F-113,
    потому что закрытое окно с несохранённым текстом и запертый ПК назад не
    отматываются.
    """
    action = str(_field(step_fields, "action") or "").strip().casefold()
    if action != "key":
        return ""
    keys = _key_parts(_field(step_fields, "key"))
    if not keys:
        return ""
    unique = set(keys)
    for combo, description in SYSTEM_KEY_COMBOS:
        if unique == set(combo):
            return description
    # Любое сочетание с клавишей Windows открывает системные вещи; одной
    # клавиши Windows (меню «Пуск») для вопроса мало.
    if "win" in unique and len(unique) > 1:
        return "use the Windows key combination"
    return ""


def _field(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


class ComputerUseStep(_Strict):
    """Один шаг агента: что именно он просит сделать в комнате."""

    action: _ACTION
    #: Приложение шага (``app`` открывает/фокусирует его, для остальных —
    #: какое окно агент считает активным).
    app: str = Field(default="", max_length=120)
    text: str = Field(default="", max_length=2000)
    #: Координаты — ДОЛИ экрана (0…1), а не пиксели: клиент сам знает размер.
    x: float | None = Field(default=None, ge=0.0, le=1.0)
    y: float | None = Field(default=None, ge=0.0, le=1.0)
    key: str = Field(default="", max_length=60)
    button: _BUTTON = "left"
    clicks: int = Field(default=1, ge=1, le=3)
    direction: _DIRECTION = ""
    amount: int = Field(default=3, ge=1, le=50)
    seconds: float = Field(default=0.0, ge=0.0, le=10.0)

    def describe(self) -> str:
        """Человеческая строка шага — для трассы, аудита и оверлея."""
        if self.action == "click":
            where = (f"at {self.x:.2f}, {self.y:.2f}" if self.x is not None
                     and self.y is not None else "at the current position")
            return f"click ({self.button}) {where}"
        if self.action == "type":
            return f"type {len(self.text)} characters"
        if self.action == "key":
            return f"press {self.key or 'a key'}"
        if self.action == "scroll":
            return f"scroll {self.direction or 'down'} by {self.amount}"
        if self.action == "app":
            return f"open or focus {self.app or 'an application'}"
        return f"wait {self.seconds:.1f} s"


class ComputerUsePolicy(_Strict):
    """Что разрешено агенту в этом доме (ТЗ F-512)."""

    enabled: bool = False
    #: ТЗ F-512: «максимум 15 шагов»; конфиг не может поднять планку выше.
    max_steps: int = Field(default=MAX_STEPS, ge=1, le=MAX_STEPS)
    #: ТЗ F-512: «allow-list приложений». Пустой список — агент не трогает
    #: НИ ОДНОГО приложения: безопасное умолчание, а не «разрешено всё».
    allowed_apps: list[str] = Field(default_factory=list)
    #: Набор текста разрешён (пример ТЗ — «ответь ок»), но секреты — никогда.
    allow_typing: bool = True

    @field_validator("allowed_apps")
    @classmethod
    def _clean_apps(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in value:
            name = normalize_app(item)
            if name and name not in cleaned:
                cleaned.append(name)
        return cleaned

    def allows_app(self, name: Any) -> bool:
        """Есть ли приложение в allow-list (по имени без пути, ``.exe`` и регистра)."""
        wanted = normalize_app(name)
        return bool(wanted) and wanted in self.allowed_apps

    def refuse(self, step: ComputerUseStep, *, index: int) -> str:
        """Причина, по которой шаг не выполняется, или ``""``.

        ``index`` — номер шага в НОМЕРАХ С НУЛЯ среди уже выполненных: отказ не
        занимает шаг, потому что в комнате ничего не произошло.
        """
        if not self.enabled:
            return "computer use is switched off for this home"
        if index >= self.max_steps:
            return f"the step limit of {self.max_steps} steps is reached"
        if step.action == "app" and not self.allows_app(step.app):
            return (f"{step.app or 'that application'!r} is not on the list of "
                    "applications computer use may touch")
        if step.action == "type":
            secret = sensitive_reason(step.text)
            if secret:
                return (f"typing {secret} is not allowed: the person must enter it "
                        "themselves")
            if not self.allow_typing:
                return "typing text is switched off for computer use in this home"
        return ""


__all__ = [
    "ACTIONS",
    "MAX_STEPS",
    "SENSITIVE_RULES",
    "SYSTEM_KEY_COMBOS",
    "ComputerUsePolicy",
    "ComputerUseStep",
    "changes_system",
    "normalize_app",
    "sensitive_reason",
]
