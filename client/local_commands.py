"""Локальные команды комнаты: работают и без хаба (ТЗ F-117, раздел 4.8).

The room should not go deaf-and-dumb when the brain is away: "громче",
"замолчи", "повтори", "включи свет", "сцена кино", "таймер на 20 минут" are
all things the room PC can do by itself. This module holds the vocabulary (one
place, three languages) and the runner that carries a recognized phrase out
through the client's OWN hands - the real device/PC dispatcher, the real
silence path, the real replay of the last reply.

Nothing here pretends. A command whose hand is missing (no dispatcher, no local
voice for a scene's spoken step, no hub for a scene that was never cached)
comes back as ``ok=False`` with a sentence that says so, which is what F-117 +
4.8 promise the room: fewer words, never a lie.
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from common.voice_commands import WAKE_ADDRESS_PATTERN, normalize

log = logging.getLogger(__name__)

#: What a locally recognized phrase can mean.
KINDS = (
    "stop",
    "repeat",
    "volume_set",
    "volume_up",
    "volume_down",
    "mute",
    "unmute",
    "media_next",
    "media_prev",
    "media_play_pause",
    "device",
    "scene",
    "timer",
)

#: The kinds the PC controller carries out (``client.actions.pc``).
PC_KINDS = frozenset({"volume_set", "volume_up", "volume_down", "mute", "unmute",
                      "media_next", "media_prev", "media_play_pause"})

STOP_PHRASES = frozenset({
    "stop", "stop it", "stop talking", "shut up", "be quiet", "quiet",
    "стоп", "остановись", "замолчи", "помолчи", "тихо", "хватит",
    "para", "silencio", "cállate",
})
REPEAT_PHRASES = frozenset({
    "repeat", "repeat that", "say that again", "say it again", "what did you say",
    "повтори", "повтори пожалуйста", "повтори это", "что ты сказал",
    "repite", "otra vez", "qué dijiste",
})
VOLUME_UP_PHRASES = frozenset({
    "louder", "volume up", "turn it up", "make it louder",
    "громче", "сделай громче", "прибавь звук", "прибавь громкость",
    "más alto", "sube el volumen",
})
VOLUME_DOWN_PHRASES = frozenset({
    "quieter", "volume down", "turn it down", "make it quieter",
    "тише", "сделай тише", "убавь звук", "убавь громкость",
    "más bajo", "baja el volumen",
})
MUTE_PHRASES = frozenset({"mute", "выключи звук", "без звука", "silenciar", "mutear"})
UNMUTE_PHRASES = frozenset({"unmute", "включи звук", "со звуком", "activar sonido"})
MEDIA_NEXT_PHRASES = frozenset({
    "next track", "next song", "skip track", "следующий трек", "следующая песня",
    "следующая композиция", "siguiente canción",
})
MEDIA_PREV_PHRASES = frozenset({
    "previous track", "previous song", "предыдущий трек", "предыдущая песня",
    "canción anterior",
})
PLAY_PAUSE_PHRASES = frozenset({
    "pause the music", "play the music", "resume the music", "поставь на паузу",
    "включи музыку", "продолжи музыку", "pausa la música",
})

_WAKE_PREFIX = re.compile(r"^\s*(?:(?:hey|okay|ok)\s+)?(?:" + WAKE_ADDRESS_PATTERN + r")\b[\s,:]*",
                          re.IGNORECASE)
_POLITE_PREFIX = re.compile(r"^\s*(?:please|пожалуйста|por favor)\s+")
_ARTICLE_PREFIX = re.compile(r"^\s*(?:the|my|a|мой|мою|наш|нашу)\s+")

_VOLUME_SET = re.compile(
    r"^(?:set (?:the )?volume to|volume|громкость|установи громкость(?: на)?|pon el volumen a)\s+"
    r"(\d{1,3})(?:\s*%|\s*percent|\s*процентов)?$")
_TIMER = re.compile(
    r"^(?:set |start |поставь |заведи |установи )?(?:a )?timer(?: for)?\s+(\d{1,3})\s*"
    r"(second|seconds|minute|minutes|hour|hours)$"
    r"|^(?:поставь |заведи |установи )?таймер(?: на)?\s+(\d{1,3})\s*"
    r"(секунду|секунды|секунд|минуту|минуты|минут|час|часа|часов)$"
    r"|^temporizador(?: de| para)?\s+(\d{1,3})\s*"
    r"(segundo|segundos|minuto|minutos|hora|horas)$")
_DEVICE_ON = re.compile(r"^(?:turn on|switch on|включи|зажги|enciende)\s+(.+)$")
_DEVICE_OFF = re.compile(r"^(?:turn off|switch off|выключи|погаси|apaga)\s+(.+)$")
_SCENE = re.compile(
    r"^(?:run |start |запусти |включи |активируй )?(?:the )?"
    r"(?:scene|сцену|сцена|escena)\s+(.+)$")

_UNIT_SECONDS = {
    "second": 1, "seconds": 1, "секунду": 1, "секунды": 1, "секунд": 1, "segundo": 1, "segundos": 1,
    "minute": 60, "minutes": 60, "минуту": 60, "минуты": 60, "минут": 60, "minuto": 60, "minutos": 60,
    "hour": 3600, "hours": 3600, "час": 3600, "часа": 3600, "часов": 3600, "hora": 3600, "horas": 3600,
}


@dataclass(frozen=True)
class LocalCommand:
    """One phrased command the room PC can carry out by itself."""

    kind: str
    value: int = 0
    device: str = ""
    state: str = ""
    scene: str = ""
    seconds: int = 0
    heard: str = ""

    def action(self) -> dict[str, Any] | None:
        """The ``pc_control`` action of a PC kind (``None`` for the rest)."""
        if self.kind not in PC_KINDS:
            return None
        args: dict[str, Any] = {"command": self.kind}
        if self.kind == "volume_set":
            args["value"] = int(self.value)
        return {"id": f"local-{self.kind}", "tool": "pc_control", "args": args}


def strip_address(text: Any, wake_words: Iterable[str] = ()) -> str:
    """The bare words: the wake name, "please" and punctuation out."""
    cleaned = str(text or "").strip()
    for word in sorted((str(w) for w in wake_words if w), key=len, reverse=True):
        cleaned = re.sub(r"^" + re.escape(word) + r"\b[\s,:]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = _WAKE_PREFIX.sub("", cleaned)
    cleaned = cleaned.strip().rstrip(".!?")
    cleaned = _POLITE_PREFIX.sub("", cleaned).strip()
    return cleaned


def _match_known(asked: str, known: Iterable[str]) -> str:
    """The known name the room just said, or ``""`` (word overlap, not fuzz)."""
    words = set(normalize(_ARTICLE_PREFIX.sub("", asked)).split())
    if not words:
        return ""
    best, best_len = "", 0
    for name in known:
        key = normalize(name)
        name_words = set(key.split())
        if not name_words:
            continue
        if (name_words <= words or words <= name_words) and len(name_words) > best_len:
            best, best_len = str(name), len(name_words)
    return best


def parse_local_command(
    text: Any,
    *,
    wake_words: Iterable[str] = (),
    devices: Iterable[str] = (),
    scenes: Iterable[str] = (),
) -> LocalCommand | None:
    """The local command in ``text``, or ``None`` when the hub should handle it."""
    heard = strip_address(text, wake_words)
    phrase = normalize(_ARTICLE_PREFIX.sub("", heard))
    if not phrase:
        return None
    if phrase in STOP_PHRASES:
        return LocalCommand(kind="stop", heard=heard)
    if phrase in REPEAT_PHRASES:
        return LocalCommand(kind="repeat", heard=heard)
    if phrase in VOLUME_UP_PHRASES:
        return LocalCommand(kind="volume_up", heard=heard)
    if phrase in VOLUME_DOWN_PHRASES:
        return LocalCommand(kind="volume_down", heard=heard)
    if phrase in MUTE_PHRASES:
        return LocalCommand(kind="mute", heard=heard)
    if phrase in UNMUTE_PHRASES:
        return LocalCommand(kind="unmute", heard=heard)
    if phrase in MEDIA_NEXT_PHRASES:
        return LocalCommand(kind="media_next", heard=heard)
    if phrase in MEDIA_PREV_PHRASES:
        return LocalCommand(kind="media_prev", heard=heard)
    if phrase in PLAY_PAUSE_PHRASES:
        return LocalCommand(kind="media_play_pause", heard=heard)
    volume = _VOLUME_SET.fullmatch(phrase)
    if volume is not None and 0 <= int(volume.group(1)) <= 100:
        return LocalCommand(kind="volume_set", value=int(volume.group(1)), heard=heard)
    timer = _TIMER.fullmatch(phrase)
    if timer is not None:
        count, unit = next((pair for pair in ((timer.group(1), timer.group(2)),
                                             (timer.group(3), timer.group(4)),
                                             (timer.group(5), timer.group(6)))
                            if pair[0]), ("", ""))
        if count and unit:
            return LocalCommand(kind="timer", seconds=int(count) * _UNIT_SECONDS[unit], heard=heard)
    # The scene is checked before the device verbs: "включи сцену кино" starts
    # with a device verb but is not a device.
    scene_match = _SCENE.fullmatch(phrase)
    if scene_match is not None:
        scene = _match_known(scene_match.group(1), scenes)
        return LocalCommand(kind="scene", scene=scene, heard=heard) if scene else None
    for pattern, state in ((_DEVICE_ON, "on"), (_DEVICE_OFF, "off")):
        match = pattern.fullmatch(phrase)
        if match is not None:
            device = _match_known(match.group(1), devices)
            if device:
                return LocalCommand(kind="device", device=device, state=state, heard=heard)
            return None
    return None


def scene_action_items(steps: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """A cached scene as local action items (ТЗ F-506 steps, 4.8 offline).

    ``{"kind": "action", "tool": ..., "args": ...}``, ``{"kind": "delay", ...}``,
    ``{"kind": "say", "text": ...}`` or ``{"kind": "unsupported", ...}`` for a
    step the client cannot carry out by itself - the runner refuses that one
    with the reason instead of skipping it silently.
    """
    items: list[dict[str, Any]] = []
    for raw in steps:
        step = dict(raw or {})
        kind = str(step.get("kind") or "")
        if kind == "device":
            capability = str(step.get("capability") or "")
            value = step.get("value")
            args: dict[str, Any] = {"device": str(step.get("device") or "")}
            if capability == "on_off":
                args["state"] = "on" if value else "off"
            elif capability in {"brightness", "color_rgb"}:
                args["brightness" if capability == "brightness" else "color"] = value
                args.setdefault("state", "on")
            elif capability == "media_play":
                items.append({"kind": "action", "tool": "pc_control",
                              "args": {"command": "media_play_pause"}})
                continue
            else:
                items.append({"kind": "unsupported", "step": step,
                              "reason": f"this client cannot set {capability or 'that'}"})
                continue
            items.append({"kind": "action", "tool": "set_light", "args": args})
        elif kind == "pc":
            items.append({"kind": "action", "tool": str(step.get("tool") or ""),
                          "args": dict(step.get("args") or {})})
        elif kind == "say":
            items.append({"kind": "say", "text": str(step.get("text") or "")})
        elif kind == "delay":
            items.append({"kind": "delay", "seconds": float(step.get("seconds") or 0.0)})
        else:
            items.append({"kind": "unsupported", "step": step, "reason": "unknown step"})
    return items


@dataclass
class LocalOutcome:
    """What came of one local command: the words to say, and whether it worked."""

    ok: bool
    spoken: str
    kind: str
    steps: list[dict[str, Any]] = field(default_factory=list)


_PC_SPEECH = {
    "volume_up": "Louder.",
    "volume_down": "Quieter.",
    "mute": "Muted.",
    "unmute": "Sound on.",
    "media_next": "Next track.",
    "media_prev": "Previous track.",
    "media_play_pause": "Done.",
}


class LocalRunner:
    """Carry a local command out with the client's own hands, hub or not.

    Every hand is injected and optional: whichever one is missing turns its own
    commands into an honest "I cannot do that without the hub" instead of a
    silent success. ``dispatch`` is the real ``client.actions.dispatcher``
    entry point, ``stop`` is the client's own silence path, ``repeat`` replays
    the last reply from the local cache, ``speak`` is the client's local voice
    (its absence is exactly why a scene's spoken step is refused), and
    ``schedule_timer`` starts the local countdown.
    """

    def __init__(
        self,
        *,
        dispatch: Callable[[dict[str, Any]], Awaitable[Any]] | None = None,
        stop: Callable[[], Any] | None = None,
        repeat: Callable[[], bool] | None = None,
        speak: Callable[[str], Awaitable[Any] | Any] | None = None,
        schedule_timer: Callable[[int], str] | None = None,
        scenes: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self.dispatch = dispatch
        self.stop = stop
        self.repeat = repeat
        self.speak = speak
        self.schedule_timer = schedule_timer
        self.scenes = {str(name): list(steps) for name, steps in dict(scenes or {}).items()}
        self.sleep = sleep

    async def run(self, command: LocalCommand) -> LocalOutcome:
        """Carry out ``command``; never raises, always says what happened."""
        if command.kind in PC_KINDS:
            return await self._pc(command)
        if command.kind == "stop":
            return self._stop(command)
        if command.kind == "repeat":
            return self._repeat(command)
        if command.kind == "device":
            return await self._device(command)
        if command.kind == "scene":
            return await self._scene(command)
        if command.kind == "timer":
            return self._timer(command)
        return LocalOutcome(False, "I do not know how to do that locally.", command.kind)

    async def _pc(self, command: LocalCommand) -> LocalOutcome:
        action = command.action()
        if action is None:
            return LocalOutcome(False, "I cannot do that without the hub.", command.kind)
        if self.dispatch is None:
            return LocalOutcome(False, "I cannot reach the PC's own controls.", command.kind)
        ok, error, _output = await self._call_dispatch(action)
        spoken = f"Volume {command.value} percent." if command.kind == "volume_set" else _PC_SPEECH.get(
            command.kind, "Done.")
        return LocalOutcome(ok, spoken if ok else (error or "That did not work."),
                            command.kind,
                            [{"tool": action["tool"], "args": action["args"], "ok": ok}])

    async def _device(self, command: LocalCommand) -> LocalOutcome:
        if self.dispatch is None:
            return LocalOutcome(False, "I cannot reach the devices from here.", command.kind)
        args = {"device": command.device, "state": command.state}
        action = {"id": f"local-{command.state}", "tool": "set_light", "args": args}
        ok, error, output = await self._call_dispatch(action)
        detail = str(output or "").strip()
        spoken = detail or (f"{command.device}: {command.state}" if ok
                            else (error or f"I could not switch {command.device}."))
        return LocalOutcome(ok, spoken, command.kind,
                            [{"tool": "set_light", "args": args, "ok": ok}])

    def _stop(self, command: LocalCommand) -> LocalOutcome:
        if self.stop is None:
            return LocalOutcome(False, "I cannot stop anything from here.", command.kind)
        self.stop()
        return LocalOutcome(True, "Stopped.", command.kind)

    def _repeat(self, command: LocalCommand) -> LocalOutcome:
        if self.repeat is None or not self.repeat():
            return LocalOutcome(False, "I have nothing to repeat yet.", command.kind)
        return LocalOutcome(True, "", command.kind)

    def _timer(self, command: LocalCommand) -> LocalOutcome:
        if self.schedule_timer is None:
            return LocalOutcome(False, "I cannot keep a timer without the hub.", command.kind)
        return LocalOutcome(True, str(self.schedule_timer(command.seconds)), command.kind)

    async def _scene(self, command: LocalCommand) -> LocalOutcome:
        steps = self.scenes.get(command.scene, [])
        if not steps:
            # The scene itself was never cached on this client: say so instead
            # of guessing at the steps (ТЗ 4.8 keeps the cache, not the brain).
            return LocalOutcome(False, "I do not know that scene without the hub.", command.kind)
        rows: list[dict[str, Any]] = []
        for item in scene_action_items(steps):
            kind = str(item.get("kind") or "")
            if kind == "action":
                if self.dispatch is None:
                    ok, detail = False, "the device dispatcher is not wired"
                else:
                    ok, error, output = await self._call_dispatch(
                        {"id": "local-scene", "tool": item["tool"], "args": item["args"]})
                    detail = str(output or error or "")
                rows.append({"tool": str(item["tool"]), "args": item["args"],
                             "ok": ok, "detail": detail})
            elif kind == "delay":
                await self.sleep(float(item.get("seconds") or 0.0))
                rows.append({"tool": "delay", "args": {}, "ok": True,
                             "detail": f"waited {float(item.get('seconds') or 0.0):g}s"})
            elif kind == "say":
                if self.speak is None:
                    rows.append({"tool": "say", "args": {}, "ok": False,
                                 "detail": "no local voice on this client"})
                    continue
                spoken = self.speak(str(item.get("text") or ""))
                if asyncio.iscoroutine(spoken):
                    await spoken
                rows.append({"tool": "say", "args": {}, "ok": True, "detail": "said"})
            else:
                rows.append({"tool": "scene", "args": {}, "ok": False,
                             "detail": str(item.get("reason") or "that step is not local")})
        if not rows:
            return LocalOutcome(False, f"The scene {command.scene} has no steps yet.",
                                command.kind, rows)
        failed = [row for row in rows if not row["ok"]]
        if not failed:
            return LocalOutcome(True, f"Scene {command.scene} done ({len(rows)} step(s)).",
                                command.kind, rows)
        done = len(rows) - len(failed)
        reasons = " ".join(str(row["detail"]) for row in failed[:2])
        return LocalOutcome(
            False, f"Scene {command.scene}: {done} of {len(rows)} steps done. {reasons}",
            command.kind, rows)

    async def _call_dispatch(self, action: dict[str, Any]) -> tuple[bool, str, str]:
        """The dispatcher's ``(ok, error, output)``; a crash is a failure."""
        dispatch = self.dispatch
        if dispatch is None:
            return False, "the dispatcher is not wired", ""
        try:
            result = await dispatch(action)
        except Exception as exc:  # noqa: BLE001 - one broken hand must not crash the client
            log.info("Local command %s failed: %s", action.get("tool"), exc)
            return False, str(exc), ""
        if isinstance(result, tuple) and len(result) == 3:
            ok, error, output = result
            return bool(ok), str(error or ""), str(output or "")
        return bool(result), "", ""


__all__ = [
    "KINDS",
    "MEDIA_NEXT_PHRASES",
    "MEDIA_PREV_PHRASES",
    "MUTE_PHRASES",
    "PC_KINDS",
    "PLAY_PAUSE_PHRASES",
    "REPEAT_PHRASES",
    "STOP_PHRASES",
    "UNMUTE_PHRASES",
    "VOLUME_DOWN_PHRASES",
    "VOLUME_UP_PHRASES",
    "LocalCommand",
    "LocalOutcome",
    "LocalRunner",
    "parse_local_command",
    "scene_action_items",
    "strip_address",
]
