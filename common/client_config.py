"""Standalone room-client settings; no brain or API-provider dependencies."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from common.frame_zones import FrameZone


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class RecordingConfig(_Strict):
    """Local recording retention. Zero disables an age/size limit."""

    enabled: bool = False
    retention_days: int = Field(default=0, ge=0)
    max_gb: float = Field(default=0, ge=0)
    min_free_gb: float = Field(default=5, ge=0)


# ---------------------------------------------------------------------------
# client section
# ---------------------------------------------------------------------------


class WakewordConfig(_Strict):
    """Vosk wake-word settings (``client.wakeword``)."""

    word: str = "rowan ai"
    #: Recognition variants; empty list is normalised to ``[word]``.
    phrases: list[str] = Field(default_factory=list)
    vosk_model: str = "models/vosk-model-small-en-us-0.15"

    @model_validator(mode="after")
    def _default_phrases(self) -> WakewordConfig:
        phrases = [p.strip() for p in self.phrases if isinstance(p, str) and p.strip()]
        if not phrases:
            phrases = [self.word.strip()]
        self.phrases = phrases
        return self


class AudioConfig(_Strict):
    """sounddevice settings (``client.audio``).

    ``input_device``/``output_device`` are either a device index (int), a
    substring of the device name (str) or ``None`` for the system default.
    """

    input_device: int | str | None = None
    output_device: int | str | None = None
    sample_rate: int = Field(default=16000, ge=8000)
    echo_cancellation: bool = False
    #: Barge-in (ТЗ F-102): speaking over Rowan stops the playback in under
    #: 200 ms and the utterance is recorded. Requires ``echo_cancellation``:
    #: without a running WebRTC AEC the client hears its own voice through the
    #: speakers, so the feature switches itself off and the HUD says why.
    barge_in: bool = True
    noise_suppression: bool = False
    noise_suppression_level: int = Field(default=1, ge=0, le=3)


class VADConfig(_Strict):
    """webrtcvad settings (``client.vad``)."""

    aggressiveness: int = Field(default=2, ge=0, le=3)
    silence_ms: int = Field(default=800, ge=0)
    max_utterance_s: float = Field(default=15.0, gt=0.0)
    pre_roll_ms: int = Field(default=300, ge=0)
    #: Minimum voiced audio for a recording to count as an utterance; anything
    #: shorter is a noise blip - discarded without contacting the server.
    min_speech_ms: int = Field(default=250, ge=0)


class VisionProfileConfig(_Strict):
    """One detection profile of the client (ТЗ F-312).

    "Ускорение клиента. Экспорт YOLO11x в TensorRT (FP16) для 3060 Ti; профиль
    для слабых ПК: YOLO11s и треки 10 FPS. Автовыбор профиля по измеренной
    задержке при старте клиента." A profile therefore names the model file, the
    track rate and whether FP16 CUDA inference is used; ``budget_ms`` is the
    per-frame inference time the profile is expected to fit in — a machine that
    cannot hold it steps down to the next profile.
    """

    model: str = Field(min_length=1)
    fps: float = Field(default=10.0, ge=0)
    half: bool = True
    budget_ms: float = Field(default=60.0, gt=0)


def default_vision_profiles() -> dict[str, VisionProfileConfig]:
    """The profiles of ТЗ F-312, strongest first (the order they are tried in)."""
    return {
        # A 3060 Ti (and anything faster): YOLO11x exported to TensorRT FP16.
        "tensorrt": VisionProfileConfig(model="yolo11x.engine", fps=20, half=True, budget_ms=30),
        # A CUDA machine without a TensorRT engine: the same weights in PyTorch.
        "gpu": VisionProfileConfig(model="yolo11x.pt", fps=10, half=True, budget_ms=90),
        # "Профиль для слабых ПК" — exactly what the ТЗ names: YOLO11s, 10 FPS.
        "weak": VisionProfileConfig(model="yolo11s.pt", fps=10, half=False, budget_ms=200),
    }


class CameraConfig(_Strict):
    """Room camera settings (``client.camera``, SPEC v1.4).

    The client runs YOLO on the capture and reports STATE (how many people,
    which objects), plus one JPEG every ``face_check_interval_s`` while
    somebody is visible so the server can recognise faces. Missing camera
    dependencies must never break the voice pipeline.
    """

    enabled: bool = True
    name: str = Field(default='Основная камера', min_length=1, max_length=80)
    #: OpenCV capture device index.
    index: int = Field(default=0, ge=0)
    #: Optional RTSP URL. Kept local, never included in protocol messages.
    stream_url: str | None = Field(default=None, repr=False)
    width: int = Field(default=1920, ge=320, le=7680)
    height: int = Field(default=1080, ge=240, le=4320)
    #: YOLO rate limit; 0 processes fresh frames as fast as capture/GPU allow.
    fps: int = Field(default=5, ge=0)
    #: FP16 CUDA inference on the room GPU.
    half: bool = True
    frame_recording: RecordingConfig = Field(default_factory=RecordingConfig)
    #: Ultralytics model file (downloaded automatically on first run).
    model: str = "yolo11n.pt"
    #: ТЗ F-201: a light guard that watches for the FIRST sign of a person
    #: between the heavy detector's frames. ``yolo11x`` needs ~300 ms per frame,
    #: so on its own it samples the room about three times a second - and a
    #: person who crosses it in half a second can be gone before the next
    #: sample. The light model turns that first sighting into the presence burst
    #: that carries the person's faces to the hub; the heavy model still owns
    #: the tracks, the objects and the identity. ``""`` turns the guard off.
    quick_model: str = Field(default="yolo11n.pt", max_length=120)
    #: How often the guard may look. It runs on the frames the capture thread
    #: already has, so this is an upper bound, not an extra camera read.
    quick_fps: float = Field(default=8.0, ge=0.5, le=240.0)
    #: ТЗ F-312: pick the detection profile by MEASURED latency at startup.
    #: Off by default: a shipped client keeps exactly the profile its config
    #: names, and a machine that wants the automatic choice turns it on.
    auto_profile: bool = False
    #: How many frames one profile is measured on while choosing.
    profile_measure_frames: int = Field(default=3, ge=1, le=30)
    #: The profiles the automatic choice may pick from, strongest first
    #: (``client/vision_profile.py`` walks this order).
    profiles: dict[str, VisionProfileConfig] = Field(default_factory=default_vision_profiles)
    #: One frame is sent to the server this often while a person is visible.
    face_check_interval_s: float = Field(default=0.5, gt=0.0)
    #: ТЗ F-201: report the room's person TRACKS in their own ``tracks``
    #: message (id, box, confidence, since) instead of only a person count.
    #: On by default; a hub that predates the message ignores it and the
    #: ``camera_state`` frame still carries the same tracks.
    tracks_message: bool = True
    #: ТЗ F-309: зоны кадра — «дверь», «стол», «кровать», «маска (не
    #: анализировать)». Владелец рисует их в конфиге дома, и хаб присылает их
    #: комнате сообщением ``config_update``; поле здесь нужно затем, чтобы
    #: комната без хаба (или до первого патча) уже знала свои маски. Маска
    #: закрашивается ДО JPEG, поэтому кадр уходит уже без неё.
    zones: list[FrameZone] = Field(default_factory=list)


class GesturesConfig(_Strict):
    """Жесты руки на клиенте (``client.gestures``, ТЗ F-306).

    MediaPipe Hands работает на CPU комнаты, кадры никуда не уходят. Жесты
    включаются НА ДОМ отдельно: флаг ``homes[].settings.gestures`` приходит
    комнате патчем, а поле здесь нужно комнате без хаба (или до патча).
    """

    enabled: bool = False
    #: Сколько жест держится, прежде чем он сработает («ладонь дольше 1 с»).
    hold_s: float = Field(default=1.0, gt=0.0, le=10.0)
    #: Не чаще одного распознавания за это время (CPU делится с YOLO).
    interval_s: float = Field(default=0.2, ge=0.0, le=5.0)


class PostureConfig(_Strict):
    """Поза и сон на клиенте (``client.posture``, ТЗ F-307).

    YOLO11-pose работает на CPU комнаты с низкой частотой; кадры никуда не
    уходят, наружу идут только события «уснул» и «встал». Включается НА ДОМ
    (``homes[].settings.posture``), поле здесь — для комнаты без хаба.
    """

    enabled: bool = False
    #: ТЗ F-307: один кадр в 5 секунд.
    interval_s: float = Field(default=5.0, ge=0.5, le=120.0)
    #: ТЗ F-307: «лежит неподвижно дольше 10 мин».
    still_s: float = Field(default=600.0, ge=60.0, le=7200.0)


class OverlayConfig(_Strict):
    """Sci-fi HUD overlay on the TV (``client.overlay``).

    A transparent, always-on-top, click-through Tkinter window showing an
    animated orb that reacts to the assistant's state. Entirely optional: a
    missing display or Tk simply disables it, never touching the voice client.
    """

    enabled: bool = True
    position: str = "bottom_right"
    idle_hidden: bool = False
    scale: float = Field(default=1.0, gt=0.0, le=4.0)


class DeviceConfig(BaseModel):
    """One controllable device from ``client.devices``.

    ``name``/``type``/``area``/``description`` are first-class fields; every
    other key of the YAML mapping (``host``, ``dev_id``, ``local_key``,
    ``version``, ``mac``, ``mode``, ...) is collected into :attr:`params` and is
    also kept as an attribute of the model.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    name: str
    #: "magichome" | "tuya" | "switchbot_bot"
    type: str
    area: str | None = None
    description: str | None = None
    #: ТЗ F-606: a device a GUEST may not touch (the owner's own kit). The
    #: client reports it in its ``hello`` device list, and the hub refuses a
    #: guest with a sentence about the owner instead of "device not found".
    restricted: bool = False
    #: Type-specific fields taken from the same YAML mapping.
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _collect_params(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        first_class = {"name", "type", "area", "description", "restricted", "params"}
        explicit = data.get("params")
        params: dict[str, Any] = dict(explicit) if isinstance(explicit, dict) else {}
        for key, value in data.items():
            if key not in first_class:
                params[str(key)] = value
        merged = dict(data)
        merged["params"] = params
        return merged


class ClientOTAConfig(_Strict):
    """Client updates from the hub's release tag (ТЗ 4.9).

    Off by default: a room PC updates itself only when somebody switched the
    feature on for that PC. The hub says *which* tag to run; the client fetches,
    migrates its config, restarts, and rolls back if the new tag cannot stay up
    for the guard window.
    """

    enabled: bool = False
    remote: str = Field(default="origin", min_length=1, max_length=60)
    interval_s: float = Field(default=3600.0, ge=60.0, le=86400.0)
    #: How long a freshly checked-out tag must survive before it counts as good.
    healthy_after_s: float = Field(default=60.0, ge=10.0, le=600.0)
    state_path: str = Field(default="data/ota_state.json", min_length=1, max_length=200)


class OfflineSttConfig(_Strict):
    """Локальный распознаватель комнаты на время, когда хаба нет (ТЗ 4.8).

    ТЗ называет faster-whisper small (или base) на своём GPU/CPU. Модель
    грузится только при первой надобности: клиент без неё продолжает работать
    на локальных фразах (``client/voice_controls.py``, F-117), а не молчит.
    """

    enabled: bool = True
    model: Literal["base", "small"] = "base"
    #: ``auto`` предпочитает CUDA и отступает на CPU, если её нет.
    device: Literal["auto", "cuda", "cpu"] = "auto"
    compute_type: str = Field(default="int8", min_length=1, max_length=30)
    #: Пусто — язык берётся из распознавания (autodetect) faster-whisper.
    language: str = Field(default="", max_length=10)
    #: Дольше этого локальный распознаватель не думает: комната ждёт ответа.
    timeout_s: float = Field(default=20.0, gt=0.0, le=120.0)


class OfflineConfig(_Strict):
    """Деградация комнаты без хаба (ТЗ 4.8)."""

    enabled: bool = True
    #: ТЗ 4.8: «потеря соединения дольше 3 с» — тогда HUD говорит «мозг
    #: оффлайн», а голос объясняет, что Rowan работает локально.
    after_s: float = Field(default=3.0, ge=0.0, le=120.0)
    #: Реконнект с экспоненциальной задержкой: base * factor**n, не больше max.
    backoff_base_s: float = Field(default=1.0, gt=0.0, le=60.0)
    backoff_factor: float = Field(default=2.0, ge=1.0, le=10.0)
    backoff_max_s: float = Field(default=30.0, gt=0.0, le=600.0)
    stt: OfflineSttConfig = Field(default_factory=OfflineSttConfig)
    #: Кэш заранее синтезированных фраз (ТЗ 4.8): без хаба своя TTS на
    #: комнатном ПК не живёт, поэтому нужные строки приносит хаб заранее.
    phrases_cache: bool = True
    phrases_dir: str = Field(default="data/tts_cache", min_length=1, max_length=200)
    #: Сколько событий присутствия копить, пока хаба нет (ТЗ 4.8: после
    #: восстановления связи накопленное досылается).
    presence_buffer: int = Field(default=120, ge=0, le=2000)


class ClientConfig(_Strict):
    """Everything the room PC reads (``client``)."""

    #: WebSocket URL of the brain PC, e.g. ``ws://192.168.1.100:8765/ws``.
    server_url: str
    client_id: str = "livingroom"
    workplace_name: str = Field(default='', max_length=80)
    wakeword: WakewordConfig = Field(default_factory=WakewordConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    #: Room camera: YOLO presence state + face frames for the server (v1.4).
    camera: CameraConfig = Field(default_factory=CameraConfig)
    #: ТЗ F-306: жесты руки (MediaPipe Hands на CPU комнаты).
    gestures: GesturesConfig = Field(default_factory=GesturesConfig)
    #: ТЗ F-307: поза и сон (YOLO11-pose, 1 кадр в 5 с).
    posture: PostureConfig = Field(default_factory=PostureConfig)
    #: Sci-fi HUD overlay on the TV.
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    #: Seconds to keep listening after a reply without the wake word (0 = off).
    followup_window_s: float = Field(default=0.0, ge=0.0, le=30.0)
    #: Wake word for EVERY turn; legacy unattended follow-ups are opt-in.
    attention_mode: Literal["wake_word", "window"] = "wake_word"
    #: Soft repeating blips while the server is still working on a reply
    #: (vision, tool rounds) so silence never looks like a hang.
    thinking_sounds: bool = True
    #: Friendly app name -> executable path / command; OVERRIDES on top of the
    #: client's installed-app index. Empty by default.
    apps: dict[str, str] = Field(default_factory=dict)
    #: Physical devices; empty by default (none are installed yet).
    devices: list[DeviceConfig] = Field(default_factory=list)
    #: Self-update from the hub's release tag (ТЗ 4.9); off by default.
    ota: ClientOTAConfig = Field(default_factory=ClientOTAConfig)
    #: Degradation without the hub (ТЗ 4.8).
    offline: OfflineConfig = Field(default_factory=OfflineConfig)
    #: Hub identity (ТЗ phase 0): which home this client belongs to and how it
    #: authenticates. ``hub_url`` falls back to the legacy ``server_url`` and the
    #: token itself is only ever read from the named environment variable.
    home_id: str = Field(default="", max_length=64)
    hub_url: str | None = Field(default=None, repr=False)
    kind: Literal["room_pc", "phone", "sensor_node"] = "room_pc"
    token_env: str = Field(default="ROWAN_CLIENT_TOKEN", max_length=100)
    caps: list[str] = Field(default_factory=list)

    @property
    def websocket_url(self) -> str:
        """The v2 hub URL, falling back to the legacy ``server_url``."""
        return self.hub_url or self.server_url

    @field_validator("apps", mode="after")
    @classmethod
    def _expand_app_paths(cls, value: dict[str, str]) -> dict[str, str]:
        # Allow %USERNAME%-style environment variables in the example config.
        return {name: os.path.expandvars(path) for name, path in value.items()}

    @model_validator(mode="after")
    def _unique_device_names(self) -> ClientConfig:
        seen: set[str] = set()
        for device in self.devices:
            key = device.name.strip().lower()
            if key in seen:
                raise ValueError(f"duplicate device name: {device.name!r}")
            seen.add(key)
        return self


class ClientSettings(_Strict):
    """The client reads only its own settings, including from legacy combined YAML."""

    client: ClientConfig = Field(default_factory=lambda: ClientConfig(server_url="ws://127.0.0.1:8765/ws"))


#: Sections of a combined config.yaml that only the hub reads. The client
#: drops them before validating, exactly as it always dropped ``server``:
#: one file stays the norm, and the client never parses provider settings.
HUB_SECTIONS = ("server", "models", "homes")


def load_client_config(path: str | os.PathLike[str] = "config.yaml") -> ClientSettings:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read client config {config_path}: {type(exc).__name__}") from None
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f" at line {mark.line + 1}" if mark else ""
        raise ValueError(f"Invalid YAML in {config_path}{location}") from None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("Client config must be a mapping with a client: section")
    # Existing installations use a combined file. Never retain or validate a
    # hub-only section on the client, and never import its provider
    # dependencies.
    raw = {key: value for key, value in raw.items()
           if key not in HUB_SECTIONS and value is not None}
    try:
        return ClientSettings.model_validate(raw)
    except ValidationError as exc:
        # Do not echo arbitrary YAML values (RTSP credentials, device keys, etc.).
        fields = [".".join(map(str, error["loc"])) + ": " + error["type"] for error in exc.errors()]
        raise ValueError("Invalid client config: " + "; ".join(fields)) from None
