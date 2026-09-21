"""Standalone room-client settings; no brain or API-provider dependencies."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


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
    phrases: List[str] = Field(default_factory=list)
    vosk_model: str = "models/vosk-model-small-en-us-0.15"

    @model_validator(mode="after")
    def _default_phrases(self) -> "WakewordConfig":
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

    input_device: Union[int, str, None] = None
    output_device: Union[int, str, None] = None
    sample_rate: int = Field(default=16000, ge=8000)
    echo_cancellation: bool = False
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
    #: One frame is sent to the server this often while a person is visible.
    face_check_interval_s: float = Field(default=0.5, gt=0.0)


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
    area: Optional[str] = None
    description: Optional[str] = None
    #: Type-specific fields taken from the same YAML mapping.
    params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _collect_params(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        first_class = {"name", "type", "area", "description", "params"}
        explicit = data.get("params")
        params: Dict[str, Any] = dict(explicit) if isinstance(explicit, dict) else {}
        for key, value in data.items():
            if key not in first_class:
                params[str(key)] = value
        merged = dict(data)
        merged["params"] = params
        return merged


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
    apps: Dict[str, str] = Field(default_factory=dict)
    #: Physical devices; empty by default (none are installed yet).
    devices: List[DeviceConfig] = Field(default_factory=list)

    @field_validator("apps", mode="after")
    @classmethod
    def _expand_app_paths(cls, value: Dict[str, str]) -> Dict[str, str]:
        # Allow %USERNAME%-style environment variables in the example config.
        return {name: os.path.expandvars(path) for name, path in value.items()}

    @model_validator(mode="after")
    def _unique_device_names(self) -> "ClientConfig":
        seen: set = set()
        for device in self.devices:
            key = device.name.strip().lower()
            if key in seen:
                raise ValueError(f"duplicate device name: {device.name!r}")
            seen.add(key)
        return self


class ClientSettings(_Strict):
    """The client reads only its own settings, including from legacy combined YAML."""

    client: ClientConfig = Field(default_factory=lambda: ClientConfig(server_url="ws://127.0.0.1:8765/ws"))


def load_client_config(path: str | os.PathLike = "config.yaml") -> ClientSettings:
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
    # Existing installations use a combined file. Never retain or validate the
    # server section on the client, and never import its provider dependencies.
    raw = {key: value for key, value in raw.items() if key != "server" and value is not None}
    try:
        return ClientSettings.model_validate(raw)
    except ValidationError as exc:
        # Do not echo arbitrary YAML values (RTSP credentials, device keys, etc.).
        fields = [".".join(map(str, error["loc"])) + ": " + error["type"] for error in exc.errors()]
        raise ValueError("Invalid client config: " + "; ".join(fields)) from None
