"""First-run setup for the standalone Windows client. No cloud keys required."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import stat
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from urllib.request import urlopen

import yaml

from common.client_config import ClientConfig, load_client_config

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"


def validate_server_url(value: str) -> str:
    value = value.strip()
    try:
        parts = urlsplit(value)
        valid = parts.scheme in {"ws", "wss"} and parts.hostname and parts.port != 0
        valid = valid and not (parts.username or parts.password or parts.query or parts.fragment)
        valid = valid and not re.search(r"\s", value)
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Enter a ws:// or wss:// server URL without credentials or query parameters.")
    return value


def save_settings(path: Path, *, server_url: str, input_device: int | None,
                  camera_enabled: bool, camera_index: int, workplace_name: str | None = None,
                  camera_name: str | None = None, camera_model: str | None = None) -> None:
    # Preserve unrelated settings on reconfiguration, including legacy server
    # settings when this helper is used in the full development checkout.
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("Existing config must be a YAML mapping.")
    current = dict(raw.get("client") or {})
    current.setdefault("client_id", "room-" + uuid.uuid4().hex[:12])
    current["server_url"] = validate_server_url(server_url)
    current["audio"] = {**(current.get("audio") or {}), "input_device": input_device}
    current["camera"] = {**(current.get("camera") or {}), "enabled": camera_enabled, "index": camera_index}
    if workplace_name is not None:
        current["workplace_name"] = workplace_name.strip()
    if camera_name is not None:
        current["camera"]["name"] = camera_name.strip()
    if camera_model is not None:
        current["camera"]["model"] = camera_model.strip()
    current["camera"].setdefault("fps", 0)
    current.setdefault("vad", {"aggressiveness": 2, "silence_ms": 1100,
                               "max_utterance_s": 25, "pre_roll_ms": 1500})
    current.setdefault("wakeword", {"word": "rowan ai", "phrases": ["rowan ai", "rowan a i", "roan ai", "roan a i", "rowen ai", "rowen a i", "rowanai"]})
    ClientConfig.model_validate(current)
    raw["client"] = current
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        temporary.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
        if path.exists():
            shutil.copy2(path, path.with_name(path.name + ".bak-" + uuid.uuid4().hex))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _unpack_model(archive: zipfile.ZipFile, target: Path) -> None:
    infos = archive.infolist()
    if sum(info.file_size for info in infos) > 500 * 1024 * 1024:
        raise ValueError("Unexpected wake-word model archive size.")
    for info in infos:
        part = PurePosixPath(info.filename)
        if (part.is_absolute() or ".." in part.parts or "\\" in info.filename
                or ":" in info.filename or not part.parts or part.parts[0] != MODEL_NAME
                or stat.S_ISLNK(info.external_attr >> 16)):
            raise ValueError("Unexpected path in wake-word model archive.")
    archive.extractall(target)


def download_wake_model(root: Path = ROOT) -> None:
    models = root / "models"
    destination = models / MODEL_NAME
    if (destination / "am" / "final.mdl").is_file():
        print("Wake-word model is already installed.")
        return
    if destination.exists():
        raise ValueError("Incomplete wake-word model exists. Rename its folder and run setup again.")
    models.mkdir(parents=True, exist_ok=True)
    print("Downloading the local wake-word model (~40 MB)...")
    with tempfile.TemporaryDirectory(prefix="wake-download-", dir=models) as staging:
        stage = Path(staging)
        archive_path = stage / "model.zip"
        with urlopen(MODEL_URL, timeout=60) as response, archive_path.open("wb") as output:
            received = 0
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > 100 * 1024 * 1024:
                    raise ValueError("Unexpected download size.")
                output.write(chunk)
        with zipfile.ZipFile(archive_path) as archive:
            _unpack_model(archive, stage)
        if not (stage / MODEL_NAME / "am" / "final.mdl").is_file():
            raise ValueError("Wake-word model archive is incomplete.")
        (stage / MODEL_NAME).rename(destination)
    print("Wake-word model installed.")


def _ask_int(prompt: str, default: int) -> int:
    while True:
        answer = input(f"{prompt} [{default}]: ").strip()
        try:
            value = int(answer) if answer else default
            if value >= 0:
                return value
        except ValueError:
            pass
        print("Enter a non-negative number.")


def configure(path: Path) -> None:
    import sounddevice as sd

    existing = load_client_config(path).client if path.exists() else None
    print("\nRowan client setup. OpenAI and Gemini keys stay on the server.")
    print("The connected server can receive microphone/camera/screen data and control this PC.")
    print("Use the address of a server you trust. You can stop the client with Ctrl+C.")
    while True:
        # Do not print an existing URL: old configurations can contain credentials.
        suffix = " (Enter keeps the saved address)" if existing else ""
        entered = input("Server WebSocket URL" + suffix + ": ").strip()
        try:
            url = validate_server_url(entered or (existing.server_url if existing else ""))
            break
        except ValueError as exc:
            print(exc)
    devices = sd.query_devices()
    inputs = {index: item for index, item in enumerate(devices) if item["max_input_channels"] > 0}
    if not inputs:
        raise ValueError("No microphones found. Connect a microphone and run setup again.")
    apis = sd.query_hostapis()
    print("\nMicrophones:")
    for index, item in inputs.items():
        print(f"  {index}: {item['name']} ({apis[item['hostapi']]['name']})")
    while True:
        answer = input("Microphone number (Enter = Windows default): ").strip()
        if not answer:
            microphone = None
            break
        if answer.isdigit() and int(answer) in inputs:
            microphone = int(answer)
            break
        print("Choose a number from the list.")
    camera_default = bool(existing and existing.camera.enabled)
    while True:
        answer = input("Enable camera? " + ("[Y/n]: " if camera_default else "[y/N]: ")).strip().lower()
        if answer in {"", "y", "yes", "n", "no"}:
            enabled = camera_default if not answer else answer in {"y", "yes"}
            break
    camera_index = existing.camera.index if existing else 0
    workplace_default = existing.workplace_name if existing and existing.workplace_name else "My room"
    workplace_name = input(f"Workplace name shown in Telegram [{workplace_default}]: ").strip() or workplace_default
    camera_name = existing.camera.name if existing else "Main camera"
    camera_model = existing.camera.model if existing else "yolo11n.pt"
    if enabled:
        camera_index = _ask_int("USB camera number", camera_index)
        camera_name = input(f"Camera name [{camera_name}]: ").strip() or camera_name
        camera_model = input(f"YOLO model filename [{camera_model}]: ").strip() or camera_model
    save_settings(path, server_url=url, input_device=microphone,
                  camera_enabled=enabled, camera_index=camera_index, workplace_name=workplace_name,
                  camera_name=camera_name, camera_model=camera_model)
    print(f"Settings saved to {path.name}. This local file must not be uploaded to GitHub.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--download-wake-model", action="store_true")
    parser.add_argument("--check", action="store_true", help="Validate local config without starting devices")
    parser.add_argument("--camera-enabled", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.camera_enabled:
            print(json.dumps(load_client_config(args.config).client.camera.enabled))
        elif args.download_wake_model:
            download_wake_model()
        elif args.check:
            load_client_config(args.config)
            print("Client configuration is valid.")
        else:
            configure(args.config)
        return 0
    except (ValueError, OSError, yaml.YAMLError) as exc:
        # SDK/YAML exceptions can echo configuration contents. Keep output local
        # and concise; never print a YAML parser's source snippet.
        if isinstance(exc, yaml.YAMLError):
            print("Invalid YAML in the existing config.")
        else:
            print(str(exc))
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nSetup cancelled.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
