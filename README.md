# Rowan client

Windows microphone, camera and desktop client for a separately hosted Rowan
brain server. The client detects the wake word locally, streams requests to
the server, plays responses, shows the personal chat overlay and executes
desktop/browser actions. It does not run the language model or contain the
brain server.

## Install and connect

Already have an Anaconda environment? Use the [existing-environment setup](#use-your-existing-anaconda-environment) below.

1. Install **Python 3.11 or 3.12, 64-bit** from [python.org](https://www.python.org/downloads/windows/), including the Python launcher.
2. Clone with Git for Windows: `git clone https://github.com/rudnytskyi1/rowanai.git`.
   A ZIP download also works, but Git makes later updates easier.
3. Run `setup-client.bat`. Setup creates `.venv`, installs client dependencies,
   asks for the server WebSocket address, microphone, workplace/camera names and YOLO model, and downloads the
   small local Vosk wake-word model. Camera support is optional. No cloud API
   key is requested.
4. Run `start-client.bat`. Say **Rowan AI**, then your request. Press **Ctrl+C** in
   the console to stop the client.

Ask the server operator for a reachable `ws://.../ws` or `wss://.../ws` URL.
The example `ws://127.0.0.1:8765/ws` works only when the brain runs on the same
PC. Use a private network/VPN or your operator's protected endpoint. This
client package does not create an account or grant access to a server.

To change devices/address, run setup again. Advanced local settings are in
`config.yaml`; `config.example.yaml` documents the client fields. Every fresh
setup generates a unique `client_id`; retain it when reconnecting the same PC.

### Use your existing Anaconda environment

Open **Anaconda Prompt** and run these commands. Replace `YOUR_ENV` with your
existing environment name and the directory with your Rowan client checkout:

```bat
conda activate YOUR_ENV
python --version
cd /d C:\Path\To\rowanai
setup-client-conda.bat
start-client-conda.bat
```

The environment must contain **Python 3.11 or 3.12, 64-bit**. Setup uses that
environment for dependency installation, the device/server setup wizard,
optional camera packages and the wake-word model download. It does not create
another environment or a `.venv` directory. Keep Anaconda Prompt open until
setup finishes; `start-client-conda.bat` uses the currently activated environment.

If you prefer an explicit interpreter path, activation is optional:

```bat
setup-client-conda.bat -Python "C:\Path\To\env\python.exe"
start-client-conda.bat -Python "C:\Path\To\env\python.exe"
```

The chosen interpreter is saved locally in `.rowan-python`, which is ignored
by Git and excluded from releases. Later you can double-click ordinary
`start-client.bat`, `update-client.bat` or `setup-client.bat`; they keep using
the saved environment, even if an older `.venv` exists. If the environment is
moved or removed, the scripts stop and ask you to select it again. To choose
another environment, activate it and rerun `setup-client-conda.bat`; the old
selection is backed up locally. The Conda launchers require activation or
`-Python` and never pick an arbitrary Python from `PATH`.

For a fresh install without a saved environment, ordinary `setup-client.bat`
keeps its default behavior of creating a local `.venv`.

## Updating

Stop Rowan, run `update-client.bat`, then `start-client.bat`. The updater pulls
the next release and installs its dependencies. It preserves `config.yaml`,
models and local data, and refuses to overwrite edited source files. With a ZIP
installation, download the new ZIP into another directory and copy your local
`config.yaml` and `models/` there, then run setup. For an Anaconda installation,
activate your existing environment and run `setup-client-conda.bat` in the new
directory so its launchers remember the same interpreter.

The runtime files here come directly from the same `client/` and `common/`
source used by the developer's room PC. There is one maintained implementation;
publication checks the exact file bytes and records their SHA-256 hashes in
`release-manifest.json`. Behavior depends on your local hardware/settings and
the server you connect to; no cloud provider keys are stored in this repository.

## Several rooms and Telegram

Each PC needs its own `client_id`. In local `config.yaml` configure:

```yaml
client:
  client_id: second-room
  workplace_name: Second room
  camera:
    enabled: true
    name: Entrance
    index: 0
    model: yolo11n.pt
    fps: 0
```

This is a fragment to merge into your existing client settings, keeping the
server URL. `fps: 0` processes fresh frames as fast as the camera/GPU allows.
Select a larger compatible YOLO model if desired; it needs more GPU time.
One active camera per client is currently supported. Several connected clients
appear as separate workplaces/cameras in the server owner's Telegram `/tools`
panel. The owner can choose the camera separately in the group and in DM, take
photos and enable presence alerts with photos or short silent clips. If several
clients are connected without a selection, Rowan asks which workplace to use.
Offline selections never silently switch to another room.

Telegram polling, permissions, memory, face/voice profiles, SAM3, image generation
and notification rules all run on the brain server. Install no Telegram bot
token or OpenAI/Gemini API key on this client. Contact the server owner to obtain
Telegram access; installing the client does not grant owner permissions.

If Python is installed without the launcher, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install-client.ps1 -Python 'C:\Path\To\python.exe'
```

## What the connection allows

Connect only to a server you trust: it can request camera frames and desktop
screenshots and execute actions on this PC, including commands, browser
interaction and device control. Camera presence frames are sent while the
camera is enabled and a person is visible; this is not limited to voice turns.
Microphone requests start after the local wake word. Server settings determine
voice/face recognition, personal history, cloud processing, and recording
retention. Ask the server operator which data they keep.

OpenAI/Gemini credentials belong on the **server**. The client needs neither.
Keep your `config.yaml`, camera/device credentials, `data/`, recordings, browser
profiles, downloaded models and environment files private. They are ignored
by Git. Local camera archiving is off by default in this distribution.

## Optional features

The commands below use the default `.venv`. If you chose an Anaconda
environment, first run `conda activate YOUR_ENV` in Anaconda Prompt and use
`python` in place of `.venv\Scripts\python` in each command.

- The installer includes the chat overlay and browser control. Browser control
  uses your ordinary open Chrome/Edge window through Windows accessibility,
  with your current tabs and sign-in. It does not create another browser profile.
- The camera package installs YOLO/OpenCV only when camera support is enabled.
  CPU works; for NVIDIA acceleration install the appropriate CUDA PyTorch build
  into the selected environment using the [official selector](https://pytorch.org/get-started/locally/)
  **before** installing camera requirements. The installer does not assume a
  particular graphics card. First camera start downloads the YOLO weights.
- Optional local audio processing: `.venv\Scripts\python -m pip install -r client/requirements-audio.txt`.
  Enable the corresponding `audio` settings afterward.
- Validate settings without opening devices:
  `.venv\Scripts\python -m client.setup --check`.
- Download/retry the wake-word model:
  `.venv\Scripts\python -m client.setup --download-wake-model`.
- Logs are local in `data/client.log`; review/redact them before sharing, since
  they may include transcripts and connection details.

This repository contains the client runtime and shared wire protocol only.
The server, API keys, recordings, personal profiles and the original project's
Git history are not part of this distribution.
