"""Jarvis room client package.

Runs on the living-room PC: microphone capture, Vosk wake word, WebRTC VAD,
WebSocket streaming to the brain server, TTS playback and action execution.

Entry point: ``python -m client.main`` from the repository root.
"""

__all__ = ["audio", "vad", "wakeword", "ws_client", "main"]
