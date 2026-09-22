"""Audio capture and playback for the Jarvis room client.

Capture: ``sounddevice.RawInputStream`` delivers mono int16 blocks of
``frame_ms`` (30 ms => 480 samples @ 16 kHz) from a PortAudio thread into a
thread-safe ``queue.Queue``; the asyncio side pops frames with
:meth:`AudioInput.read_frame`.

Playback: raw PCM s16le is written to a ``sounddevice.RawOutputStream`` by a
single writer task, so the event loop never blocks on the audio device.
Beeps are synthesised in code (no wav assets).
"""

from __future__ import annotations

import asyncio
import collections
import logging
import queue
import threading
import time
from typing import Union

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

#: Bytes per sample for the PCM format used everywhere in this project.
SAMPLE_WIDTH = 2
#: Default capture frame length in milliseconds (webrtcvad accepts 10/20/30).
FRAME_MS = 30
#: Ack beep parameters (SPEC §7 step 2).
BEEP_FREQ_HZ = 880.0
BEEP_MS = 120
#: Error beep (SPEC §4, server -> client ``error``).
ERROR_BEEP_FREQ_HZ = 330.0
ERROR_BEEP_MS = 220

DeviceSpec = Union[int, str, None]


def frame_samples(sample_rate: int, frame_ms: int = FRAME_MS) -> int:
    """Number of samples in one frame."""
    return int(sample_rate * frame_ms // 1000)


def frame_bytes(sample_rate: int, frame_ms: int = FRAME_MS) -> int:
    """Number of bytes in one mono int16 frame."""
    return frame_samples(sample_rate, frame_ms) * SAMPLE_WIDTH


def make_tone(freq: float, ms: int, sample_rate: int, volume: float = 0.35) -> bytes:
    """Generate a mono int16 sine tone with short fades (click-free)."""
    n = max(1, int(sample_rate * ms / 1000))
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    wave = np.sin(2.0 * np.pi * float(freq) * t)
    fade = min(int(sample_rate * 0.006), n // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float64)
        wave[:fade] *= ramp
        wave[n - fade:] *= ramp[::-1]
    volume = max(0.0, min(1.0, float(volume)))
    return (wave * volume * 32767.0).astype(np.int16).tobytes()


def resample_pcm(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resampler (fallback when the device refuses a rate)."""
    if src_rate == dst_rate or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size == 0:
        return b""
    n_out = max(1, int(round(samples.size * float(dst_rate) / float(src_rate))))
    x_old = np.arange(samples.size, dtype=np.float64)
    x_new = np.linspace(0.0, float(samples.size - 1), n_out, dtype=np.float64)
    out = np.interp(x_new, x_old, samples.astype(np.float64))
    return np.clip(out, -32768.0, 32767.0).astype(np.int16).tobytes()


class RingBuffer:
    """Fixed-length ring of audio frames used for the pre-roll (SPEC §7 step 3)."""

    def __init__(self, max_frames: int) -> None:
        self._frames: collections.deque[bytes] = collections.deque(maxlen=max(0, int(max_frames)))

    def push(self, frame: bytes) -> None:
        self._frames.append(frame)

    def snapshot(self) -> bytes:
        return b"".join(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._frames)


class AudioInput:
    """Microphone capture: mono int16 frames of ``frame_ms`` at ``sample_rate``."""

    def __init__(
        self,
        device: DeviceSpec = None,
        sample_rate: int = 16000,
        frame_ms: int = FRAME_MS,
        max_queue_frames: int = 600,
        processor=None,
    ) -> None:
        self.device = device
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.blocksize = frame_samples(self.sample_rate, self.frame_ms)
        self.frame_bytes = self.blocksize * SAMPLE_WIDTH
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=max(10, int(max_queue_frames)))
        self._stream: sd.RawInputStream | None = None
        self._dropped = 0
        self._processor = processor
        self._raw_queue = queue.Queue(maxsize=8)
        self._worker = None
        self._worker_stop = threading.Event()
        self._generation = 0

    # -- PortAudio thread ------------------------------------------------
    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            log.debug("Microphone: stream status %s", status)
        data = bytes(indata)
        if self._worker is not None:
            from .audio_processing import capture_time
            item = (data, capture_time(time_info, frames, self.sample_rate), self._generation)
            try:
                self._raw_queue.put_nowait(item)
            except queue.Full:
                try:
                    self._raw_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._raw_queue.put_nowait(item)
                except queue.Full:
                    pass
            return
        self._put_frame(data)

    def _put_frame(self, data: bytes) -> None:
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(data)
            except queue.Full:
                pass
            self._dropped += 1
            if self._dropped % 200 == 1:
                log.warning("The microphone queue is full, frames are dropped (%d)", self._dropped)

    # -- control ---------------------------------------------------------
    def start(self) -> None:
        if self._stream is not None:
            return
        if self._processor is not None:
            try:
                self._processor.start()
                self._worker_stop.clear()
                self._worker = threading.Thread(target=self._process_frames, name='jarvis-audio-dsp', daemon=True)
                self._worker.start()
            except Exception as exc:
                log.warning('Audio processing unavailable; using original microphone: %s', exc)
                self._processor.close()
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            device=self.device,
            channels=1,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()
        log.info(
            "Microphone started: device=%s, %d Hz, frame %d ms (%d samples)",
            self.device if self.device is not None else "default",
            self.sample_rate,
            self.frame_ms,
            self.blocksize,
        )

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        self._worker_stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2)
            self._worker = None
        if self._processor is not None:
            self._processor.close()
        if stream is None:
            return
        try:
            stream.stop()
        except Exception as exc:  # pragma: no cover - device teardown
            log.debug("Error while stopping the microphone: %s", exc)
        try:
            stream.close()
        except Exception as exc:  # pragma: no cover - device teardown
            log.debug("Error while closing the microphone: %s", exc)

    def close(self) -> None:
        self.stop()
        self.clear()

    def _process_frames(self):
        while not self._worker_stop.is_set():
            try:
                data, started, generation = self._raw_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            # Never replay stale commands after a temporary processing stall.
            if time.monotonic() - started < 0.5:
                processed = self._processor.process(data, started)
                if generation == self._generation and not self._worker_stop.is_set():
                    self._put_frame(processed)

    @property
    def running(self) -> bool:
        return self._stream is not None

    @property
    def processor(self):
        """The local DSP stage, or ``None`` (ТЗ F-102 reads its ``aec_active``)."""
        return self._processor

    def clear(self) -> None:
        """Drop everything captured so far (echo of our own playback, etc.)."""
        self._generation += 1
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def read_frame_nowait(self) -> bytes | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def _get_blocking(self, timeout: float) -> bytes | None:
        try:
            return self._queue.get(True, timeout)
        except queue.Empty:
            return None

    async def read_frame(self, timeout: float = 0.5) -> bytes | None:
        """Await one captured frame; ``None`` if nothing arrived within ``timeout``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_blocking, float(timeout))


class AudioOutput:
    """Raw PCM playback at an arbitrary samplerate + generated beeps."""

    def __init__(self, device: DeviceSpec = None, default_sample_rate: int = 16000) -> None:
        self.device = device
        self.default_sample_rate = int(default_sample_rate)
        self._stream: sd.RawOutputStream | None = None
        self._declared_rate: int | None = None   # rate of the PCM we are handed
        self._device_rate: int | None = None     # rate the device actually runs at
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._writer: asyncio.Task | None = None
        self._tail = b""

    # -- stream management ----------------------------------------------
    @property
    def sample_rate(self) -> int:
        """Rate of the PCM currently accepted (falls back to the configured default)."""
        return self._declared_rate or self.default_sample_rate

    def _open_stream(self, rate: int) -> sd.RawOutputStream:
        stream = sd.RawOutputStream(
            samplerate=rate,
            device=self.device,
            channels=1,
            dtype="int16",
        )
        stream.start()
        return stream

    async def open(self, samplerate: int) -> None:
        """Make sure an output stream able to play ``samplerate`` PCM is running."""
        samplerate = int(samplerate)
        if self._stream is not None and self._declared_rate == samplerate:
            return
        await self.drain()
        self._close_stream()
        self._tail = b""
        device_rate = samplerate
        try:
            stream = await asyncio.to_thread(self._open_stream, samplerate)
        except Exception as exc:
            fallback = self._device_default_rate()
            if fallback is None or int(fallback) == samplerate:
                log.error("Could not open the audio output at %d Hz: %s", samplerate, exc)
                raise
            log.warning(
                "The device refused %d Hz (%s), opening at %d Hz with software resampling",
                samplerate, exc, int(fallback),
            )
            device_rate = int(fallback)
            stream = await asyncio.to_thread(self._open_stream, device_rate)
        self._stream = stream
        self._declared_rate = samplerate
        self._device_rate = device_rate
        self._ensure_writer()
        log.debug("Audio output open: %d Hz (device at %d Hz)", samplerate, device_rate)

    def _device_default_rate(self) -> float | None:
        try:
            info = sd.query_devices(self.device, "output")
            return float(info["default_samplerate"])
        except Exception as exc:  # pragma: no cover - depends on host audio
            log.debug("Could not query the output device parameters: %s", exc)
            return None

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        self._declared_rate = None
        self._device_rate = None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception as exc:  # pragma: no cover - device teardown
            log.debug("Error while stopping the audio output: %s", exc)
        try:
            stream.close()
        except Exception as exc:  # pragma: no cover - device teardown
            log.debug("Error while closing the audio output: %s", exc)

    # -- writer task -----------------------------------------------------
    def _ensure_writer(self) -> None:
        if self._writer is None or self._writer.done():
            self._writer = asyncio.get_running_loop().create_task(
                self._writer_loop(), name="jarvis-audio-writer"
            )

    @staticmethod
    def _write_blocking(stream: sd.RawOutputStream, data: bytes) -> None:
        stream.write(data)

    async def _writer_loop(self) -> None:
        while True:
            chunk = await self._queue.get()
            try:
                stream = self._stream
                if stream is not None and chunk:
                    await asyncio.to_thread(self._write_blocking, stream, chunk)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("Playback error: %s", exc)
            finally:
                self._queue.task_done()

    # -- playback --------------------------------------------------------
    async def write(self, pcm: bytes) -> None:
        """Queue raw PCM s16le (at the rate passed to :meth:`open`) for playback."""
        if not pcm:
            return
        if self._stream is None:
            await self.open(self.default_sample_rate)
        data = self._tail + bytes(pcm)
        extra = len(data) % SAMPLE_WIDTH
        if extra:
            data, self._tail = data[:-extra], data[-extra:]
        else:
            self._tail = b""
        if not data:
            return
        if self._device_rate is not None and self._declared_rate is not None:
            data = resample_pcm(data, self._declared_rate, self._device_rate)
        self._ensure_writer()
        self._queue.put_nowait(data)

    def cancel_pending(self) -> int:
        """Drop everything queued but not yet written (barge-in interrupt).

        The chunk currently inside ``stream.write`` still finishes — chunks are
        short, so speech stops within a fraction of a second.
        """
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._queue.task_done()
            dropped += 1
        self._tail = b""
        return dropped

    def abort(self) -> int:
        """Go silent NOW: drop the queue *and* the device's own buffer (F-102).

        ``cancel_pending`` empties the queue, but the audio device still holds
        whatever PortAudio already accepted — with a 100 ms device block that
        is another 100 ms of speech after the person started interrupting. The
        ТЗ 15.1 budget for barge-in is 200 ms, so the stream itself is aborted
        (which discards the queued device buffer) instead of stopped politely.

        The stream is then dropped, so the next reply reopens it — reopening is
        an output concern of that reply, not of the interruption.
        """
        dropped = self.cancel_pending()
        stream, self._stream = self._stream, None
        self._declared_rate = None
        self._device_rate = None
        if stream is None:
            return dropped
        abort = getattr(stream, "abort", None)
        try:
            if callable(abort):
                abort()
            else:  # pragma: no cover - a stream stub without abort()
                stream.stop()
        except Exception as exc:  # pragma: no cover - device teardown
            log.debug("Error while aborting the audio output: %s", exc)
        try:
            stream.close()
        except Exception as exc:  # pragma: no cover - device teardown
            log.debug("Error while closing the aborted audio output: %s", exc)
        return dropped

    async def drain(self) -> None:
        """Wait until everything queued has been handed to the device and played."""
        if self._writer is None:
            return
        try:
            await self._queue.join()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("Error while waiting for playback to finish: %s", exc)
        stream = self._stream
        if stream is not None:
            try:
                latency = float(getattr(stream, "latency", 0.1) or 0.1)
            except Exception:  # pragma: no cover - defensive
                latency = 0.1
            await asyncio.sleep(min(1.0, max(0.05, latency)))

    async def play_pcm(self, pcm: bytes, samplerate: int) -> None:
        await self.open(samplerate)
        await self.write(pcm)
        await self.drain()

    async def play_beep(
        self,
        freq: float = BEEP_FREQ_HZ,
        ms: int = BEEP_MS,
        volume: float = 0.35,
    ) -> None:
        """Play a short generated sine beep on the current output stream."""
        rate = self._declared_rate or self.default_sample_rate
        try:
            await self.open(rate)
            await self.write(make_tone(freq, ms, rate, volume))
            await self.drain()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("Could not play the beep: %s", exc)

    # -- teardown --------------------------------------------------------
    def close(self) -> None:
        writer, self._writer = self._writer, None
        if writer is not None and not writer.done():
            writer.cancel()
        self._close_stream()

    async def aclose(self) -> None:
        writer, self._writer = self._writer, None
        if writer is not None and not writer.done():
            writer.cancel()
            try:
                await writer
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # pragma: no cover - defensive
                log.debug("Error while stopping the audio writer task: %s", exc)
        self._close_stream()
