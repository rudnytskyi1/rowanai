"""Local WebRTC processing with a timestamped Windows speaker reference.

The reference is WASAPI loopback of the *actual* playback endpoint, including
browser audio. Neither reference nor continuous microphone audio is saved.
Processing runs in AudioInput's worker, never in the microphone callback.
"""
from __future__ import annotations

from collections import deque
import logging
import threading
import time

import numpy as np

log = logging.getLogger(__name__)


class AudioLevelWindow:
    """Summarize the real capture path without retaining room audio."""

    def __init__(self):
        self.frames = 0
        self.samples = 0
        self.raw_energy = self.processed_energy = 0.0
        self.raw_peak = self.processed_peak = 0

    def observe(self, raw: bytes, processed: bytes):
        before = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        after = np.frombuffer(processed, dtype=np.int16).astype(np.float64)
        if not before.size or before.size != after.size:
            return None
        self.frames += 1
        self.samples += before.size
        self.raw_energy += float(before @ before)
        self.processed_energy += float(after @ after)
        self.raw_peak = max(self.raw_peak, int(np.abs(before).max()))
        self.processed_peak = max(self.processed_peak, int(np.abs(after).max()))
        if self.frames < 1000:
            return None
        def dbfs(energy):
            return round(float(10 * np.log10(max(energy / self.samples, 1e-6) / 32768**2)), 1)
        report = dict(frames=self.frames, raw_rms_dbfs=dbfs(self.raw_energy),
                      processed_rms_dbfs=dbfs(self.processed_energy),
                      raw_peak=self.raw_peak, processed_peak=self.processed_peak)
        self.__init__()
        return report


def capture_time(time_info, frames: int, rate: int, *, now=None) -> float:
    """Translate PortAudio's stream clock into our monotonic timeline."""
    now = time.monotonic() if now is None else now
    try:
        if isinstance(time_info, dict):
            age = float(time_info['current_time']) - float(time_info['input_buffer_adc_time'])
        else:
            age = float(time_info.currentTime) - float(time_info.inputBufferAdcTime)
        if 0 < age < 2:
            return now - age
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return now - frames / rate


class ReferenceTimeline:
    """Bounded, timestamped PCM blocks; missing time is silence, never old audio."""

    def __init__(self, rate: int, seconds: float = 3.0):
        self.rate = rate
        self.seconds = seconds
        self._blocks = deque(maxlen=400)
        self._lock = threading.Lock()

    def push(self, pcm: bytes, rate: int, channels: int, started: float):
        # Copy in the capture callback; conversion is deferred to the DSP worker.
        with self._lock:
            self._blocks.append((started, bytes(pcm), rate, channels))
            while self._blocks and self._blocks[0][0] < started - self.seconds:
                self._blocks.popleft()

    def read(self, started: float, count: int) -> np.ndarray:
        result = np.zeros(count, dtype=np.int16)
        end = started + count / self.rate
        with self._lock:
            blocks = list(self._blocks)
        for at, pcm, rate, channels in blocks:
            size = len(pcm) // (2 * channels)
            if at >= end or at + size / rate <= started:
                continue
            samples = np.frombuffer(pcm, dtype=np.int16).reshape(-1, channels).mean(axis=1)
            first = max(0, int(np.ceil((at - started) * self.rate)))
            last = min(count, int(np.ceil((at + size / rate - started) * self.rate)))
            positions = (started + np.arange(first, last) / self.rate - at) * rate
            result[first:last] = np.interp(positions, np.arange(size), samples).astype(np.int16)
        return result


def select_loopback(devices, output_name: str):
    """Require one matching endpoint; never silently cancel an unrelated device."""
    key = output_name.casefold().strip()
    exact = [d for d in devices if d['name'].casefold().replace(' [loopback]', '').strip() == key]
    if len(exact) == 1:
        return exact[0]
    matches = [d for d in devices if key and key in d['name'].casefold()]
    if len(matches) != 1:
        raise RuntimeError(f'No unique WASAPI loopback for output {output_name!r}')
    return matches[0]


class LoopbackReference:
    def __init__(self, timeline: ReferenceTimeline, output_device=None):
        self.timeline = timeline
        self.output_device = output_device
        self._pa = None
        self._stream = None

    def start(self):
        import pyaudiowpatch as pa
        import sounddevice as sd

        self._pa = pa.PyAudio()
        try:
            if self.output_device is None:
                host = self._pa.get_host_api_info_by_type(pa.paWASAPI)
                name = self._pa.get_device_info_by_index(host['defaultOutputDevice'])['name']
            else:
                name = sd.query_devices(self.output_device, 'output')['name']
            device = select_loopback(list(self._pa.get_loopback_device_info_generator()), name)
            rate = int(device['defaultSampleRate'])
            channels = int(device['maxInputChannels'])

            def callback(data, frames, timing, status):
                self.timeline.push(data, rate, channels, capture_time(timing, frames, rate))
                return None, pa.paContinue

            self._stream = self._pa.open(
                format=pa.paInt16, channels=channels, rate=rate,
                frames_per_buffer=rate // 100, input=True,
                input_device_index=device['index'], stream_callback=callback,
            )
            log.info('AEC speaker reference ready: %s (%d Hz)', device['name'], rate)
        except Exception:
            self.close()
            raise

    def close(self):
        stream, self._stream = self._stream, None
        try:
            if stream is not None:
                try:
                    stream.stop_stream()
                finally:
                    stream.close()
        finally:
            pa, self._pa = self._pa, None
            if pa is not None:
                pa.terminate()


class AudioPreprocessor:
    def __init__(self, rate: int, *, echo_cancellation=False,
                 noise_suppression=False, ns_level=1, output_device=None):
        self.rate = rate
        self.echo_cancellation = echo_cancellation
        self.noise_suppression = noise_suppression
        self.ns_level = ns_level
        self.output_device = output_device
        self.timeline = ReferenceTimeline(rate)
        self.reference = None
        self.processor = None
        self.frames = 0
        self.processing_ms = 0.0
        self.levels = AudioLevelWindow()

    def start(self):
        from pywebrtc_audio import AudioProcessor

        aec = self.echo_cancellation
        if aec:
            self.reference = LoopbackReference(self.timeline, self.output_device)
            try:
                self.reference.start()
            except Exception as exc:
                log.warning('AEC unavailable; microphone remains usable with noise reduction: %s', exc)
                self.reference = None
                aec = False
        self.processor = AudioProcessor(
            sample_rate=self.rate, echo_cancellation=aec,
            noise_suppression=self.noise_suppression, ns_level=self.ns_level,
            high_pass_filter=True, auto_gain_control=False,
        )
        log.info('Local audio processing ready: echo=%s, noise=%s, AGC=off', aec, self.noise_suppression)

    def process(self, pcm: bytes, started: float) -> bytes:
        if self.processor is None:
            return pcm
        before = time.perf_counter()
        try:
            near = np.frombuffer(pcm, dtype=np.int16)
            far = self.timeline.read(started, len(near)) if self.reference else None
            result = self.processor.process(near, far).tobytes()
        except Exception:
            log.exception('Audio DSP failed; falling back to the original microphone signal')
            self.processor = None
            return pcm
        self.frames += 1
        self.processing_ms += (time.perf_counter() - before) * 1000
        report = self.levels.observe(pcm, result)
        if report is not None:
            log.info('Microphone signal (30 s): %s; DSP %.2f ms/frame',
                     report, self.processing_ms / self.frames)
        return result

    def close(self):
        if self.reference is not None:
            self.reference.close()
            self.reference = None
        self.processor = None
