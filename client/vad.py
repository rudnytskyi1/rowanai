"""Voice activity detection / utterance segmentation (SPEC §7).

``webrtcvad`` accepts only 10, 20 or 30 ms frames of 8/16/32/48 kHz mono
int16 PCM — the client captures exactly 30 ms frames, so frames go straight
from :mod:`client.audio` into the VAD.

An utterance ends once the last ``silence_ms`` worth of frames is (almost)
entirely non-speech — sporadic false "speech" frames from a noisy mic are
tolerated — or when ``max_utterance_s`` of speech has been recorded. If no speech starts within the
lead-in timeout (5 s by default, or the follow-up window), the recorder returns
``None``.
"""

from __future__ import annotations

import collections
import inspect
import logging
import math
import re
import statistics
import time
from array import array
from collections.abc import Awaitable, Callable

try:
    import webrtcvad  # provided by the `webrtcvad-wheels` package on Windows
except ImportError as exc:  # pragma: no cover - depends on installation
    raise ImportError(
        "The webrtcvad-wheels package is not installed. Install the client "
        "dependencies: pip install -r client/requirements.txt"
    ) from exc

from client.audio import FRAME_MS, SAMPLE_WIDTH

log = logging.getLogger(__name__)

#: Rates accepted by webrtcvad.
VALID_SAMPLE_RATES = (8000, 16000, 32000, 48000)
#: Frame lengths accepted by webrtcvad.
VALID_FRAME_MS = (10, 20, 30)
#: Default lead-in: how long we wait for the user to start speaking.
DEFAULT_LEAD_IN_S = 5.0
#: The user always gets at least this long to start talking (after the wake
#: word, after a reply, and again after a discarded noise blip).
MIN_LEAD_IN_S = 3.0
#: A "started" recording whose voiced content never reaches min_speech_ms is a
#: noise blip: it is discarded without ever being streamed to the server, and
#: listening continues. At most this many blips per record() call.
MAX_NOISE_RESETS = 3
#: Default amount of pre-speech audio kept in front of the utterance (SPEC §6
#: ``client.vad.pre_roll_ms``); the rest of the lead-in silence is discarded.
DEFAULT_PRE_ROLL_MS = 300
#: If the microphone stops delivering frames mid-utterance, give up after this.
STALL_TIMEOUT_S = 3.0

#: Владелец 2026-09-23: «если я на секунду перестану говорить, запись уже
#: останавливается — надо грамотно сделать». Слова, после которых пауза почти
#: наверняка не конец реплики, а раздумье перед продолжением: «открой ютуб И
#: включи видео». Живая расшифровка приходит в клиент кадром
#: ``transcript_partial`` (хаб её уже считает), поэтому вопрос «закончил ли
#: человек» решается по её последнему слову, а не по громкости.
UNFINISHED_TAIL_WORDS = frozenset({
    "a", "also", "an", "and", "as", "at", "because", "but", "by", "for",
    "from", "in", "into", "of", "on", "or", "so", "than", "that", "the",
    "then", "to", "with",
    "а", "без", "будет", "бы", "в", "во", "вот", "для", "до", "если", "же",
    "за", "и", "или", "к", "как", "ко", "на", "надо", "но", "ну", "о", "об",
    "от", "по", "потом", "при", "про", "с", "со", "так", "то", "у", "чтобы",
    "это", "я",
    "entonces", "para", "pero", "por", "que", "y",
})

_TAIL_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def last_word(text: str) -> str:
    """Последнее слово расшифровки (без регистра и знаков), ``""`` если пусто."""
    found = _TAIL_WORD.findall(str(text or ""))
    return found[-1].lower() if found else ""


def sentence_unfinished(text: str) -> bool:
    """Закончилось ли сказанное на слове-связке («и», «потом», «and»)."""
    return last_word(text) in UNFINISHED_TAIL_WORDS

FrameReader = Callable[[], Awaitable[bytes | None]]
AudioSink = Callable[[bytes], None | Awaitable[None]]


class VadRecorder:
    """Records one utterance from an async frame source using webrtcvad."""

    def __init__(
        self,
        aggressiveness: int = 2,
        silence_ms: int = 800,
        hold_ms: int = 0,
        max_utterance_s: float = 15.0,
        sample_rate: int = 16000,
        frame_ms: int = FRAME_MS,
        lead_in_s: float = DEFAULT_LEAD_IN_S,
        onset_frames: int = 3,
        onset_window: int = 5,
        pre_roll_ms: int = DEFAULT_PRE_ROLL_MS,
        min_speech_ms: int = 250,
        energy_endpoint: bool = False,
        hold_while: Callable[[], bool] | None = None,
    ) -> None:
        sample_rate = int(sample_rate)
        frame_ms = int(frame_ms)
        if sample_rate not in VALID_SAMPLE_RATES:
            raise ValueError(
                f"client.audio.sample_rate={sample_rate} is not supported by the VAD; "
                f"allowed: {VALID_SAMPLE_RATES}"
            )
        if frame_ms not in VALID_FRAME_MS:
            raise ValueError(
                f"A frame length of {frame_ms} ms is not supported by the VAD; "
                f"allowed: {VALID_FRAME_MS}"
            )
        aggressiveness = max(0, min(3, int(aggressiveness)))

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * frame_ms // 1000) * SAMPLE_WIDTH
        self.aggressiveness = aggressiveness
        self.silence_ms = max(frame_ms, int(silence_ms))
        #: Extra silence the recorder tolerates while the caller says the
        #: sentence is not finished yet (``hold_while``). 0 keeps the old
        #: behaviour: the first quiet window ends the utterance.
        self.hold_ms = max(0, int(hold_ms))
        #: Вопрос «человек ещё не договорил?», если он известен на всю жизнь
        #: этого рекордера. Он же может прийти и в конкретный вызов ``record``;
        #: аргумент вызова сильнее. Так ``client.main`` задаёт его один раз, и
        #: подменённые в тестах рекордеры продолжают работать без изменений.
        self.hold_while = hold_while
        self.max_utterance_s = max(1.0, float(max_utterance_s))
        self.lead_in_s = max(0.5, float(lead_in_s))
        self.onset_window = max(1, int(onset_window))
        self.onset_frames = max(1, min(int(onset_frames), self.onset_window))
        # Pre-speech frames kept in front of the utterance: pre_roll_ms worth, but
        # never fewer than the onset window, or the first words would be clipped.
        try:
            pre_roll_frames = int(round(float(pre_roll_ms) / self.frame_ms))
        except (TypeError, ValueError):
            pre_roll_frames = int(round(DEFAULT_PRE_ROLL_MS / self.frame_ms))
        self.pre_roll_frames = max(self.onset_window, pre_roll_frames)
        self.min_speech_frames = max(1, int(round(max(0, int(min_speech_ms)) / self.frame_ms)))
        self.energy_endpoint = bool(energy_endpoint)
        self._vad = webrtcvad.Vad(aggressiveness)

    # -- helpers ---------------------------------------------------------
    def is_speech(self, frame: bytes) -> bool:
        """True if this exact-length frame contains speech."""
        if len(frame) != self.frame_bytes:
            return False
        try:
            return bool(self._vad.is_speech(frame, self.sample_rate))
        except Exception as exc:  # pragma: no cover - native errors
            log.debug("VAD error: %s", exc)
            return False

    @staticmethod
    def _level_dbfs(frame: bytes) -> float:
        samples = array('h', frame)
        energy = sum(value * value for value in samples) / max(1, len(samples))
        return 10 * math.log10(max(energy, 1e-6) / (32768 ** 2))

    @staticmethod
    async def _emit(sink: AudioSink | None, data: bytes) -> None:
        if sink is None or not data:
            return
        result = sink(data)
        if inspect.isawaitable(result):
            await result

    # -- main API --------------------------------------------------------
    async def record(
        self,
        read_frame: FrameReader,
        pre_roll: bytes = b"",
        lead_in_s: float | None = None,
        on_audio: AudioSink | None = None,
        hold_while: Callable[[], bool] | None = None,
    ) -> bytes | None:
        """Record one utterance.

        ``read_frame`` is awaited repeatedly and must return a chunk of PCM
        (normally exactly one frame) or ``None`` on timeout. ``pre_roll`` is
        prepended to the recording (SPEC §7 step 3). ``on_audio`` — if given —
        receives the audio as it becomes part of the utterance: one flush of
        ``pre_roll`` plus the last ``pre_roll_ms`` before speech was detected,
        then one frame at a time, so the caller can stream to the server live
        and send nothing at all when the trigger was false.

        ``hold_while`` — необязательный вопрос «человек ещё не договорил?».
        Пока он отвечает «да», пауза не закрывает реплику: запись терпит ещё
        ``hold_ms`` тишины (владелец 2026-09-23: пауза на секунду обрывала
        запись). Без ``hold_ms`` или без вопроса поведение прежнее.

        Returns the recorded PCM, or ``None`` if no speech started in time.
        """
        lead_in = self.lead_in_s if lead_in_s is None else float(lead_in_s)
        if hold_while is None:
            hold_while = self.hold_while
        # The user always gets at least MIN_LEAD_IN_S to start talking.
        lead_in = max(MIN_LEAD_IN_S, lead_in)
        base_roll: list[bytes] = [pre_roll] if pre_roll else []
        collected: list[bytes] = list(base_roll)
        # Frames captured before speech starts: a bounded ring, so a long lead-in
        # (5 s, or followup_window_s) is not prepended to the utterance.
        lead_in_ring: collections.deque[bytes] = collections.deque(maxlen=self.pre_roll_frames)
        pending = b""
        window: collections.deque[bool] = collections.deque(maxlen=self.onset_window)
        started = False
        finished = False
        silence_limit = max(1, int(round(self.silence_ms / self.frame_ms)))
        # End-of-utterance detection tolerates sporadic false "speech" frames:
        # requiring silence_limit CONSECUTIVE non-speech frames lets a noisy mic
        # (webcam AGC, TV hum) reset the counter over and over, stretching the
        # 0.8 s tail into many seconds. Instead the utterance ends when at most
        # 10% of the last silence_ms worth of frames were flagged as speech.
        tail: collections.deque[bool] = collections.deque(maxlen=silence_limit)
        tail_allowed_speech = max(0, int(silence_limit * 0.1))
        # Сколько кадров тишины запись уже терпит СВЕРХ обычного окна, пока
        # ``hold_while`` говорит, что человек не договорил.
        hold_limit = max(0, int(round(self.hold_ms / self.frame_ms)))
        held_frames = 0
        # Nothing is streamed to the server until min_speech_frames voiced frames
        # have been seen. A "recording" that ends before that is a noise blip: it
        # is dropped without the server ever hearing about it, and listening
        # resumes so noise cannot eat the user's window to speak.
        emitted = False
        voiced = 0
        noise_resets = 0
        speech_started_at = 0.0
        recorded_frames = 0
        max_frames = max(1, int(math.ceil(self.max_utterance_s * 1000 / self.frame_ms)))
        levels: collections.deque[float] = collections.deque(maxlen=max(3, int(round(300 / self.frame_ms))))
        level_speech: collections.deque[bool] = collections.deque(maxlen=levels.maxlen)
        speech_level = None
        now = time.monotonic()
        deadline = now + lead_in
        last_frame_at = now

        def _reset_listening() -> None:
            nonlocal started, voiced, emitted, collected, recorded_frames, speech_level
            nonlocal held_frames
            started = False
            voiced = 0
            emitted = False
            held_frames = 0
            collected = list(base_roll)
            window.clear()
            tail.clear()
            lead_in_ring.clear()
            recorded_frames = 0
            speech_level = None
            levels.clear()
            level_speech.clear()

        while not finished:
            chunk = await read_frame()
            now = time.monotonic()
            if not chunk:
                if not started and now >= deadline:
                    return None
                if started and (now - last_frame_at) >= STALL_TIMEOUT_S:
                    log.warning("The microphone stopped delivering frames - ending the recording")
                    break
                continue
            last_frame_at = now
            pending += chunk

            noise_blip = False
            while len(pending) >= self.frame_bytes:
                frame = pending[: self.frame_bytes]
                pending = pending[self.frame_bytes :]
                speech = self.is_speech(frame)

                if not started:
                    lead_in_ring.append(frame)
                    window.append(speech)
                    if sum(window) >= self.onset_frames:
                        started = True
                        speech_started_at = now
                        recorded_frames = len(window)
                        voiced = sum(window)
                        tail.clear()
                        collected.extend(lead_in_ring)
                        lead_in_ring.clear()
                        log.debug("Possible speech onset, buffering...")
                    continue

                collected.append(frame)
                recorded_frames += 1
                if speech:
                    voiced += 1
                if not emitted and voiced >= self.min_speech_frames:
                    emitted = True
                    log.info("Speech detected, recording...")
                    await self._emit(on_audio, b"".join(collected))
                elif emitted:
                    await self._emit(on_audio, frame)
                endpoint_speech = speech
                if self.energy_endpoint:
                    level = self._level_dbfs(frame)
                    levels.append(level)
                    level_speech.append(speech)
                    # A sustained 300 ms speech level supplies a reference;
                    # single clicks cannot raise it. Hold the reference so a
                    # noise-only tail cannot slowly redefine itself as speech.
                    if len(levels) == levels.maxlen and sum(level_speech) >= len(level_speech) / 2:
                        sustained = statistics.median(levels)
                        speech_level = sustained if speech_level is None else max(speech_level, sustained)
                    if speech_level is not None:
                        # Relative to this speaker, never a fixed minimum mic
                        # volume. The -42 dBFS ceiling keeps quiet continuations
                        # eligible after an unusually loud word.
                        threshold = min(-42.0, speech_level - 18.0)
                        endpoint_speech = speech and level >= threshold
                tail.append(endpoint_speech)
                if endpoint_speech:
                    held_frames = 0
                if len(tail) == silence_limit and sum(tail) <= tail_allowed_speech:
                    if not emitted:
                        noise_blip = True
                        break
                    if hold_limit:
                        # The ordinary window is over, but the sentence does not
                        # look finished (see ``UNFINISHED_TAIL_WORDS``). Keep
                        # recording until the hold runs out or the speaker
                        # resumes; the person's next words belong to THIS turn,
                        # so they are never lost to a one-second pause.
                        held_frames += 1
                        holding = False
                        if held_frames <= hold_limit and hold_while is not None:
                            try:
                                holding = bool(hold_while())
                            except Exception as exc:  # noqa: BLE001 - a hint never breaks capture
                                log.debug("hold_while failed (%s); the pause ends the utterance", exc)
                                holding = False
                        if holding:
                            continue
                        if held_frames <= hold_limit:
                            log.debug("Hold ended: the sentence reads as finished")
                    log.debug(
                        "~%d ms of (near-)silence - end of the utterance",
                        silence_limit * self.frame_ms,
                    )
                    finished = True
                    break
                if recorded_frames >= max_frames or (now - speech_started_at) >= self.max_utterance_s:
                    if not emitted:
                        noise_blip = True
                        break
                    log.info("Maximum utterance length reached (%.0f s)", self.max_utterance_s)
                    finished = True
                    break

            if noise_blip:
                noise_resets += 1
                log.info(
                    "Discarded a noise blip (%d/%d) - still listening",
                    noise_resets,
                    MAX_NOISE_RESETS,
                )
                if noise_resets >= MAX_NOISE_RESETS:
                    return None
                _reset_listening()
                deadline = max(deadline, time.monotonic() + MIN_LEAD_IN_S)
                continue

            if not started and time.monotonic() >= deadline:
                return None

        if not emitted:
            return None
        audio = b"".join(collected)
        log.info("Recorded %.2f s of audio", len(audio) / float(self.sample_rate * SAMPLE_WIDTH))
        return audio

    async def wait_for_speech(self, read_frame: FrameReader, timeout_s: float) -> bool:
        """True if speech starts within ``timeout_s`` (frames are consumed)."""
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        window: collections.deque[bool] = collections.deque(maxlen=self.onset_window)
        pending = b""
        while time.monotonic() < deadline:
            chunk = await read_frame()
            if not chunk:
                continue
            pending += chunk
            while len(pending) >= self.frame_bytes:
                frame = pending[: self.frame_bytes]
                pending = pending[self.frame_bytes :]
                window.append(self.is_speech(frame))
                if sum(window) >= self.onset_frames:
                    return True
        return False
