"""Barge-in: the room cuts Rowan's playback short (ТЗ F-102).

The rule from the spec has two halves, and both live here:

* speaking over Rowan stops the playback — the ТЗ 15.1 budget is
  ``barge-in → тишина ≤ 200 мс``, measured from the moment the microphone
  confirms real speech to the moment the speaker is actually silent;
* the utterance is recorded and handed on as a normal turn.

The second half of F-102 is a condition, not a nicety: barge-in *requires*
echo cancellation (WebRTC AEC) on the client. Without it the microphone hears
Rowan's own voice through the speakers, so every reply would interrupt itself.
When the AEC is missing the feature is switched off and the HUD says so.

This module holds the pure parts — how much speech counts as speech, whether
the feature may be on, and the stopwatch that checks the 200 ms budget — so the
client loop stays a wiring layer and the rules can be tested without a
microphone, a speaker or a GPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

#: ТЗ 15.1: from the start of the interruption to silence.
BARGE_IN_BUDGET_MS = 200

#: Speech must last this long before the playback is cut. One frame of a cough,
#: a door or the tail of Rowan's own audio must not stop a sentence.
SPEECH_CONFIRM_MS = 150

#: Shown in the HUD when F-102 cannot work on this machine.
NO_AEC_WARNING = (
    "Barge-in is off: this PC has no echo cancellation (WebRTC AEC). "
    "Rowan finishes its reply before listening again."
)


class SpeechGate:
    """``confirm_ms`` of speech inside one uninterrupted stretch.

    A single voiced frame is not an interruption; a person talking over Rowan
    keeps talking. The gate resets on silence, so only *continuous* speech
    counts, and it reports ``True`` exactly once per interruption.
    """

    def __init__(self, frame_ms: int, *, confirm_ms: int = SPEECH_CONFIRM_MS) -> None:
        self.frame_ms = max(1, int(frame_ms))
        self.confirm_ms = max(self.frame_ms, int(confirm_ms))
        self._voiced_ms = 0
        self.fired = False
        #: When the uninterrupted stretch of speech began (ТЗ 15.1 measures the
        #: 200 ms budget from the interruption, not from the confirmation).
        self.started_at: float | None = None

    @property
    def frames_needed(self) -> int:
        return max(1, -(-self.confirm_ms // self.frame_ms))

    def accept(self, is_speech: bool) -> bool:
        """Feed one frame; ``True`` on the frame that confirms the barge-in."""
        if not is_speech:
            self.reset()
            return False
        if self.started_at is None:
            self.started_at = time.monotonic()
        if self.fired:
            return False
        self._voiced_ms += self.frame_ms
        if self._voiced_ms < self.confirm_ms:
            return False
        self.fired = True
        return True

    def reset(self) -> None:
        self._voiced_ms = 0
        self.fired = False
        self.started_at = None

    @property
    def voiced_ms(self) -> int:
        return self._voiced_ms


@dataclass(frozen=True)
class BargeInAvailability:
    """May this client cut its own playback short (ТЗ F-102)?

    Two conditions, both necessary: the owner left the feature on in the
    config, and the WebRTC echo canceller is really running. The second one is
    a measurement, not a wish: ``aec_active`` is set by the audio pipeline
    after the loopback reference and the canceller came up.
    """

    configured: bool = True
    aec_active: bool = False

    @property
    def speech(self) -> bool:
        """Barge-in on real speech (no wake word needed)."""
        return bool(self.configured and self.aec_active)

    @property
    def wake_word(self) -> bool:
        """The wake-word interruption is always available.

        Saying the wake word is an explicit request and the detector is built
        for that; it does not need the canceller to decide what it heard.
        """
        return bool(self.configured)

    @property
    def warning(self) -> str:
        """The HUD line, or ``""`` when barge-in is available."""
        return "" if self.speech else NO_AEC_WARNING

    def detail(self) -> str:
        return (f"barge-in: configured={self.configured}, aec={self.aec_active}"
                f", speech={self.speech}, wake word={self.wake_word}")


@dataclass
class BargeInStop:
    """The measured interruption: speech confirmed → the speaker went quiet.

    ``stop`` is called after the playback queue and the device buffer are
    actually dropped, which is what the person in the room experiences; the
    budget of ТЗ 15.1 (200 ms) is checked here rather than assumed.
    """

    speech_at: float
    stopped_at: float | None = None
    reason: str = "speech"
    extra: dict = field(default_factory=dict)

    def stop(self, *, now: float | None = None) -> float:
        """Record the moment of silence; the first call wins."""
        if self.stopped_at is None:
            self.stopped_at = time.monotonic() if now is None else now
        return self.delay_ms()

    def delay_ms(self) -> int:
        if self.stopped_at is None:
            return 0
        return max(0, int(round((self.stopped_at - self.speech_at) * 1000.0)))

    def met(self, *, budget_ms: int = BARGE_IN_BUDGET_MS) -> bool:
        return self.stopped_at is not None and self.delay_ms() <= budget_ms

    def report(self, *, budget_ms: int = BARGE_IN_BUDGET_MS) -> tuple[bool, str]:
        """``(inside the budget, the line to log)``."""
        delay = self.delay_ms()
        inside = self.met(budget_ms=budget_ms)
        if inside:
            return True, (f"Barge-in ({self.reason}): silence {delay} ms after the "
                          f"interruption (budget {budget_ms} ms)")
        return False, (f"Barge-in ({self.reason}): silence took {delay} ms - over the "
                       f"{budget_ms} ms budget of ТЗ F-102")


__all__ = [
    "BARGE_IN_BUDGET_MS",
    "NO_AEC_WARNING",
    "SPEECH_CONFIRM_MS",
    "BargeInAvailability",
    "BargeInStop",
    "SpeechGate",
]
