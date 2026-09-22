"""Local stop-command spotting while Rowan works or speaks; no cloud audio."""
import json
import logging
from pathlib import Path

from common.voice_commands import has_wake_prefix, is_silence_command, normalize

log = logging.getLogger(__name__)


class ConfirmedWakeDetector:
    """Check a wake candidate against common competing words before waking.

    A grammar containing only names maps ordinary words such as "bro" to
    "roan". Give the second decoder plausible alternatives, while keeping only
    configured wake phrases eligible to activate. An unrestricted small Vosk
    model often writes the real name as "road" or "rowing"; those words are
    never added as accepted wake aliases. No audio leaves this local check.
    """

    CANDIDATE_SECONDS = 1.5
    STABLE_SECONDS = 0.18
    # Vosk splits confidence between the configured homophones Rowan/Roan/
    # Rowen (recorded true wake: Rowen at 0.5). This threshold applies only to
    # a wake-name token; distractor words never become eligible at any score.
    MIN_CONFIDENCE = 0.45
    DISTRACTORS = ('bro', 'dude', 'hello', 'yeah', 'yes', 'no', 'wow', 'oh',
                   'thank you', 'what', 'why', 'sorry', 'okay', 'right', 'well')

    def __init__(self, wake):
        self._wake = wake
        self.sample_rate = wake.sample_rate
        self.phrases = wake.phrases
        grammar = json.dumps(list(dict.fromkeys([*self.phrases, *self.DISTRACTORS, '[unk]'])))
        self._plain = wake._vosk.KaldiRecognizer(wake._model, float(wake.sample_rate), grammar)
        self._plain.SetWords(True)
        if hasattr(self._plain, 'SetPartialWords'):
            self._plain.SetPartialWords(True)
        self._samples = 0
        self._candidate_at = None
        self._heard_since = None
        log.info('Wake confirmation ready: wake phrases compete with ordinary room speech')

    @staticmethod
    def _result(payload):
        try:
            value = json.loads(payload)
        except (ValueError, TypeError, AttributeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _confident_wake(self, result, field):
        text = result.get(field, '')
        if not isinstance(text, str) or not has_wake_prefix(text, self.phrases):
            return False
        words = result.get('result' if field == 'text' else 'partial_result')
        if not isinstance(words, list) or not words:
            return True  # Older Vosk: require stability before accepting a partial.
        # Vosk exposes one item per word: confidence must cover the ENTIRE
        # configured phrase, including AI, rather than a standalone name token.
        tokens = [normalize(word.get('word', '')) if isinstance(word, dict) else '' for word in words]
        for phrase in self.phrases:
            wanted = normalize(phrase).split()
            if not wanted:
                continue
            for start in range(len(tokens) - len(wanted) + 1):
                if tokens[start:start + len(wanted)] != wanted:
                    continue
                matches = words[start:start + len(wanted)]
                if all(type(word.get('conf')) in (int, float) and word['conf'] >= self.MIN_CONFIDENCE
                       for word in matches):
                    return True
        return False

    def reset(self):
        self._wake.reset()
        self._plain.Reset()
        self._samples = 0
        self._candidate_at = None
        self._heard_since = None

    def accept_frame(self, frame):
        if not frame:
            return False
        self._samples += len(frame) // 2
        now = self._samples / self.sample_rate
        if self._wake.accept_frame(frame):
            self._candidate_at = now
        try:
            final = self._plain.AcceptWaveform(frame)
            result = self._result(self._plain.Result() if final else self._plain.PartialResult())
            field = 'text' if final else 'partial'
            text = result.get(field, '')
        except Exception as exc:
            log.warning('Local wake confirmation failed (%s); candidate ignored', type(exc).__name__)
            self.reset()
            return False
        if not self._confident_wake(result, field):
            self._heard_since = None
        elif self._heard_since is None:
            self._heard_since = now
        if self._candidate_at is not None and now - self._candidate_at > self.CANDIDATE_SECONDS:
            log.debug('Unconfirmed local wake candidate ignored: %r', text)
            self._candidate_at = None
        confirmed = (self._candidate_at is not None and self._heard_since is not None
                     and (final or now - self._heard_since >= self.STABLE_SECONDS))
        if confirmed:
            log.info('Wake word confirmed locally: %r', text)
            self.reset()
            return True
        # Decoder endpoints differ. Keep the bounded candidate until expiry,
        # but a completed verification utterance cannot lend its words later.
        if final:
            self._heard_since = None
        return False


class SilenceDetector:
    def __init__(self, wake, russian_model=None):
        # Share the English acoustic model, but use an unrestricted decoder:
        # a tiny command grammar can turn "don't shut up" into "shut up".
        self._models = [wake._model]
        self._recognizers = [wake._vosk.KaldiRecognizer(wake._model, float(wake.sample_rate))]
        if russian_model and Path(russian_model).is_dir():
            model = wake._vosk.Model(str(russian_model))
            self._models.append(model)
            self._recognizers.append(wake._vosk.KaldiRecognizer(model, float(wake.sample_rate)))
        log.info('Local silence controls ready (%d language model(s))', len(self._models))

    def reset(self):
        for rec in self._recognizers:
            rec.Reset()

    def accept_phrase(self, frame):
        """The finalized local phrase, or ``None`` (F-117 uses the words too)."""
        for rec in self._recognizers:
            if rec.AcceptWaveform(frame):
                text = json.loads(rec.Result()).get('text', '')
                # Only finalized, complete utterances. A partial "shut up"
                # might continue as "is a rude thing to say".
                if text:
                    self.reset()
                    return str(text)
        return None

    def accept_frame(self, frame):
        """True when the room said a stop command (the pre-F-117 contract)."""
        text = self.accept_phrase(frame)
        return bool(text) and is_silence_command(text)
