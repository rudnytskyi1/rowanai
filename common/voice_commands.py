"""Explicit voice controls with bounded recovery for conversational hesitation."""
import re

SILENCE_PHRASES = frozenset({
    'shut up', 'stop talking', 'stop speaking', 'be quiet',
    'that s enough', 'that is enough',
    'замолчи', 'молчи', 'помолчи', 'заткнись', 'хватит говорить',
})

WAKE_NAMES = frozenset({'rowan', 'roan', 'rowen', 'роуэн', 'роуен', 'роуан', 'роан', 'рован'})
ROWAN_AI_PHRASES = ('rowan ai', 'rowan a i', 'roan ai', 'roan a i', 'rowen ai', 'rowen a i', 'rowanai')
# Command parsers accept both current and legacy address forms. This pattern
# does not authorize waking: has_wake_prefix uses only the configured phrases.
WAKE_ADDRESS_PATTERN = '(?:' + '|'.join(re.escape(name) for name in
    sorted(set(ROWAN_AI_PHRASES) | WAKE_NAMES, key=len, reverse=True)) + ')'
_QUOTED_TEXT = re.compile(
    r'"[^"\n]*"|“[^”\n]*”|«[^»\n]*»|(?<!\w)\'[^\'\n]*\'(?!\w)|‘[^’\n]*’'
)
# A small grammar, not arbitrary speech preceding/following a command. In
# particular, reported speech, negation and a continuation such as "is rude"
# cannot match. Spaces are supplied by normalize() before this is used.
_INTERJECTION = (
    r'(?:are you kidding me|are you kidding|hey|hi|hello|okay|ok|well|um|uh|'
    r'bro|dude|wait|no|oh|what|why|please|эй|ну|эм|так|пожалуйста)'
)
_INTERJECTIONS = re.compile(rf'(?:{_INTERJECTION} )*{_INTERJECTION}')
_WAKE_NAME = WAKE_ADDRESS_PATTERN
_SILENCE = '(?:' + '|'.join(re.escape(phrase) for phrase in sorted(SILENCE_PHRASES)) + ')'
_POLITE = r'(?:(?:please|пожалуйста) )?'
_POLITE_END = r'(?: (?:please|пожалуйста))?'
_ADDRESSED_SILENCE = rf'{_WAKE_NAME} {_POLITE}{_SILENCE}{_POLITE_END}'
_RECOVERED_SILENCE = re.compile(
    rf'(?:{_INTERJECTION} )*{_ADDRESSED_SILENCE}'
    rf'(?: (?:{_INTERJECTION}|{_ADDRESSED_SILENCE}))*'
)


def normalize(text):
    return ' '.join(re.sub(r'[^\w\s]', ' ', str(text).casefold()).split())


def is_silence_command(text):
    # Keep quoted commands as a nonmatching token instead of turning a quote
    # into a real command by stripping punctuation. Apostrophes in "that's"
    # and "don't" are not quote delimiters.
    text = normalize(_QUOTED_TEXT.sub(' quoted speech ', str(text)))
    if re.fullmatch(rf'{_POLITE}{_SILENCE}{_POLITE_END}', text):
        return True
    return _RECOVERED_SILENCE.fullmatch(text) is not None


def mentions_silence_command(text):
    text = ' ' + normalize(text) + ' '
    return any(' ' + phrase + ' ' in text for phrase in SILENCE_PHRASES)


def has_wake_prefix(text: str, phrases=()) -> bool:
    """Confirm a local wake trigger in STT without treating 'bro' as Rowan."""
    text = _QUOTED_TEXT.sub(' quoted speech ', str(text)).casefold().replace('’', "'")
    # Contractions are one spoken word: "But why don't do it Rowan" must not
    # lose its wake name merely because an apostrophe created an extra token.
    word_pattern = r"[^\W_]+(?:'[^\W_]+)*"
    words = re.findall(word_pattern, text, re.UNICODE)[:12]
    # An explicit configured phrase list is authoritative. Adding bare names
    # here would make "Rowan AI" silently fall back to the old "Rowan" trigger.
    aliases = {tuple(re.findall(word_pattern, str(phrase).casefold().replace('’', "'")))
               for phrase in phrases if phrase}
    if not aliases:
        aliases = {(name,) for name in WAKE_NAMES}
    for start in range(min(len(words), 10)):
        if start >= 6 and not _INTERJECTIONS.fullmatch(' '.join(words[:start])):
            continue
        if any(alias and tuple(words[start:start + len(alias)]) == alias for alias in aliases):
            return True
    return False
