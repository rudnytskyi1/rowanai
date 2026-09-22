"""ULID identifiers for utterances (ТЗ section 4.5).

Every utterance carries one identifier from the moment the client notices that
somebody started talking: it is stamped on the client's frames, echoed back by
the hub on the frames of that turn, written into the logs and the database and
counted in the metrics. A ULID is used rather than a UUID because it is
lexicographically sortable by creation time and stays readable in a log line.

The layout is the usual one: 48 bits of milliseconds since the Unix epoch,
80 bits of randomness, rendered as 26 Crockford base32 characters. Within one
millisecond the random part is incremented instead of redrawn, so two
utterances from the same client never collide and never sort backwards.
"""
from __future__ import annotations

import secrets
import threading
import time

#: Crockford base32: no ``I``, ``L``, ``O`` or ``U``, so a hand-copied id
#: cannot be misread as a digit or an accidental word.
ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
#: Characters in a rendered ULID (10 timestamp + 16 randomness).
ULID_LENGTH = 26
#: Characters used by the millisecond timestamp.
ULID_TIMESTAMP_CHARS = 10
#: Characters used by the randomness (16 * 5 = 80 bits).
ULID_RANDOM_CHARS = 16

_TIMESTAMP_BITS = 48
_RANDOM_BITS = 80
_MAX_TIMESTAMP_MS = (1 << _TIMESTAMP_BITS) - 1
_MAX_RANDOM = (1 << _RANDOM_BITS) - 1
_DECODE = {char: value for value, char in enumerate(ULID_ALPHABET)}

_lock = threading.Lock()
_last_ms = 0
_last_random = 0


def _now_ms() -> int:
    return int(time.time() * 1000)


def _encode(value: int, length: int) -> str:
    chars: list[str] = []
    for _ in range(length):
        value, remainder = divmod(value, 32)
        chars.append(ULID_ALPHABET[remainder])
    return "".join(reversed(chars))


def _decode(text: str) -> int:
    value = 0
    for char in text:
        value = value * 32 + _DECODE[char]
    return value


def new_ulid(timestamp_ms: int | None = None) -> str:
    """A fresh, time-sortable identifier.

    ``timestamp_ms`` exists for tests and for replaying an archived turn; the
    live pipeline always uses the clock. Ids created within one millisecond
    share that millisecond but never collide and never sort backwards: the
    random part is incremented instead of redrawn. On the live path a clock
    that steps backwards is ignored, so an id can never be older than the one
    handed out before it.
    """
    global _last_ms, _last_random
    moment = int(timestamp_ms) if timestamp_ms is not None else _now_ms()
    if not 0 <= moment <= _MAX_TIMESTAMP_MS:
        raise ValueError(f"timestamp out of range for a ULID: {moment}")
    with _lock:
        if timestamp_ms is None and moment < _last_ms:
            # A clock that jumped back must not rewind the id sequence.
            moment = _last_ms
        if moment == _last_ms:
            _last_random += 1
            if _last_random > _MAX_RANDOM:
                _last_ms += 1
                moment = _last_ms
                _last_random = secrets.randbits(_RANDOM_BITS)
        else:
            _last_ms = moment
            _last_random = secrets.randbits(_RANDOM_BITS)
        return _encode(moment, ULID_TIMESTAMP_CHARS) + _encode(_last_random, ULID_RANDOM_CHARS)


def is_ulid(value: object) -> bool:
    """True when ``value`` is a well-formed ULID (case-insensitive)."""
    if not isinstance(value, str) or len(value) != ULID_LENGTH:
        return False
    return all(char in _DECODE for char in value.upper())


def ulid_timestamp_ms(value: str) -> int:
    """The creation time of ``value`` in milliseconds since the Unix epoch."""
    if not is_ulid(value):
        raise ValueError(f"not a ULID: {value!r}")
    return _decode(value.upper()[:ULID_TIMESTAMP_CHARS])


__all__ = [
    "ULID_ALPHABET",
    "ULID_LENGTH",
    "ULID_RANDOM_CHARS",
    "ULID_TIMESTAMP_CHARS",
    "is_ulid",
    "new_ulid",
    "ulid_timestamp_ms",
]
