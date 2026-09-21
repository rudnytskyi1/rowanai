"""Local addressing policy. Identifying a voice does not identify its addressee."""
from __future__ import annotations

import math


def followup_seconds(mode: str, configured: float, requested: float = 0) -> float:
    # Fail closed for missing/unknown modes. Server hints cannot override this.
    if mode != "window":
        return 0.0
    values = [float(v) for v in (configured, requested)]
    return min(30.0, max([v for v in values if math.isfinite(v)] + [0.0]))
