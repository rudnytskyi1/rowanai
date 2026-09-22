"""Detection profiles and their startup measurement (ТЗ F-312).

"Ускорение клиента. Экспорт YOLO11x в TensorRT (FP16) для 3060 Ti; профиль для
слабых ПК: YOLO11s и треки 10 FPS. Автовыбор профиля по измеренной задержке при
старте клиента."

Three pieces, deliberately separate from the camera threads so all of them can be
checked without a GPU, a camera or Ultralytics:

* :func:`plan_profiles` turns ``client.camera.profiles`` into the ordered list of
  profiles to try, and reports the ones that are skipped BEFORE measuring — a
  TensorRT engine is built on this machine by ``scripts/export_tensorrt.py``, so
  a PC that never ran the export must not try to load one;
* :func:`measure_ms` times real inferences (warm-up excluded, median of the
  rest) — it never estimates a latency from a model name;
* :func:`choose_profile` walks the profiles strongest-first and returns the first
  one whose measured latency fits its budget, or the fastest measured one when
  nothing fits. A profile whose measurement raised is skipped with the reason,
  never counted as fast.
"""
from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Inferences timed per profile (ТЗ F-312: "по измеренной задержке").
MEASURE_FRAMES = 3
#: Untimed inferences before measuring: the first call of a model loads kernels.
WARMUP_FRAMES = 1
#: ``.engine`` files are built locally; everything else may be downloaded.
ENGINE_SUFFIX = ".engine"


@dataclass(frozen=True)
class Profile:
    """One named detection profile of the client."""

    name: str
    model: str
    fps: float
    half: bool
    budget_ms: float


@dataclass(frozen=True)
class Attempt:
    """What happened to one profile while choosing."""

    name: str
    latency_ms: float | None
    reason: str


@dataclass(frozen=True)
class Choice:
    """The profile the client will run, and how it was chosen."""

    profile: Profile
    latency_ms: float | None
    reason: str
    attempts: tuple[Attempt, ...]

    def text(self) -> str:
        """One line for the log: profile, rate, measured latency, reason."""
        latency = "не измерили" if self.latency_ms is None else f"{self.latency_ms:.0f} мс/кадр"
        return (f"{self.profile.name}: {self.profile.model}, "
                f"{self.profile.fps:g} FPS, FP16={'да' if self.profile.half else 'нет'}, "
                f"{latency} ({self.reason})")


def plan_profiles(cfg_camera: Any, *, base_dir: Path | str | None = None) -> tuple[
        tuple[Profile, ...], tuple[Attempt, ...]]:
    """The profiles to try, plus the ones skipped before any measurement.

    The order is the config order: strongest first, as :func:`choose_profile`
    walks exactly this sequence. An entry without a model name is skipped, and so
    is a profile whose TensorRT engine has not been built on this machine yet.
    """
    entries = getattr(cfg_camera, "profiles", None) or {}
    if isinstance(entries, Mapping):
        iterator: Iterable[tuple[str, Any]] = entries.items()
    else:  # a plain list of profile objects: keep their own names
        iterator = ((str(getattr(item, "name", "")), item) for item in entries)
    usable: list[Profile] = []
    skipped: list[Attempt] = []
    for name, entry in iterator:
        model = str(getattr(entry, "model", "") or "").strip()
        label = str(name or model or "profile")
        if not model:
            skipped.append(Attempt(label, None, "у профиля нет модели"))
            continue
        profile = Profile(
            name=label,
            model=model,
            fps=max(0.0, float(getattr(entry, "fps", 10.0) or 0.0)),
            half=bool(getattr(entry, "half", True)),
            budget_ms=max(1.0, float(getattr(entry, "budget_ms", 60.0) or 60.0)),
        )
        if is_engine(profile) and not model_path(profile.model, base_dir).exists():
            # ТЗ F-312: движок TensorRT собирается на этой машине.
            skipped.append(Attempt(label, None, f"нет файла {profile.model} (собери export_tensorrt.py)"))
            continue
        usable.append(profile)
    return tuple(usable), tuple(skipped)


def is_engine(profile: Profile) -> bool:
    return profile.model.lower().endswith(ENGINE_SUFFIX)


def model_path(model: str, base_dir: Path | str | None = None) -> Path:
    """Where a profile's model file is looked for (engine check only)."""
    path = Path(model).expanduser()
    if path.is_absolute() or base_dir is None:
        return path
    return Path(base_dir) / path


def measure_ms(run: Callable[[], Any], *, frames: int = MEASURE_FRAMES,
               warmup: int = WARMUP_FRAMES) -> float:
    """Median milliseconds of ``frames`` real inferences; warm-up is not timed.

    A raise from ``run`` is the caller's problem: an inference that fails is not
    a fast inference, and a model that cannot run must be reported, not scored.
    """
    for _ in range(max(0, int(warmup))):
        run()
    costs: list[float] = []
    for _ in range(max(1, int(frames))):
        started = time.perf_counter()
        run()
        costs.append((time.perf_counter() - started) * 1000.0)
    return float(statistics.median(costs))


def choose_profile(candidates: Iterable[Profile], measure: Callable[[Profile], float]) -> Choice | None:
    """First profile that fits its budget when measured, else the fastest one.

    ``measure`` runs the real inference for one profile and returns milliseconds.
    Nothing here trusts a model name: a profile is only chosen after a number
    came back for it, and a profile whose measurement raised is skipped with the
    reason attached. ``None`` means nothing could be measured at all — the
    caller then keeps the profile its config names.
    """
    attempts: list[Attempt] = []
    measured: list[tuple[Profile, float]] = []
    for profile in candidates:
        try:
            latency = float(measure(profile))
        except Exception as exc:  # noqa: BLE001 - a broken profile is skipped, not scored
            attempts.append(Attempt(profile.name, None, f"замер не удался: {type(exc).__name__}: {exc}"))
            continue
        measured.append((profile, latency))
        if latency <= profile.budget_ms:
            attempts.append(Attempt(profile.name, latency, ""))
            return Choice(profile, latency, f"укладывается в бюджет {profile.budget_ms:g} мс",
                          tuple(attempts))
        attempts.append(Attempt(profile.name, latency, f"бюджет {profile.budget_ms:g} мс превышен"))
    if not measured:
        return None
    profile, latency = measured[-1]
    return Choice(profile, latency,
                  "ни один профиль не уложился в бюджет — остаётся самый слабый измеренный",
                  tuple(attempts))


__all__ = [
    "Attempt",
    "Choice",
    "ENGINE_SUFFIX",
    "MEASURE_FRAMES",
    "Profile",
    "WARMUP_FRAMES",
    "choose_profile",
    "is_engine",
    "measure_ms",
    "model_path",
    "plan_profiles",
]
