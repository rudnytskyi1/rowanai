"""Треки людей на клиенте: id живёт, пока человек в поле зрения (ТЗ F-201).

BoT-SORT (through Ultralytics' ``model.track``) does the association; what this
module owns is the two things the ТЗ asks for on top of it:

* the tracker's memory is sized for the ТЗ's thirty seconds - a person who
  steps out of the frame and comes back is the SAME track, not a new arrival,
  and that is a property of ``track_buffer``, which is measured in frames and
  therefore has to be derived from the actual frame rate;
* the client reports TRACKS (id + box + confidence + since) instead of a bare
  person count, and says when a track was re-associated after a gap.

The registry is deliberately independent of the camera: it can be reasoned
about, and tested, without a GPU, a video stream or Ultralytics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: ТЗ F-201: "re-association до 30 с после потери".
REASSOCIATION_S = 30.0
#: BoT-SORT works in frames, so thirty seconds needs a frame budget.
MIN_TRACK_BUFFER = 30
#: A person who reappears after this long is reported as re-associated rather
#: than as a track that was simply still there.
RETURN_GAP_S = 1.0
#: RoomState on the hub reads at most 24 tracks; more than that is not a room.
MAX_TRACKS = 24


@dataclass(frozen=True)
class TrackDetection:
    """One track as the detector reported it in the current frame."""

    track_id: str
    bbox: tuple[float, float, float, float]
    conf: float = 0.0


@dataclass(frozen=True)
class TrackReport:
    """One track of the current frame, with its life story so far."""

    track_id: str
    bbox: tuple[float, float, float, float]
    conf: float
    since: float
    event: str
    gap_s: float = 0.0

    def as_wire(self) -> dict[str, Any]:
        """The F-201 ``tracks`` message entry (protocol ``Track`` fields)."""
        return {
            "track_id": self.track_id,
            "bbox": [float(value) for value in self.bbox],
            "conf": float(self.conf),
            "zone": "",
            "since": float(self.since),
        }


def tracker_settings(fps: Any, *, seconds: float = REASSOCIATION_S,
                     base: dict[str, Any] | None = None) -> dict[str, Any]:
    """The BoT-SORT parameters, with a buffer that covers ``seconds``.

    ``track_buffer`` is counted in FRAMES: twenty of them is a fifth of a
    second on a fast camera and a full second on a slow one, and neither is the
    thirty seconds F-201 promises. The buffer is therefore derived from the
    real frame rate, with the floor the tracker needs to work at all.
    """
    settings = dict(base or {})
    rate = float(fps or 0.0)
    if rate <= 0:
        rate = 10.0
    settings["track_buffer"] = max(MIN_TRACK_BUFFER, int(round(float(seconds) * rate)))
    return settings


def write_tracker_config(path: str | Path, fps: Any, *,
                         seconds: float = REASSOCIATION_S) -> Path:
    """Write the tracker's runtime config next to the client.

    The shipped ``room-tracker.yaml`` stays the template (it holds the
    thresholds the stand was tuned with); only the frame buffer depends on the
    machine, so only the buffer is recomputed - into its own file.
    """
    import yaml

    target = Path(path)
    template = target.with_name("room-tracker.yaml")
    base: dict[str, Any] = {}
    if template.is_file():
        try:
            loaded = yaml.safe_load(template.read_text(encoding="utf-8"))
            base = dict(loaded) if isinstance(loaded, dict) else {}
        except Exception as exc:  # noqa: BLE001 - a broken template must not stop the camera
            log.warning("Could not read %s (%s); using the built-in tracker settings", template, exc)
    settings = tracker_settings(fps, seconds=seconds, base=base)
    target.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    log.info("Tracker buffer covers %.0f s at %.1f fps (%d frames)",
             seconds, float(fps or 10.0), settings["track_buffer"])
    return target


class TrackRegistry:
    """The live tracks of one camera, with the ТЗ's thirty-second memory.

    ``observe`` is called once per processed frame with whatever the tracker
    found in that frame. A track that is missing from a frame is NOT forgotten:
    it stays in the memory (with the moment it was last seen) for
    ``seconds``, so a person who steps out and comes back keeps the same id.
    Reports say which of the three things happened - the track is new
    (``entered``), it came back after a gap (``returned``), or it was simply
    still there (``active``).
    """

    def __init__(self, *, seconds: float = REASSOCIATION_S,
                 max_tracks: int = MAX_TRACKS) -> None:
        self.seconds = float(seconds)
        self.max_tracks = int(max_tracks)
        #: track_id -> {"last_seen", "since", "bbox", "conf", "misses"}
        self._tracks: dict[str, dict[str, Any]] = {}

    def observe(self, detections: Any, *, now: float) -> list[TrackReport]:
        """Apply one frame; return a report per track seen in it."""
        reports: list[TrackReport] = []
        for detection in _detections(detections)[: self.max_tracks]:
            row = self._tracks.get(detection.track_id)
            if row is None:
                row = self._tracks[detection.track_id] = {
                    "since": float(now), "last_seen": float(now), "misses": 0,
                }
                event, gap = "entered", 0.0
            else:
                gap = max(0.0, float(now) - float(row["last_seen"]))
                event = "returned" if gap >= RETURN_GAP_S else "active"
                row["last_seen"] = float(now)
            row.update(bbox=detection.bbox, conf=float(detection.conf), misses=0)
            reports.append(TrackReport(track_id=detection.track_id, bbox=detection.bbox,
                                       conf=float(detection.conf), since=float(row["since"]),
                                       event=event, gap_s=gap))
        seen = {report.track_id for report in reports}
        for track_id, row in self._tracks.items():
            if track_id not in seen:
                row["misses"] = int(row.get("misses", 0)) + 1
        self._expire(now)
        return reports

    def active(self, *, now: float, max_age: float = RETURN_GAP_S) -> list[TrackDetection]:
        """The tracks that were in the last frames (``max_age`` seconds)."""
        self._expire(now)
        return [TrackDetection(track_id=track_id, bbox=tuple(row["bbox"]),
                               conf=float(row.get("conf") or 0.0))
                for track_id, row in self._tracks.items()
                if float(now) - float(row["last_seen"]) <= float(max_age)]

    def remembered(self, *, now: float) -> list[str]:
        """Every id still remembered (visible, or lost less than 30 s ago)."""
        self._expire(now)
        return list(self._tracks)

    def gap_s(self, track_id: str, *, now: float) -> float:
        row = self._tracks.get(str(track_id))
        return 0.0 if row is None else max(0.0, float(now) - float(row["last_seen"]))

    def _expire(self, now: float) -> None:
        for track_id, row in list(self._tracks.items()):
            if float(now) - float(row["last_seen"]) > self.seconds:
                del self._tracks[track_id]


def _detections(raw: Any) -> list[TrackDetection]:
    """Normalize whatever the caller has into :class:`TrackDetection` records."""
    items: list[TrackDetection] = []
    for entry in raw or ():
        if isinstance(entry, TrackDetection):
            items.append(entry)
            continue
        if isinstance(entry, dict):
            track_id = entry.get("track_id", entry.get("id"))
            box = entry.get("bbox", entry.get("box"))
            conf = entry.get("conf", 0.0)
        else:
            continue
        if track_id is None or box is None:
            continue
        try:
            values = [float(value) for value in box]
        except (TypeError, ValueError):
            continue
        if len(values) != 4:
            continue
        items.append(TrackDetection(track_id=str(track_id),
                                    bbox=(values[0], values[1], values[2], values[3]),
                                    conf=float(conf or 0.0)))
    return items


__all__ = [
    "MAX_TRACKS",
    "MIN_TRACK_BUFFER",
    "REASSOCIATION_S",
    "RETURN_GAP_S",
    "TrackDetection",
    "TrackRegistry",
    "TrackReport",
    "tracker_settings",
    "write_tracker_config",
]
