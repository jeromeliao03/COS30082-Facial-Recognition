"""
Temporal identity tracker: gates attendance on consistent per-slot name votes.
Innovative feature by Aston Lynch
"""

import time
import yaml

with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)["tracker"]

WINDOW = _cfg["window_frames"]
MIN_RATIO = _cfg["min_ratio"]
UNKNOWN_RATIO = _cfg["unknown_ratio"]
STALE_AFTER = _cfg["stale_after"]

# slot index -> {"votes": [], "confirmed": bool, "top_name": str|None, "last_seen": float}
# confirmed with top_name=None means "confirmed unknown" — stop collecting
_slots = {}


def observe(slot, name):
    """Record a name vote for a face slot, resetting if the slot has gone stale."""
    now = time.monotonic()
    buf = _slots.get(slot)

    if buf is None or (now - buf["last_seen"]) > STALE_AFTER:
        buf = {"votes": [], "confirmed": False, "top_name": None, "last_seen": now}
        _slots[slot] = buf

    buf["last_seen"] = now

    if buf["confirmed"]:
        return

    buf["votes"].append(name)
    if len(buf["votes"]) > WINDOW:
        buf["votes"].pop(0)


def query(slot):
    """Return tracking state for a slot as {state, name, progress}."""
    buf = _slots.get(slot)
    if buf is None:
        return {"state": None, "name": None, "progress": (0, WINDOW)}

    if buf["confirmed"]:
        # top_name=None means confirmed unknown — surface as state="unknown"
        if buf["top_name"] is None:
            return {"state": "unknown", "name": None, "progress": (WINDOW, WINDOW)}
        return {"state": "confirmed", "name": buf["top_name"], "progress": (WINDOW, WINDOW)}

    votes = buf["votes"]
    if not votes:
        return {"state": None, "name": None, "progress": (0, WINDOW)}

    none_count = votes.count(None)

    # Confirm unknown: once enough of the window is None, stop trying
    if none_count / WINDOW >= UNKNOWN_RATIO:
        buf["confirmed"] = True
        buf["top_name"] = None
        return {"state": "unknown", "name": None, "progress": (WINDOW, WINDOW)}

    named_votes = [v for v in votes if v is not None]
    if not named_votes:
        return {"state": "collecting", "name": None, "progress": (len(votes), WINDOW)}

    top = max(set(named_votes), key=named_votes.count)

    if len(votes) >= WINDOW and named_votes.count(top) / WINDOW >= MIN_RATIO:
        buf["confirmed"] = True
        buf["top_name"] = top
        return {"state": "confirmed", "name": top, "progress": (WINDOW, WINDOW)}

    return {"state": "collecting", "name": top, "progress": (len(votes), WINDOW)}


def clear_all():
    """Wipe all slot buffers when no faces are detected."""
    _slots.clear()
