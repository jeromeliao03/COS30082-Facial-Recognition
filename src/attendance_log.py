import csv
import os 
from datetime import datetime, timedelta

LOG_PATH = 'output/attendance_log.csv'
MAX_SPOOF_ATTEMPTS = 3
LOCKOUT_DURATION = 60

#State
_consecutive_spoofs = 0
_lockout_until = None
_logged_today = set() # Track identities logged today to prevent duplicates

os.makedirs("output", exist_ok=True)
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['timestamp','identity', 'liveness','emotion', 'action']
        )

# Check if system is currently locked out due to spoofing attempts
def is_locked():
    global _lockout_until
    if _lockout_until is None:
        return False
    if datetime.now() < _lockout_until:
        return True
    _lockout_until = None
    return False

def get_remaining_seconds():
    global _lockout_until
    if _lockout_until is None:
        return 0
    remaining = (_lockout_until - datetime.now()).total_seconds()
    return max(0, int(remaining))

# Check if identity has already been logged today
def already_logged(identity):
    return identity in _logged_today


def evict(identity):
    """Remove identity from today's log set (call when identity is deleted)."""
    _logged_today.discard(identity)


def process(identity, liveness, emotion):
    global _consecutive_spoofs, _lockout_until, _logged_today

    locked = is_locked()

    if locked:
        action = "System locked"

    elif not liveness:
        # spoof — always increment, never reset by real faces
        _consecutive_spoofs += 1
        if _consecutive_spoofs >= MAX_SPOOF_ATTEMPTS:
            _lockout_until      = datetime.now() + timedelta(seconds=LOCKOUT_DURATION)
            action              = f"ALERT — system locked {LOCKOUT_DURATION}s"
            _consecutive_spoofs = 0
        else:
            action = f"Spoof detected ({_consecutive_spoofs}/{MAX_SPOOF_ATTEMPTS})"

    elif identity and identity in _logged_today:
        return "Already logged"

    else:
        # real face — only reset counter if NO spoof was detected this frame
        # counter reset is handled separately, not here
        action = "Access granted"
        if identity:
            _logged_today.add(identity)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_PATH, 'a', newline='') as f:
        csv.writer(f).writerow([
            timestamp,
            'Unknown' if not liveness else (identity or 'Unknown'),
            'REAL' if liveness else 'SPOOF',
            emotion or '-',
            action
        ])

    return action