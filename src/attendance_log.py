import csv
import os 
from datetime import datetime, timedelta

LOG_PATH = 'output/attendance_log.csv'
MAX_SPOOF_ATTEMPTS = 3
LOCKOUT_DURATION = 60

#State
_consecutive_spoof = 0
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


def process(identity, liveness, emotion):
    """
    Call after every inference result.
    - Spoof attempts are ALWAYS logged
    - Real face attendance is only logged ONCE per session per person
    """
    global _consecutive_spoof, _lockout_until, _logged_today

    locked = is_locked()

    if locked:
        action = "System locked"

    elif not liveness:
        # always log spoof attempts
        _consecutive_spoof += 1
        if _consecutive_spoof >= MAX_SPOOF_ATTEMPTS:
            _lockout_until      = datetime.now() + timedelta(seconds=LOCKOUT_DURATION)
            action              = f"ALERT — system locked {LOCKOUT_DURATION}s"
            _consecutive_spoof = 0
        else:
            action = f"Spoof detected ({_consecutive_spoof}/{MAX_SPOOF_ATTEMPTS})"

    elif identity and identity in _logged_today:
        # real face but already logged — skip
        return "Already logged"

    else:
        # real face, not yet logged
        _consecutive_spoof = 0
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