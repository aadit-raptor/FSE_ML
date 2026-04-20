"""
Learns user preferences from session history.
Stores in local SQLite database.
No external dependencies beyond sqlite3 (built into Python).
"""

import sqlite3
import json
import os
import hashlib
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any

BASE     = os.path.dirname(__file__)
DB_PATH  = os.path.join(BASE, 'user_prefs.db')

# Fields we track and their default values
TRACKED_FIELDS = {
    'd_growth':       5.0,
    'd_gross_margin': 40.0,
    'd_opex':         18.0,
    'd_tax':          25.0,
    'd_da':            4.0,
    'd_capex':         4.0,
    'd_nwc':           1.0,
    'd_base_rate':     6.5,
    'd_mezz_spread':   4.0,
    'd_debt_pct':     60.0,
    'd_senior_pct':   70.0,
    'd_entry_mult':   10.0,
    'd_exit_mult':    11.0,
    'd_hold':          5.0,
    'mc_growth_mean':  5.0,
    'mc_growth_std':   3.0,
    'mc_exit_mean':   10.0,
    'mc_exit_std':     1.5,
    'mc_rate_mean':    6.5,
    'mc_gm_mean':     40.0,
}


def _get_user_id() -> str:
    """
    Generate a stable anonymous user ID from machine characteristics.
    No personal data stored — just a stable hash for session continuity.
    """
    import platform
    machine_id = platform.node() + platform.machine() + platform.processor()
    return hashlib.sha256(machine_id.encode()).hexdigest()[:16]


def _init_db():
    """Initialize SQLite database if not exists."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            field_name  TEXT NOT NULL,
            value       REAL NOT NULL,
            default_val REAL NOT NULL,
            changed     INTEGER NOT NULL,  -- 1 if user changed from default
            irr_result  REAL,              -- IRR from this session (if run)
            sector      TEXT
        )
    ''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_user ON sessions(user_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_field ON sessions(field_name)')
    conn.commit()
    conn.close()


def log_session(session_state, irr_result: Optional[float] = None,
                sector: Optional[str] = None):
    """
    Log current session state to database.
    Call this when user clicks "Run deal model" or "Run simulation".
    """
    _init_db()
    user_id   = _get_user_id()
    timestamp = datetime.now().isoformat()
    conn      = sqlite3.connect(DB_PATH)
    c         = conn.cursor()

    for field, default_val in TRACKED_FIELDS.items():
        current_val = session_state.get(field, default_val)
        try:
            current_float = float(current_val)
        except (TypeError, ValueError):
            continue

        changed = int(abs(current_float - default_val) > 0.01)

        c.execute('''
            INSERT INTO sessions
            (user_id, timestamp, field_name, value, default_val, changed, irr_result, sector)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, timestamp, field, current_float,
              default_val, changed, irr_result, sector))

    conn.commit()
    conn.close()


def get_personalized_defaults(min_sessions: int = 3) -> Optional[Dict[str, float]]:
    """
    Get personalized default values for this user.
    Returns None if insufficient history (< min_sessions runs).
    Returns dict of {field_name: preferred_value} otherwise.
    """
    _init_db()
    user_id = _get_user_id()
    conn    = sqlite3.connect(DB_PATH)
    c       = conn.cursor()

    # Count sessions for this user
    c.execute('''
        SELECT COUNT(DISTINCT timestamp) FROM sessions
        WHERE user_id = ? AND changed = 1
    ''', (user_id,))
    n_sessions = c.fetchone()[0]

    if n_sessions < min_sessions:
        conn.close()
        return None

    # For each field, compute median of user's non-default values
    personalized = {}
    for field in TRACKED_FIELDS:
        c.execute('''
            SELECT value FROM sessions
            WHERE user_id = ? AND field_name = ? AND changed = 1
            ORDER BY timestamp DESC LIMIT 20
        ''', (user_id, field))
        rows = c.fetchall()

        if len(rows) >= 2:
            values = [r[0] for r in rows]
            # Use median to be robust to outliers
            personalized[field] = float(np.median(values))

    conn.close()
    return personalized if personalized else None


def get_user_stats() -> dict:
    """Get summary statistics about user's usage patterns."""
    _init_db()
    user_id = _get_user_id()
    conn    = sqlite3.connect(DB_PATH)
    c       = conn.cursor()

    c.execute('''
        SELECT COUNT(DISTINCT timestamp), MIN(timestamp), MAX(timestamp)
        FROM sessions WHERE user_id = ?
    ''', (user_id,))
    row = c.fetchone()
    n_sessions = row[0] if row[0] else 0
    first_use  = row[1] if row[1] else 'N/A'
    last_use   = row[2] if row[2] else 'N/A'

    c.execute('''
        SELECT field_name, COUNT(*) as n
        FROM sessions WHERE user_id = ? AND changed = 1
        GROUP BY field_name ORDER BY n DESC LIMIT 5
    ''', (user_id,))
    most_changed = c.fetchall()

    conn.close()
    return {
        'n_sessions':    n_sessions,
        'first_use':     first_use,
        'last_use':      last_use,
        'most_changed':  most_changed,
    }