"""
utils/database.py
SQLite-based storage for prediction history.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "predictions.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def init_db():
    """Create the predictions table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            patient     TEXT,
            prediction  TEXT,
            benign_prob REAL,
            mal_prob    REAL,
            risk_score  REAL,
            features    TEXT,
            timestamp   TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(patient: str, prediction: str, benign_prob: float,
                    mal_prob: float, features: dict):
    """Insert a single prediction record."""
    init_db()
    import json
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (patient, prediction, benign_prob, mal_prob, risk_score, features, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            patient or "Anonymous",
            prediction,
            round(benign_prob, 4),
            round(mal_prob, 4),
            round(mal_prob, 4),
            json.dumps({k: round(float(v), 4) for k, v in features.items()}),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    )
    conn.commit()
    conn.close()


def get_history(limit: int = 100) -> pd.DataFrame:
    """Retrieve the last `limit` prediction records."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"SELECT * FROM predictions ORDER BY id DESC LIMIT {limit}", conn
    )
    conn.close()
    return df


def clear_history():
    """Delete all records from the predictions table."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()


def get_stats() -> dict:
    """Return high-level stats about stored predictions."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM predictions")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Benign'")
    benign = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Malignant'")
    malignant = cur.fetchone()[0]
    conn.close()
    return {"total": total, "benign": benign, "malignant": malignant}