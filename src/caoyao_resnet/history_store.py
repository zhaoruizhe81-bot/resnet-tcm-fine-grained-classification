from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .utils import ensure_dir


def default_history_db_path(output_root: str | Path = "outputs") -> Path:
    return Path(output_root) / "webapp" / "app_history.db"


def init_history_db(db_path: str | Path) -> Path:
    target = Path(db_path)
    ensure_dir(target.parent)
    with sqlite3.connect(target) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS recognition_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                record_type TEXT NOT NULL,
                input_name TEXT NOT NULL,
                checkpoint_path TEXT NOT NULL,
                model_name TEXT NOT NULL,
                summary TEXT NOT NULL,
                output_path TEXT,
                duration_seconds REAL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_recognition_history_created_at "
            "ON recognition_history(created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_recognition_history_record_type "
            "ON recognition_history(record_type)"
        )
        connection.commit()
    return target


def insert_history_record(
    db_path: str | Path,
    *,
    created_at: str,
    record_type: str,
    input_name: str,
    checkpoint_path: str,
    model_name: str,
    summary: str,
    output_path: str | None,
    duration_seconds: float | None,
    metadata: dict[str, Any],
) -> int:
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO recognition_history (
                created_at,
                record_type,
                input_name,
                checkpoint_path,
                model_name,
                summary,
                output_path,
                duration_seconds,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                record_type,
                input_name,
                checkpoint_path,
                model_name,
                summary,
                output_path,
                duration_seconds,
                json.dumps(metadata, ensure_ascii=False, indent=2),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def fetch_history_records(
    db_path: str | Path,
    *,
    limit: int = 200,
    record_type: str | None = None,
) -> list[dict[str, Any]]:
    query = """
        SELECT
            id,
            created_at,
            record_type,
            input_name,
            checkpoint_path,
            model_name,
            summary,
            output_path,
            duration_seconds,
            metadata_json
        FROM recognition_history
    """
    params: list[Any] = []
    if record_type:
        query += " WHERE record_type = ?"
        params.append(record_type)
    query += " ORDER BY created_at DESC, id DESC LIMIT ?"
    params.append(limit)

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, params).fetchall()

    records: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        record["metadata"] = json.loads(record.pop("metadata_json"))
        records.append(record)
    return records


def fetch_history_stats(db_path: str | Path) -> dict[str, Any]:
    with sqlite3.connect(db_path) as connection:
        total = connection.execute("SELECT COUNT(*) FROM recognition_history").fetchone()[0]
        grouped_rows = connection.execute(
            """
            SELECT record_type, COUNT(*) AS count
            FROM recognition_history
            GROUP BY record_type
            ORDER BY record_type
            """
        ).fetchall()

    by_type = {row[0]: row[1] for row in grouped_rows}
    return {
        "total_records": total,
        "by_type": by_type,
    }
