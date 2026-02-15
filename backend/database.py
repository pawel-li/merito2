import sqlite3
import os
import json
from pathlib import Path

DB_PATH = Path("/app/data/app.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            filename TEXT NOT NULL,
            rows INTEGER NOT NULL,
            columns TEXT NOT NULL,
            target_column TEXT,
            model_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS csv_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            data TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            rule_id INTEGER NOT NULL,
            conditions TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            coverage REAL NOT NULL,
            precision_val REAL NOT NULL,
            support INTEGER NOT NULL,
            quality REAL NOT NULL,
            p_value REAL NOT NULL,
            class_distribution TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS confusion_matrices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            matrix TEXT NOT NULL,
            accuracy REAL NOT NULL,
            classes TEXT NOT NULL,
            extra_metrics TEXT,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
    """)
    conn.commit()
    conn.close()


def save_project(project_id: str, name: str, filename: str, rows: int, columns: list):
    conn = get_db()
    conn.execute(
        "INSERT INTO projects (id, name, filename, rows, columns) VALUES (?, ?, ?, ?, ?)",
        (project_id, name, filename, rows, json.dumps(columns))
    )
    conn.commit()
    conn.close()


def get_project(project_id: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    conn.close()
    if row:
        return {
            "id": row["id"],
            "name": row["name"],
            "filename": row["filename"],
            "rows": row["rows"],
            "columns": json.loads(row["columns"]),
            "target_column": row["target_column"],
            "model_id": row["model_id"],
            "created_at": row["created_at"],
        }
    return None


def list_projects():
    conn = get_db()
    rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "filename": r["filename"],
            "rows": r["rows"],
            "columns": json.loads(r["columns"]),
            "target_column": r["target_column"],
            "model_id": r["model_id"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def update_project(project_id: str, **kwargs):
    conn = get_db()
    for key, value in kwargs.items():
        conn.execute(f"UPDATE projects SET {key} = ? WHERE id = ?", (value, project_id))
    conn.commit()
    conn.close()


def save_csv_rows(project_id: str, df):
    conn = get_db()
    rows = []
    for idx, row in df.iterrows():
        rows.append((project_id, idx, json.dumps(row.to_dict(), default=str)))
    conn.executemany(
        "INSERT INTO csv_data (project_id, row_index, data) VALUES (?, ?, ?)",
        rows
    )
    conn.commit()
    conn.close()


def get_csv_data(project_id: str):
    import pandas as pd
    conn = get_db()
    rows = conn.execute(
        "SELECT data FROM csv_data WHERE project_id = ? ORDER BY row_index",
        (project_id,)
    ).fetchall()
    conn.close()
    if not rows:
        return None
    data = [json.loads(r["data"]) for r in rows]
    return pd.DataFrame(data)


def save_rules(project_id: str, rules_data: list):
    conn = get_db()
    # Clear existing rules
    conn.execute("DELETE FROM rules WHERE project_id = ?", (project_id,))
    for rule in rules_data:
        conn.execute(
            """INSERT INTO rules (project_id, rule_id, conditions, predicted_class,
               coverage, precision_val, support, quality, p_value, class_distribution)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                project_id,
                rule["rule_id"],
                rule["conditions"],
                rule["predicted_class"],
                rule["coverage"],
                rule["precision"],
                rule["support"],
                rule["quality"],
                rule["p_value"],
                json.dumps(rule["class_distribution"])
            )
        )
    conn.commit()
    conn.close()


def get_rules(project_id: str):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM rules WHERE project_id = ? ORDER BY rule_id",
        (project_id,)
    ).fetchall()
    conn.close()
    return [
        {
            "rule_id": r["rule_id"],
            "conditions": r["conditions"],
            "predicted_class": r["predicted_class"],
            "coverage": r["coverage"],
            "precision": r["precision_val"],
            "support": r["support"],
            "quality": r["quality"],
            "p_value": r["p_value"],
            "class_distribution": json.loads(r["class_distribution"]),
        }
        for r in rows
    ]


def add_rule(project_id: str, rule_data: dict):
    conn = get_db()
    # Get next rule_id
    row = conn.execute(
        "SELECT COALESCE(MAX(rule_id), 0) + 1 as next_id FROM rules WHERE project_id = ?",
        (project_id,)
    ).fetchone()
    next_id = row["next_id"]
    conn.execute(
        """INSERT INTO rules (project_id, rule_id, conditions, predicted_class,
           coverage, precision_val, support, quality, p_value, class_distribution)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            project_id,
            next_id,
            rule_data["conditions"],
            rule_data["predicted_class"],
            rule_data.get("coverage", 0.0),
            rule_data.get("precision", 0.0),
            rule_data.get("support", 0),
            rule_data.get("quality", 0.0),
            rule_data.get("p_value", 1.0),
            json.dumps(rule_data.get("class_distribution", {})),
        )
    )
    conn.commit()
    conn.close()
    return next_id


def delete_rule(project_id: str, rule_id: int):
    conn = get_db()
    cursor = conn.execute(
        "DELETE FROM rules WHERE project_id = ? AND rule_id = ?",
        (project_id, rule_id)
    )
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted > 0


def save_confusion_matrix(project_id: str, matrix, accuracy, classes, extra_metrics=None):
    conn = get_db()
    conn.execute("DELETE FROM confusion_matrices WHERE project_id = ?", (project_id,))
    conn.execute(
        """INSERT INTO confusion_matrices (project_id, matrix, accuracy, classes, extra_metrics)
           VALUES (?, ?, ?, ?, ?)""",
        (
            project_id,
            json.dumps(matrix),
            accuracy,
            json.dumps(classes),
            json.dumps(extra_metrics) if extra_metrics else None,
        )
    )
    conn.commit()
    conn.close()


def update_rule_metrics(project_id: str, rule_id: int, coverage: float,
                        precision_val: float, support: int, quality: float,
                        p_value: float, class_distribution: dict):
    conn = get_db()
    conn.execute(
        """UPDATE rules SET coverage = ?, precision_val = ?, support = ?,
           quality = ?, p_value = ?, class_distribution = ?
           WHERE project_id = ? AND rule_id = ?""",
        (coverage, precision_val, support, quality, p_value,
         json.dumps(class_distribution), project_id, rule_id)
    )
    conn.commit()
    conn.close()


def get_confusion_matrix(project_id: str):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM confusion_matrices WHERE project_id = ?", (project_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    result = {
        "matrix": json.loads(row["matrix"]),
        "accuracy": row["accuracy"],
        "classes": json.loads(row["classes"]),
    }
    if row["extra_metrics"]:
        result.update(json.loads(row["extra_metrics"]))
    return result
