from __future__ import annotations
from pathlib import Path
import sqlite3
from agentos.sandbox.simulation_runner import _DB_PATH, _init_db


class ComparisonReport:
    @staticmethod
    def generate(
        run_id_a: str, run_id_b: str, db_path: str | Path | None = None
    ) -> dict:
        db = Path(db_path) if db_path else _DB_PATH
        conn = sqlite3.connect(db)
        _init_db(conn)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT a.scenario_id, a.score as score_a, b.score as score_b
            FROM sim_results a
            JOIN sim_results b ON a.scenario_id = b.scenario_id
            WHERE a.run_id = ? AND b.run_id = ?
            """,
            (run_id_a, run_id_b),
        )
        rows = cur.fetchall()
        conn.close()
        deltas = []
        for r in rows:
            s_id = r["scenario_id"]
            sa = float(r["score_a"])
            sb = float(r["score_b"])
            deltas.append(
                {
                    "scenario_id": s_id,
                    "score_a": sa,
                    "score_b": sb,
                    "delta": sb - sa,
                }
            )
        return {
            "run_id_a": run_id_a,
            "run_id_b": run_id_b,
            "scenarios": deltas,
            "avg_delta": sum(d["delta"] for d in deltas) / len(deltas)
            if deltas
            else 0.0,
        }
