from __future__ import annotations
import asyncio
import sqlite3
import time
import uuid
from pathlib import Path
from agentos.core.agent import Agent
from agentos.sandbox.evaluation_scenario import EvaluationScenario
from agentos.sandbox.scorer import LLMJudgeScorer

_DB_PATH = Path(__file__).parent / "sim_results.db"


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sim_results (
            run_id TEXT,
            scenario_id TEXT,
            score REAL,
            latency_ms REAL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            PRIMARY KEY (run_id, scenario_id)
        )
    """)
    conn.commit()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    _init_db(conn)
    return conn


def _tokens_from_events(agent: Agent) -> tuple[int, int]:
    inp, out = 0, 0
    for e in agent.events:
        inp += e.data.get("prompt_tokens", 0)
        out += e.data.get("completion_tokens", 0)
    if inp == 0 and out == 0:
        total = sum(e.tokens_used for e in agent.events)
        half = total // 2
        inp, out = half, total - half
    return inp, out


class SimulationRunner:
    def __init__(self, db_path: str | Path | None = None):
        self._db = Path(db_path) if db_path else _DB_PATH
        conn = sqlite3.connect(self._db)
        _init_db(conn)
        conn.close()

    async def run_batch(
        self,
        scenarios: list[EvaluationScenario],
        agent: Agent,
        concurrency: int = 10,
    ) -> str:
        run_id = str(uuid.uuid4())
        scorer = LLMJudgeScorer()
        sem = asyncio.Semaphore(concurrency)

        async def run_one(s: EvaluationScenario) -> None:
            async with sem:
                a = Agent(
                    name=agent.config.name,
                    model=agent.config.model,
                    tools=agent.tools,
                    system_prompt=agent.config.system_prompt,
                    temperature=agent.config.temperature,
                )
                loop = asyncio.get_event_loop()
                start = time.perf_counter()
                try:
                    msg = await loop.run_in_executor(None, lambda: a.run(s.input))
                    actual = msg.content or ""
                except Exception:
                    actual = ""
                latency_ms = (time.perf_counter() - start) * 1000
                score = await scorer.score(s, actual)
                inp, out = _tokens_from_events(a)
                conn = sqlite3.connect(self._db)
                conn.execute(
                    "INSERT OR REPLACE INTO sim_results (run_id, scenario_id, score, latency_ms, input_tokens, output_tokens) VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, s.scenario_id, score, latency_ms, inp, out),
                )
                conn.commit()
                conn.close()

        await asyncio.gather(*[run_one(s) for s in scenarios])
        return run_id


def get_run_report(run_id: str, db_path: str | Path | None = None) -> dict | None:
    db = Path(db_path) if db_path else _DB_PATH
    conn = sqlite3.connect(db)
    _init_db(conn)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT scenario_id, score, latency_ms, input_tokens, output_tokens FROM sim_results WHERE run_id = ?",
        (run_id,),
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return None
    results = [
        {
            "scenario_id": r["scenario_id"],
            "score": r["score"],
            "latency_ms": r["latency_ms"],
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
        }
        for r in rows
    ]
    return {
        "run_id": run_id,
        "results": results,
        "avg_score": sum(r["score"] for r in results) / len(results),
        "total_latency_ms": sum(r["latency_ms"] for r in results),
        "total_input_tokens": sum(r["input_tokens"] for r in results),
        "total_output_tokens": sum(r["output_tokens"] for r in results),
    }
