"""
DeFiScope — Pipeline Execution Logger
======================================
Author: RAVINJEET SINGH
Module: Logging & Observability

Provides structured logging for the multi-agent pipeline. Captures per-agent
execution time, LLM call counts, success/failure status, and produces JSON
execution summaries for debugging and performance analysis.

Usage:
    from logger import get_logger, AgentTimer

    logger = get_logger("OrchestratorAgent")
    with AgentTimer("ChainAgent") as t:
        result = chain_agent.run()
    logger.info(f"Chain done in {t.elapsed:.2f}s")
"""

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

# =========================================================================
# 1. Logger Configuration
# =========================================================================
LOG_FORMAT = "%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
LOG_DIR = Path("logs")


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger instance.
    Idempotent — repeated calls with the same name return the same logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# =========================================================================
# 2. Agent Execution Timer (context manager)
# =========================================================================
class AgentTimer:
    """
    Context manager for tracking agent execution time and outcomes.

    Example:
        with AgentTimer("ChainAgent") as t:
            result = run_chain_agent()
        print(t.elapsed, t.status)
    """

    def __init__(self, agent_name: str, llm_calls: int = 0):
        self.agent_name = agent_name
        self.llm_calls = llm_calls
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed: float = 0.0
        self.status: str = "pending"
        self.error_msg: str = ""
        self.logger = get_logger(agent_name)

    def __enter__(self) -> "AgentTimer":
        self.start_time = time.time()
        self.logger.info(f"[START] {self.agent_name} execution begins")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time

        if exc_type is None:
            self.status = "success"
            self.logger.info(
                f"[OK]    {self.agent_name} done in {self.elapsed:.2f}s "
                f"(LLM calls: {self.llm_calls})"
            )
        else:
            self.status = "failed"
            self.error_msg = str(exc_val)
            self.logger.error(
                f"[FAIL]  {self.agent_name} crashed after {self.elapsed:.2f}s: "
                f"{exc_type.__name__}: {self.error_msg}"
            )
        return False  # do not suppress exceptions

    def to_dict(self) -> dict[str, Any]:
        """Serialize timer state for execution summaries."""
        return {
            "agent": self.agent_name,
            "elapsed_seconds": round(self.elapsed, 3),
            "llm_calls": self.llm_calls,
            "status": self.status,
            "error": self.error_msg or None,
        }


# =========================================================================
# 3. Pipeline Execution Summary
# =========================================================================
class PipelineExecutionSummary:
    """Aggregates timing metadata from all agents in a single recommendation run."""

    def __init__(self, query: str = ""):
        self.query = query
        self.start_time = datetime.now()
        self.timers: list[AgentTimer] = []

    def add(self, timer: AgentTimer) -> None:
        self.timers.append(timer)

    @property
    def total_elapsed(self) -> float:
        return sum(t.elapsed for t in self.timers)

    @property
    def total_llm_calls(self) -> int:
        return sum(t.llm_calls for t in self.timers)

    @property
    def overall_status(self) -> str:
        if any(t.status == "failed" for t in self.timers):
            return "failed"
        if all(t.status == "success" for t in self.timers):
            return "success"
        return "partial"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "started_at": self.start_time.isoformat(timespec="seconds"),
            "total_elapsed_seconds": round(self.total_elapsed, 3),
            "total_llm_calls": self.total_llm_calls,
            "overall_status": self.overall_status,
            "agents": [t.to_dict() for t in self.timers],
        }

    def print_summary(self) -> None:
        """Print formatted summary to console."""
        print("\n" + "=" * 60)
        print(f"  Pipeline Execution Summary")
        print("=" * 60)
        print(f"  Query        : {self.query[:50]}...")
        print(f"  Total time   : {self.total_elapsed:.2f}s")
        print(f"  LLM calls    : {self.total_llm_calls}")
        print(f"  Status       : {self.overall_status.upper()}\n")
        print(f"  {'Agent':<22}{'Time':<10}{'Calls':<8}{'Status':<10}")
        print("  " + "-" * 50)
        for t in self.timers:
            print(
                f"  {t.agent_name:<22}{t.elapsed:.2f}s{'':<5}"
                f"{t.llm_calls:<8}{t.status:<10}"
            )
        print("=" * 60 + "\n")

    def save_json(self, path: str | Path | None = None) -> Path:
        """Save summary as a JSON log file."""
        LOG_DIR.mkdir(exist_ok=True)
        if path is None:
            ts = self.start_time.strftime("%Y%m%d_%H%M%S")
            path = LOG_DIR / f"execution_{ts}.json"
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return path


@contextmanager
def pipeline_session(query: str):
    """
    Convenience context manager wrapping an entire pipeline run.

    Example:
        with pipeline_session("Allocate $10k aggressive") as session:
            with AgentTimer("ChainAgent") as t1:
                ...
            session.add(t1)
        session.print_summary()
    """
    session = PipelineExecutionSummary(query=query)
    try:
        yield session
    finally:
        session.print_summary()


# =========================================================================
# Module Self-Test
# =========================================================================
if __name__ == "__main__":
    logger = get_logger("SelfTest")
    logger.info("Logger module self-test starting")

    summary = PipelineExecutionSummary(query="Test pipeline run")

    for agent_name, sleep_time, llm_calls in [
        ("OrchestratorAgent", 0.05, 1),
        ("ChainAgent", 0.10, 1),
        ("SentimentAgent", 0.08, 1),
        ("RiskProfileAgent", 0.06, 1),
        ("OrchestratorAgent.synth", 0.04, 1),
    ]:
        with AgentTimer(agent_name, llm_calls=llm_calls) as t:
            time.sleep(sleep_time)
        summary.add(t)

    summary.print_summary()
    print("[OK] Logger module loaded successfully.")
