# utils/logger.py
import logging
import json
import uuid
import time
from typing import Any, Dict
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    """Emit logs as structured JSON with timestamp and trace_id."""

    def format(self, record: logging.LogRecord) -> str:
        # Build base log entry
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add trace_id if not already present
        if not hasattr(record, "trace_id"):
            log_entry["trace_id"] = str(uuid.uuid4())[:8]
        else:
            log_entry["trace_id"] = record.trace_id

        # Add any extra fields from record
        for key, value in record.__dict__.items():
            if key not in {"name", "msg", "args", "levelname", "levelno",
                          "pathname", "filename", "module", "lineno",
                          "funcName", "created", "msecs", "relativeCreated",
                          "thread", "threadName", "processName", "process",
                          "message", "asctime", "trace_id"}:
                log_entry[key] = value

        return json.dumps(log_entry)

def get_logger(agent_name: str) -> Any:
    """
    Get a structured logger for an agent.

    Args:
        agent_name: Name of the agent (e.g., "router", "retriever")

    Returns:
        Logger with structured JSON formatting
    """
    logger = logging.getLogger(agent_name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)

    return logger
