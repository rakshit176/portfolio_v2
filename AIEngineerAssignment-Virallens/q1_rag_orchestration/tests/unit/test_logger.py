# tests/unit/test_logger.py
import pytest
import json
import logging
from io import StringIO
from utils.logger import get_logger, StructuredFormatter

def test_logger_emits_structured_json():
    """Logger should emit JSON with timestamp, trace_id, agent, event."""
    logger = get_logger("test_agent")

    # Remove existing handlers to avoid duplicate output
    logger.handlers.clear()

    # Capture log output
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("test_event", extra={"query": "test query", "latency_ms": 45})

    log_output = log_stream.getvalue()
    log_entry = json.loads(log_output.strip())

    assert "timestamp" in log_entry
    assert log_entry["logger"] == "test_agent"
    assert log_entry["message"] == "test_event"
    assert log_entry["query"] == "test query"
    assert log_entry["latency_ms"] == 45

def test_logger_generates_trace_id():
    """Logger should auto-generate trace_id if not provided."""
    logger = get_logger("test_agent_unique")

    # Remove existing handlers to avoid duplicate output
    logger.handlers.clear()

    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("test_event")

    log_output = log_stream.getvalue()
    log_entry = json.loads(log_output.strip())

    assert "trace_id" in log_entry
    assert len(log_entry["trace_id"]) > 0
