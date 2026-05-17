"""Loguru → JSON for production-ready structured logging.

Emits one JSON object per record on stderr. The ``extra`` bag carries
per-request context (request_id, route, latency_ms, status), which the
middleware fills via ``logger.bind(...)``.
"""

from __future__ import annotations

import json
import sys

from loguru import logger


def _json_sink(message) -> None:  # type: ignore[no-untyped-def]
    record = message.record
    payload = {
        "ts": record["time"].isoformat(),
        "level": record["level"].name,
        "logger": record["name"],
        "msg": record["message"],
    }
    # Carry over bound context (request_id, route, etc.)
    extra = record.get("extra") or {}
    for k, v in extra.items():
        if k not in payload:
            payload[k] = v
    if record["exception"]:
        payload["exception"] = record["exception"].repr  # type: ignore[union-attr]
    print(json.dumps(payload, default=str), file=sys.stderr)


def configure_logging(level: str = "INFO", json_format: bool = True) -> None:
    """Replace loguru's default sink with JSON output (or human-readable)."""
    logger.remove()
    if json_format:
        logger.add(_json_sink, level=level)
    else:
        logger.add(
            sys.stderr,
            level=level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> — {message} {extra}",
        )
