"""Request-scoped middleware: request_id, timing, structured access log.

Adds an ``X-Request-ID`` header (echoed back), measures wall-clock latency,
emits one structured log line per request, and feeds Prometheus
collectors. Errors are caught and re-raised so FastAPI's exception
handlers still run.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.metrics import (
    REQUEST_COUNT,
    REQUEST_IN_FLIGHT,
    REQUEST_LATENCY,
)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attaches a request_id, logs structured access entries, updates metrics."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        request.state.request_id = request_id
        route = request.url.path
        method = request.method

        REQUEST_IN_FLIGHT.labels(route=route, method=method).inc()
        t0 = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception:
            logger.bind(request_id=request_id, route=route, method=method).exception(
                "Unhandled exception in request"
            )
            raise
        finally:
            elapsed = time.perf_counter() - t0
            REQUEST_IN_FLIGHT.labels(route=route, method=method).dec()
            REQUEST_LATENCY.labels(route=route, method=method).observe(elapsed)
            REQUEST_COUNT.labels(route=route, method=method, status=str(status_code)).inc()
            logger.bind(
                request_id=request_id,
                route=route,
                method=method,
                status=status_code,
                latency_ms=round(elapsed * 1000, 2),
            ).info("request")
