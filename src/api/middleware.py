"""Prometheus instrumentation and structured request logging middleware.

Exposes three Prometheus metrics at ``GET /metrics``:

* ``http_requests_total`` — counter labelled by ``method``, ``path``, ``status``.
* ``http_request_duration_seconds`` — histogram labelled by ``method``, ``path``.
* ``http_requests_in_progress`` — gauge labelled by ``method``, ``path``.
* ``predictions_total`` — counter labelled by predicted ``class_label`` (0/1).

Usage::

    from src.api.middleware import setup_metrics
    app = FastAPI()
    setup_metrics(app)
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Metric definitions (module-level singletons)
# --------------------------------------------------------------------------- #

REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests received, by method/path/status.",
    ["method", "path", "status"],
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds, by method/path.",
    ["method", "path"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests currently being processed.",
    ["method", "path"],
)

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total predictions served, labelled by predicted class.",
    ["class_label"],
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that records request count, latency, and in-progress gauge.

    Increments :data:`REQUESTS_TOTAL`, observes :data:`REQUEST_DURATION`, and
    manages the :data:`REQUESTS_IN_PROGRESS` gauge for every HTTP request.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process the request and record Prometheus metrics.

        Parameters
        ----------
        request:
            The incoming ASGI request.
        call_next:
            Callable that invokes the next middleware or route handler.

        Returns
        -------
        Response
            The response from the downstream handler.
        """
        method = request.method
        path = request.url.path

        REQUESTS_IN_PROGRESS.labels(method=method, path=path).inc()
        start = time.perf_counter()
        try:
            response = await call_next(request)
            status = str(response.status_code)
        except Exception:
            status = "500"
            raise
        finally:
            elapsed = time.perf_counter() - start
            REQUEST_DURATION.labels(method=method, path=path).observe(elapsed)
            REQUESTS_IN_PROGRESS.labels(method=method, path=path).dec()
            REQUESTS_TOTAL.labels(method=method, path=path, status=status).inc()

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that emits a structured log line for every request.

    Log format::

        METHOD path → status_code (elapsed_ms ms)
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """Log request and response details.

        Parameters
        ----------
        request:
            The incoming ASGI request.
        call_next:
            Callable that invokes the next middleware or route handler.

        Returns
        -------
        Response
            The response from the downstream handler.
        """
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s → %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


def setup_metrics(app: FastAPI) -> None:
    """Attach Prometheus and logging middleware to a FastAPI application.

    Call this once during application construction (before the app starts
    accepting requests).

    Parameters
    ----------
    app:
        The :class:`~fastapi.FastAPI` instance to instrument.
    """
    app.add_middleware(PrometheusMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
