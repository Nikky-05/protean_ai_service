"""Correlation-ID middleware.

Reads `X-Correlation-Id` from the incoming request (or generates one),
binds it on the structlog context for the lifetime of the request, and
echoes it back on the response so the Node orchestrator can stitch
logs together end-to-end.
"""

import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    HEADER = "X-Correlation-Id"

    async def dispatch(self, request: Request, call_next):
        cid = request.headers.get(self.HEADER) or str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(correlation_id=cid, path=request.url.path)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers[self.HEADER] = cid
        return response
