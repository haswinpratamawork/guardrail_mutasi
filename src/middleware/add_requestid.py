"""Give every request an id: honour an inbound x-request-id, else mint one."""
from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from src.core.logging import request_id_ctx


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_ctx.set(request_id)
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)
        response.headers["x-request-id"] = request_id
        return response
