"""Reject the wrong Content-Type on /v1/predict with the service's own envelope.

Without this, FastAPI's own 422 for a missing multipart body escapes before the
envelope handlers run, and callers get a differently-shaped error for what is
really just a malformed request.

Registered BEFORE RequestIdMiddleware so that the envelopes it emits still
carry a request_id.
"""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.api.schemas import ErrorCode, envelope

_MULTIPART_ROUTES = ("/v1/predict",)


class ContentTypeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and request.url.path in _MULTIPART_ROUTES:
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("multipart/form-data"):
                return JSONResponse(
                    status_code=400,
                    content=envelope(
                        400,
                        "Expected multipart/form-data with a 'file' field",
                        error_code=ErrorCode.INVALID_FILE_TYPE,
                        errors=[{"content_type": content_type or None}],
                    ),
                )
        return await call_next(request)
