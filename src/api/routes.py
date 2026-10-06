"""HTTP surface: one prediction endpoint plus health checks."""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from src.api.schemas import Envelope, ErrorCode, envelope
from src.core.config import get_config
from src.services.predictor import (
    EncryptedPdfError,
    InvalidImageError,
    InvalidPdfError,
    ModelNotLoadedError,
    PageLimitExceeded,
    UnsupportedFileTypeError,
    get_predictor,
    sniff,
)

logger = logging.getLogger(__name__)
router = APIRouter()

ACCEPTED = ("PDF", "PNG", "JPEG")


@router.get("/", tags=["meta"])
async def root() -> dict:
    cfg = get_config()
    return {
        "service": cfg.service["name"],
        "version": cfg.service["version"],
        "description": "Approve/reject guardrail for scanned bank-statement pages (mutasi rekening)",
        "accepts": list(ACCEPTED),
        "endpoints": ["/v1/predict", "/health", "/health/live", "/health/ready", "/docs"],
    }


@router.get("/health/live", tags=["health"])
async def health_live() -> dict:
    return {"status": "alive"}


@router.get("/health/ready", tags=["health"])
async def health_ready(request: Request) -> JSONResponse:
    ready = getattr(request.app.state, "model_loaded", False)
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "not_ready", "model_loaded": ready},
    )


@router.get("/health", tags=["health"])
async def health(request: Request) -> JSONResponse:
    cfg = get_config()
    loaded = getattr(request.app.state, "model_loaded", False)
    try:
        model_info = get_predictor().info()
    except ModelNotLoadedError:
        model_info = {"loaded": False}
    return JSONResponse(
        status_code=200 if loaded else 503,
        content={
            "status": "healthy" if loaded else "unhealthy",
            "service": cfg.service["name"],
            "version": cfg.service["version"],
            "model": model_info,
        },
    )


@router.post(
    "/v1/predict",
    tags=["prediction"],
    response_model=Envelope,
    summary="Classify a bank-statement PDF or page image as approve or reject",
)
async def predict(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    cfg = get_config()
    max_bytes = int(cfg.request["max_upload_mb"]) * 1024 * 1024
    max_pages = int(cfg.request["max_pages"])

    # Reject on the declared length before reading the body, so an oversized
    # upload doesn't have to be spooled to disk first. The post-read check
    # below still stands — Content-Length is client-supplied and optional.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        return JSONResponse(
            status_code=400,
            content=envelope(
                400,
                f"File exceeds the {cfg.request['max_upload_mb']} MB limit",
                error_code=ErrorCode.FILE_TOO_LARGE,
                errors=[{"size_bytes": int(declared), "limit_bytes": max_bytes}],
            ),
        )

    content = await file.read()

    if len(content) > max_bytes:
        return JSONResponse(
            status_code=400,
            content=envelope(
                400,
                f"File exceeds the {cfg.request['max_upload_mb']} MB limit",
                error_code=ErrorCode.FILE_TOO_LARGE,
                errors=[{"size_bytes": len(content), "limit_bytes": max_bytes}],
            ),
        )

    # Sniff the magic bytes rather than trusting file.content_type or the
    # extension — both are caller-controlled and neither is checked anywhere else.
    if sniff(content) == "unknown":
        return JSONResponse(
            status_code=400,
            content=envelope(
                400,
                "Uploaded file is not a PDF, PNG or JPEG",
                error_code=ErrorCode.INVALID_FILE_TYPE,
                errors=[{
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "accepted": list(ACCEPTED),
                }],
            ),
        )

    try:
        predictor = get_predictor()
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(
            request.app.state.inference_pool,
            lambda: predictor.predict(content, max_pages=max_pages, filename=file.filename),
        )
    except EncryptedPdfError as exc:
        return JSONResponse(
            status_code=400,
            content=envelope(400, str(exc), error_code=ErrorCode.ENCRYPTED_PDF),
        )
    except PageLimitExceeded as exc:
        return JSONResponse(
            status_code=400,
            content=envelope(
                400,
                str(exc),
                error_code=ErrorCode.PAGE_LIMIT_EXCEEDED,
                errors=[{"n_pages": exc.n_pages, "limit": exc.limit}],
            ),
        )
    except (InvalidPdfError, InvalidImageError, UnsupportedFileTypeError) as exc:
        return JSONResponse(
            status_code=400,
            content=envelope(400, str(exc), error_code=ErrorCode.INVALID_FILE_TYPE),
        )
    except ModelNotLoadedError as exc:
        logger.error("Prediction attempted with no model loaded: %s", exc)
        return JSONResponse(
            status_code=503,
            content=envelope(503, "Model is not loaded", error_code=ErrorCode.MODEL_NOT_LOADED),
        )
    except Exception:
        logger.exception("Inference failed for %s", file.filename)
        return JSONResponse(
            status_code=500,
            content=envelope(500, "Inference failed", error_code=ErrorCode.INFERENCE_FAILED),
        )

    logger.info(
        "%s (%s): %d page(s) -> %s (%d reject) in %.0f ms",
        file.filename, data["document"]["input_type"], data["document"]["n_pages"],
        data["document"]["verdict"], data["document"]["n_reject"], data["processing_time_ms"],
    )
    return JSONResponse(status_code=200, content=envelope(200, "Prediction successful", data=data))
