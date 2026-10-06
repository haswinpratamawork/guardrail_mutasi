"""FastAPI application for the bank-statement guardrail classifier."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.api.schemas import ErrorCode, envelope
from src.core.config import get_config
from src.core.logging import setup_logging
from src.core.validation import validate_startup_configuration
from src.middleware.add_requestid import RequestIdMiddleware
from src.middleware.content_type import ContentTypeMiddleware
from src.services.gcs_service import download_model
from src.services.predictor import cleanup_predictor, init_predictor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    setup_logging(cfg.logging["level"], cfg.logging["dir"])
    validate_startup_configuration()

    # The probe is tiny and the backbone is a 5M-parameter CNN — more torch
    # threads than this just adds contention against the inference pool.
    torch.set_num_threads(int(cfg.model["torch_threads"]))

    app.state.model_loaded = False
    app.state.inference_pool = ThreadPoolExecutor(
        max_workers=int(cfg.request["inference_workers"]), thread_name_prefix="guardrail-infer"
    )

    local_path = cfg.resolve(cfg.model["local_path"])
    if os.getenv("SKIP_MODEL_DOWNLOAD", "").lower() not in ("1", "true", "yes"):
        download_model(cfg.gcs["bucket_name"], cfg.gcs["object"], local_path)
    else:
        logger.warning("SKIP_MODEL_DOWNLOAD set — expecting the artifact to already be at %s", local_path)

    predictor = init_predictor(
        backbone=cfg.model["backbone"],
        threshold=float(cfg.model["threshold"]),
        batch_size=int(cfg.model["embedding_batch_size"]),
    )
    # Deliberately unguarded: if the model can't load, the exception propagates
    # out of lifespan and uvicorn aborts. Better a container that refuses to
    # start than one that serves 503s while looking healthy to the scheduler.
    predictor.load_model(local_path)
    predictor.warmup()
    app.state.model_loaded = True
    logger.info("%s ready on port %s", cfg.service["name"], cfg.service["port"])

    try:
        yield
    finally:
        app.state.model_loaded = False
        app.state.inference_pool.shutdown(wait=True)
        cleanup_predictor()
        logger.info("Shutdown complete")


def create_app() -> FastAPI:
    cfg = get_config()
    app = FastAPI(
        title=cfg.service["name"],
        version=str(cfg.service["version"]),
        description=(
            "Classifies each page of a scanned bank statement (mutasi rekening) as "
            "approve or reject, as a guardrail in front of the OCR pipeline."
        ),
        lifespan=lifespan,
    )

    # Order matters. CORS outermost; ContentType before RequestId so the 400
    # envelopes it emits still carry a request_id.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(ContentTypeMiddleware)
    app.add_middleware(RequestIdMiddleware)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc: HTTPException):
        error_code = ErrorCode.INTERNAL_ERROR
        if isinstance(exc.detail, str) and exc.detail in vars(ErrorCode).values():
            error_code = exc.detail
        return JSONResponse(
            status_code=exc.status_code,
            content=envelope(exc.status_code, str(exc.detail), error_code=error_code),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc: RequestValidationError):
        return JSONResponse(
            status_code=400,
            content=envelope(
                400,
                "Request validation failed",
                error_code=ErrorCode.INVALID_FILE_TYPE,
                # errors() can carry exception objects and raw bytes in `ctx`;
                # encode before it reaches the JSON serializer.
                errors=jsonable_encoder(exc.errors()),
            ),
        )

    app.include_router(router)
    return app


app = create_app()
