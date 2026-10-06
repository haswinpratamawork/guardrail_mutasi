"""Response envelope + Pydantic models.

The models are for OpenAPI documentation. Following the house pattern, routes
return the plain envelope dict rather than wiring these as `response_model` —
that keeps error and success responses provably the same shape, which a
`response_model` on the success path alone does not.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.core.logging import request_id_ctx

STATUS_DESCS = {
    200: "OK",
    400: "Bad Request",
    500: "Internal Server Error",
    503: "Service Unavailable",
}


class ErrorCode:
    INVALID_FILE_TYPE = "INVALID_FILE_TYPE"
    ENCRYPTED_PDF = "ENCRYPTED_PDF"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    PAGE_LIMIT_EXCEEDED = "PAGE_LIMIT_EXCEEDED"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


def envelope(
    status_code: int,
    message: str,
    data: Any = None,
    error_code: str | None = None,
    errors: Any = None,
) -> dict[str, Any]:
    return {
        "status_code": status_code,
        "status_desc": STATUS_DESCS.get(status_code, ""),
        "message": message,
        "data": data,
        "error_code": error_code,
        "errors": errors,
        "request_id": request_id_ctx.get() or None,
    }


class PageResult(BaseModel):
    """One page of the submitted document."""

    page: int = Field(..., ge=1, description="1-based page number, matching source_page in predictions.csv")
    verdict: str = Field(..., description="approve | reject")
    proba_reject: float = Field(..., ge=0.0, le=1.0)
    proba_approve: float = Field(..., ge=0.0, le=1.0)


class DocumentResult(BaseModel):
    filename: str | None
    input_type: str = Field(..., description="pdf | image — what the caller uploaded")
    n_pages: int = Field(..., ge=1, description="Always 1 for a page image")
    n_approve: int = Field(..., ge=0)
    n_reject: int = Field(..., ge=0)
    verdict: str = Field(..., description="approve | reject")
    verdict_rule: str = Field("any_page_reject", description="How the document verdict was derived")


class ModelInfo(BaseModel):
    backbone: str
    head: str
    artifact_version: str
    render_dpi: int
    threshold: float


class PredictData(BaseModel):
    document: DocumentResult
    pages: list[PageResult]
    model: ModelInfo
    processing_time_ms: float


class Envelope(BaseModel):
    status_code: int
    status_desc: str
    message: str
    data: PredictData | None = None
    error_code: str | None = None
    errors: Any = None
    request_id: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "status_code": 200,
                "status_desc": "OK",
                "message": "Prediction successful",
                "data": {
                    "document": {
                        "filename": "PF24018IOR_mutasi_rekening.pdf",
                        "input_type": "pdf",
                        "n_pages": 7,
                        "n_approve": 5,
                        "n_reject": 2,
                        "verdict": "reject",
                        "verdict_rule": "any_page_reject",
                    },
                    "pages": [
                        {"page": 1, "verdict": "approve", "proba_reject": 0.0421, "proba_approve": 0.9579},
                        {"page": 2, "verdict": "reject", "proba_reject": 0.8817, "proba_approve": 0.1183},
                    ],
                    "model": {
                        "backbone": "efficientnet_b0",
                        "head": "logreg",
                        "artifact_version": "20260902T061500Z",
                        "render_dpi": 200,
                        "threshold": 0.5,
                    },
                    "processing_time_ms": 1830.4,
                },
                "error_code": None,
                "errors": None,
                "request_id": "9f2c1e4a7b3d4e8f",
            }
        }
    }
