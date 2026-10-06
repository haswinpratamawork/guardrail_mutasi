"""Scores every page of an in-memory PDF with the champion probe.

Parity with the offline pipeline is the whole point of this module, so it
reuses the training code rather than restating it:

  * `render_page` (src/data/render.py) at the artifact's own `render_dpi`
  * `build_feature_extractor` (src/models/backbones.py) with no `image_size`,
    i.e. the timm default transform the probe was fitted behind
  * `ProbeResult` (src/models/linear_probe.py) for predict/predict_proba

An uploaded JPEG/PNG is treated as one already-rasterized page and skips
straight to preprocessing. That is not a shortcut: it is exactly what
`verify_parity --mode png` does, because the cached PNGs in
`data_cache/test467_images/` *are* the offline pipeline's rasterization. A
200-dpi page image therefore takes the same path it took offline.

The one thing it does restate is the three-line embed loop from
`train_probe.embed_split`: that function takes a DataFrame of file paths, and
here the pages only ever exist as PIL objects in memory. The important
property it preserves is preprocessing one page at a time and releasing it
before opening the next — some scans decode to ~144MB and holding a batch of
them is what OOM-killed the offline runs.
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pypdfium2 as pdfium
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from src.data.render import render_page
from src.models.backbones import build_feature_extractor
from src.models.linear_probe import load_probe

logger = logging.getLogger(__name__)


class PredictorError(Exception):
    """Base for errors the API maps onto a specific error_code."""


class InvalidPdfError(PredictorError):
    """Bytes that pdfium can't open as a PDF at all."""


class InvalidImageError(PredictorError):
    """Bytes that PIL can't open as an image, or that are implausibly large."""


class UnsupportedFileTypeError(PredictorError):
    """Not one of the accepted formats (PDF, PNG, JPEG)."""


class EncryptedPdfError(PredictorError):
    """Password-protected PDF.

    Deliberately terminal: the service must never try passwords. In this
    corpus approve/ and reject/ are encrypted with different passwords, so
    'which password opens it' is a perfect label oracle on historical files
    and meaningless on genuinely new ones (README "Known open issue").
    Callers send decrypted PDFs.
    """


class PageLimitExceeded(PredictorError):
    def __init__(self, n_pages: int, limit: int) -> None:
        super().__init__(f"Document has {n_pages} pages, limit is {limit}")
        self.n_pages = n_pages
        self.limit = limit


class ModelNotLoadedError(PredictorError):
    pass


PDF_MAGIC = b"%PDF-"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"

# A page rendered at 200 dpi is ~1700x2200. This ceiling is far above any real
# scan but well under PIL's own decompression-bomb limit, and it is checked
# from the header before any pixels are decoded.
MAX_IMAGE_PIXELS = 80_000_000


def sniff(content: bytes) -> str:
    """Identify the upload from its magic bytes. Never trust the filename or
    the client-supplied Content-Type — both are caller-controlled."""
    if content.startswith(PDF_MAGIC):
        return "pdf"
    if content.startswith(PNG_MAGIC):
        return "png"
    if content.startswith(JPEG_MAGIC):
        return "jpeg"
    return "unknown"


def _open_image(image_bytes: bytes) -> Image.Image:
    """Decode one page image to RGB, the way the offline cache stores it.

    Two things a PDF page never needs but an uploaded photo/scan does:
      * EXIF orientation — phone and scanner JPEGs record rotation in metadata
        rather than in the pixels, and the model would otherwise see a sideways
        page.
      * Alpha — a transparent PNG flattened with a plain convert("RGB") turns
        transparent areas black, which is the opposite of a document's ground.
        Composite onto white instead.
    """
    try:
        handle = Image.open(io.BytesIO(image_bytes))
        handle.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(f"Could not decode the uploaded file as an image: {exc}") from exc

    with handle:
        width, height = handle.size
        if width * height > MAX_IMAGE_PIXELS:
            raise InvalidImageError(
                f"Image is {width}x{height} ({width * height / 1e6:.0f} MP); "
                f"the limit is {MAX_IMAGE_PIXELS / 1e6:.0f} MP"
            )
        image = ImageOps.exif_transpose(handle)
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(canvas, image)
        return image.convert("RGB")


def _open_pdf(pdf_bytes: bytes) -> pdfium.PdfDocument:
    try:
        return pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    except pdfium.PdfiumError as exc:
        if "password" in str(exc).lower():
            raise EncryptedPdfError(
                "PDF is password-protected. Send a decrypted PDF — this service does not "
                "accept or guess passwords."
            ) from exc
        raise InvalidPdfError(f"Could not open the uploaded file as a PDF: {exc}") from exc


class Predictor:
    def __init__(self, backbone: str, threshold: float, batch_size: int) -> None:
        self.backbone = backbone
        self.threshold = float(threshold)
        self.batch_size = int(batch_size)
        self._fe = None
        self._probe = None
        self._meta: dict[str, Any] = {}

    # ---- lifecycle -------------------------------------------------------

    def load_model(self, artifact_path: str | Path) -> None:
        path = Path(artifact_path)
        if not path.exists():
            raise ModelNotLoadedError(f"Model artifact not found at {path}")

        probe, payload = load_probe(path)
        meta = payload["meta"]

        if meta["backbone"] != self.backbone:
            raise ModelNotLoadedError(
                f"Artifact holds backbone {meta['backbone']!r} but config asks for {self.backbone!r}"
            )

        # pretrained=False: architecture only, no hub call. The weights come
        # from the bundle, so the served backbone is the one the probe was fit on.
        fe = build_feature_extractor(meta["backbone"], pretrained=False)
        fe.model.load_state_dict(payload["backbone_state_dict"], strict=True)
        fe.model.eval()

        if fe.embedding_dim != meta["embedding_dim"]:
            raise ModelNotLoadedError(
                f"Backbone emits {fe.embedding_dim}-d embeddings, artifact expects {meta['embedding_dim']}"
            )

        self._fe, self._probe, self._meta = fe, probe, meta
        logger.info(
            "Model loaded: %s + %s (artifact %s, dpi=%s, threshold=%s, fitted on %s)",
            meta["backbone"], meta["classifier"], meta["created_at"],
            meta["render_dpi"], self.threshold, meta.get("embedded_on", "?"),
        )

    @property
    def is_loaded(self) -> bool:
        return self._fe is not None and self._probe is not None

    def info(self) -> dict[str, Any]:
        if not self.is_loaded:
            return {"loaded": False}
        return {
            "loaded": True,
            "backbone": self._meta["backbone"],
            "head": self._meta["classifier"],
            "artifact_version": self._meta["created_at"],
            "render_dpi": self._meta["render_dpi"],
            "threshold": self.threshold,
            "embedding_dim": self._meta["embedding_dim"],
            "fitted_metrics": self._meta.get("internal_test_metrics"),
            "device": "cpu",
        }

    def warmup(self) -> None:
        """Run one synthetic page through the full path so the first real
        request doesn't pay lazy-init costs (timm kernels, PIL codecs)."""
        pdf = pdfium.PdfDocument.new()
        pdf.new_page(200, 300)
        buffer = io.BytesIO()
        pdf.save(buffer)
        pdf.close()
        self.predict_pdf(buffer.getvalue(), max_pages=1)
        logger.info("Warmup complete")

    # ---- inference -------------------------------------------------------

    def _embed_pages(self, pdf_bytes: bytes, n_pages: int) -> np.ndarray:
        dpi = int(self._meta["render_dpi"])
        chunks: list[np.ndarray] = []
        tensors: list[torch.Tensor] = []
        for index in range(n_pages):
            image = render_page(pdf_bytes, index, dpi=dpi)  # 0-indexed, already RGB
            tensors.append(self._fe.preprocess(image))
            del image  # release the full-resolution page before opening the next
            if len(tensors) == self.batch_size:
                chunks.append(self._fe.embed_batch(torch.stack(tensors)).cpu().numpy())
                tensors = []
        if tensors:
            chunks.append(self._fe.embed_batch(torch.stack(tensors)).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def _assemble(
        self,
        proba_reject: np.ndarray,
        started: float,
        filename: str | None,
        input_type: str,
    ) -> dict[str, Any]:
        """Turn per-page P(reject) into the response body. Shared by both input
        kinds so a PDF page and an uploaded page image cannot diverge."""
        pages = []
        for index, p_reject in enumerate(proba_reject):
            # `>=` reads as the natural meaning of a threshold. At the default
            # 0.5 this agrees with the probe's own argmax everywhere except an
            # exact 0.5 tie, which argmax would call approve.
            verdict = "reject" if p_reject >= self.threshold else "approve"
            pages.append(
                {
                    "page": index + 1,  # 1-based, matching source_page in predictions.csv
                    "verdict": verdict,
                    "proba_reject": round(float(p_reject), 6),
                    "proba_approve": round(float(1.0 - p_reject), 6),
                }
            )

        n_reject = sum(1 for page in pages if page["verdict"] == "reject")
        return {
            "document": {
                "filename": filename,
                "input_type": input_type,
                "n_pages": len(pages),
                "n_approve": len(pages) - n_reject,
                "n_reject": n_reject,
                # A guardrail exists to stop unusable statements reaching OCR;
                # one bad page makes the document unusable.
                "verdict": "reject" if n_reject else "approve",
                "verdict_rule": "any_page_reject",
            },
            "pages": pages,
            "model": {
                "backbone": self._meta["backbone"],
                "head": self._meta["classifier"],
                "artifact_version": self._meta["created_at"],
                "render_dpi": int(self._meta["render_dpi"]),
                "threshold": self.threshold,
            },
            "processing_time_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    def predict(self, content: bytes, max_pages: int, filename: str | None = None) -> dict[str, Any]:
        """Score an upload, whichever accepted format it is.

        Blocking. Callers run this in a thread so the event loop stays free.
        """
        kind = sniff(content)
        if kind == "pdf":
            return self.predict_pdf(content, max_pages=max_pages, filename=filename)
        if kind in ("png", "jpeg"):
            return self.predict_image(content, filename=filename)
        raise UnsupportedFileTypeError(
            "Unsupported file type. Send a PDF, PNG or JPEG."
        )

    def predict_image(self, image_bytes: bytes, filename: str | None = None) -> dict[str, Any]:
        """Score a single already-rasterized page image (PNG/JPEG).

        No rendering step: the upload *is* the page raster, so this is the same
        path `verify_parity --mode png` takes over the cached test_467 PNGs.
        """
        if not self.is_loaded:
            raise ModelNotLoadedError("Model is not loaded")

        started = time.perf_counter()
        image = _open_image(image_bytes)
        try:
            tensor = self._fe.preprocess(image)
        finally:
            image.close()

        embeddings = self._fe.embed_batch(torch.stack([tensor])).cpu().numpy()
        proba_reject = np.asarray(self._probe.predict_proba(embeddings), dtype=float).reshape(-1)
        return self._assemble(proba_reject, started, filename, input_type="image")

    def predict_pdf(self, pdf_bytes: bytes, max_pages: int, filename: str | None = None) -> dict[str, Any]:
        """Blocking. Callers run this in a thread so the event loop stays free."""
        if not self.is_loaded:
            raise ModelNotLoadedError("Model is not loaded")

        started = time.perf_counter()
        pdf = _open_pdf(pdf_bytes)
        try:
            n_pages = len(pdf)
        finally:
            pdf.close()

        if n_pages == 0:
            raise InvalidPdfError("PDF contains no pages")
        if n_pages > max_pages:
            raise PageLimitExceeded(n_pages, max_pages)

        embeddings = self._embed_pages(pdf_bytes, n_pages)
        proba_reject = np.asarray(self._probe.predict_proba(embeddings), dtype=float).reshape(-1)
        return self._assemble(proba_reject, started, filename, input_type="pdf")


_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    if _predictor is None:
        raise ModelNotLoadedError("Predictor has not been initialised")
    return _predictor


def init_predictor(backbone: str, threshold: float, batch_size: int) -> Predictor:
    global _predictor
    _predictor = Predictor(backbone=backbone, threshold=threshold, batch_size=batch_size)
    return _predictor


def cleanup_predictor() -> None:
    global _predictor
    _predictor = None
