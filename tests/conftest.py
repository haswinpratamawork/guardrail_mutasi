from __future__ import annotations

import io
import os
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "src" / "models" / "artifacts" / "probe_efficientnet_b0_logreg.joblib"

os.environ.setdefault("SKIP_MODEL_DOWNLOAD", "1")
os.environ.setdefault("LOG_DIR", "/tmp/guardrail-test-logs")


def make_pdf(n_pages: int = 1, encrypt_with: str | None = None) -> bytes:
    """A blank n-page PDF. Blank pages are fine for the plumbing tests — what
    they assert is shape, counts and error handling, not model quality."""
    pdf = pdfium.PdfDocument.new()
    for _ in range(n_pages):
        pdf.new_page(200, 300)
    buffer = io.BytesIO()
    pdf.save(buffer)
    pdf.close()
    raw = buffer.getvalue()
    if encrypt_with is None:
        return raw

    import pikepdf

    with pikepdf.open(io.BytesIO(raw)) as handle:
        out = io.BytesIO()
        handle.save(out, encryption=pikepdf.Encryption(user=encrypt_with, owner=encrypt_with, R=6))
        return out.getvalue()



def make_image(
    fmt: str = "PNG",
    size: tuple[int, int] = (1700, 2200),
    mode: str = "RGB",
    exif_orientation: int | None = None,
) -> bytes:
    """A blank page-shaped image. 1700x2200 is roughly an A4 page at 200 dpi,
    the resolution the offline pipeline rasterized at."""
    image = Image.new(mode, size, "white" if mode != "RGBA" else (255, 255, 255, 0))
    buffer = io.BytesIO()
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation  # Orientation
        image.save(buffer, fmt, exif=exif)
    else:
        image.save(buffer, fmt)
    return buffer.getvalue()


@pytest.fixture(scope="session")
def client():
    """Boots the real app, model and all. Skips if the artifact isn't staged —
    run `python -m src.persist_probe` first."""
    if not ARTIFACT.exists():
        pytest.skip(f"No model artifact at {ARTIFACT}; run python -m src.persist_probe")

    from fastapi.testclient import TestClient

    from src.main import app

    with TestClient(app) as test_client:
        yield test_client
