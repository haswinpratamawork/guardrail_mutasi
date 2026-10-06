"""Rasterize a (decrypted) single-page PDF to a PIL image.

Uses pypdfium2 rather than PyMuPDF: permissive license (Apache/BSD via
Google's PDFium bindings), no system poppler dependency, fast — a better fit
than PyMuPDF's AGPL license for internal bank tooling.
"""
from __future__ import annotations

import io

import pypdfium2 as pdfium
from PIL import Image


def render_page(pdf_bytes: bytes, page_index: int, dpi: int = 200) -> Image.Image:
    """Render an arbitrary page (0-indexed) of a plain (already-decrypted)
    PDF to an RGB PIL image."""
    scale = dpi / 72.0  # PDF user space is 72 dpi by definition
    pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    try:
        page = pdf[page_index]
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil().convert("RGB")
    finally:
        pdf.close()
    return pil_image


def render_first_page(pdf_bytes: bytes, dpi: int = 200) -> Image.Image:
    """Render page 0 of a plain (already-decrypted) PDF to an RGB PIL image."""
    return render_page(pdf_bytes, 0, dpi=dpi)
