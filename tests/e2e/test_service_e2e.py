"""Smoke tests against a running container.

Skipped unless BASE_URL is set, so they never gate the image build:

    BASE_URL=http://localhost:8090 .venv/bin/python -m pytest tests/e2e -q
"""
from __future__ import annotations

import io
import os

import pytest

from tests.conftest import make_image, make_pdf

BASE_URL = os.environ.get("BASE_URL")
pytestmark = pytest.mark.skipif(not BASE_URL, reason="BASE_URL not set — no running service to test")


@pytest.fixture(scope="module")
def http():
    import httpx

    with httpx.Client(base_url=BASE_URL, timeout=120) as client:
        yield client


def test_container_is_live(http):
    assert http.get("/health/live").json() == {"status": "alive"}


def test_container_is_ready_with_a_model(http):
    body = http.get("/health/ready").json()
    assert body["model_loaded"] is True


def test_container_reports_model_provenance(http):
    model = http.get("/health").json()["model"]
    assert model["backbone"] == "efficientnet_b0"
    assert model["render_dpi"] == 200
    assert model["artifact_version"]


def test_container_scores_a_multi_page_pdf(http):
    response = http.post(
        "/v1/predict",
        files={"file": ("smoke.pdf", io.BytesIO(make_pdf(3)), "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["document"]["n_pages"] == 3
    assert [p["page"] for p in data["pages"]] == [1, 2, 3]


def test_container_refuses_an_encrypted_pdf(http):
    response = http.post(
        "/v1/predict",
        files={"file": ("enc.pdf", io.BytesIO(make_pdf(1, encrypt_with="pw")), "application/pdf")},
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "ENCRYPTED_PDF"


def test_container_is_open_and_needs_no_credentials(http):
    """/v1/predict is deliberately unauthenticated."""
    response = http.post(
        "/v1/predict",
        files={"file": ("x.pdf", io.BytesIO(make_pdf(1)), "application/pdf")},
    )
    assert response.status_code == 200


def test_container_scores_a_png_page(http):
    response = http.post(
        "/v1/predict",
        files={"file": ("page.png", io.BytesIO(make_image("PNG")), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["document"]["input_type"] == "image"
    assert data["document"]["n_pages"] == 1


def test_container_scores_a_jpeg_page(http):
    response = http.post(
        "/v1/predict",
        files={"file": ("page.jpg", io.BytesIO(make_image("JPEG")), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["data"]["document"]["n_pages"] == 1


def test_container_refuses_an_unsupported_type(http):
    response = http.post(
        "/v1/predict",
        files={"file": ("notes.txt", io.BytesIO(b"plain text"), "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_FILE_TYPE"
