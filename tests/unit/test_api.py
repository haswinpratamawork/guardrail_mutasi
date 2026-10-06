"""Contract tests for the prediction endpoint and its error branches."""
from __future__ import annotations

import io

import pytest

from tests.conftest import make_image, make_pdf

ENVELOPE_KEYS = {
    "status_code", "status_desc", "message", "data", "error_code", "errors", "request_id",
}


def post_pdf(client, content: bytes, filename: str = "doc.pdf", headers: dict | None = None):
    return client.post(
        "/v1/predict",
        headers=headers or {},
        files={"file": (filename, io.BytesIO(content), "application/pdf")},
    )


# --- health -------------------------------------------------------------


def test_health_live(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_health_ready_reports_loaded_model(client):
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_health_exposes_model_provenance(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["model"]["backbone"] == "efficientnet_b0"
    assert body["model"]["head"] == "logreg"
    # The artifact must carry the DPI it was trained at, not inherit a default.
    assert body["model"]["render_dpi"] == 200


# --- success path -------------------------------------------------------


def test_single_page_response_shape(client):
    response = post_pdf(client, make_pdf(1))
    assert response.status_code == 200
    body = response.json()
    assert set(body) == ENVELOPE_KEYS
    assert body["error_code"] is None
    assert body["request_id"]

    data = body["data"]
    assert data["document"]["n_pages"] == 1
    assert len(data["pages"]) == 1
    assert data["pages"][0]["page"] == 1
    assert data["model"]["backbone"] == "efficientnet_b0"
    assert data["processing_time_ms"] > 0


def test_multi_page_returns_one_result_per_page_in_order(client):
    response = post_pdf(client, make_pdf(5))
    data = response.json()["data"]
    assert data["document"]["n_pages"] == 5
    assert [p["page"] for p in data["pages"]] == [1, 2, 3, 4, 5]


def test_probabilities_are_complementary_and_match_verdict(client):
    data = post_pdf(client, make_pdf(3)).json()["data"]
    for page in data["pages"]:
        assert page["proba_reject"] + page["proba_approve"] == pytest.approx(1.0, abs=1e-6)
        assert 0.0 <= page["proba_reject"] <= 1.0
        expected = "reject" if page["proba_reject"] >= data["model"]["threshold"] else "approve"
        assert page["verdict"] == expected


def test_document_counts_agree_with_pages(client):
    data = post_pdf(client, make_pdf(4)).json()["data"]
    document, pages = data["document"], data["pages"]
    assert document["n_reject"] == sum(1 for p in pages if p["verdict"] == "reject")
    assert document["n_approve"] == sum(1 for p in pages if p["verdict"] == "approve")
    assert document["n_approve"] + document["n_reject"] == document["n_pages"]


def test_document_verdict_is_reject_if_any_page_rejects(client):
    data = post_pdf(client, make_pdf(3)).json()["data"]
    assert data["document"]["verdict_rule"] == "any_page_reject"
    expected = "reject" if data["document"]["n_reject"] else "approve"
    assert data["document"]["verdict"] == expected


def test_filename_is_echoed_back(client):
    data = post_pdf(client, make_pdf(1), filename="PF24018IOR_mutasi_rekening.pdf").json()["data"]
    assert data["document"]["filename"] == "PF24018IOR_mutasi_rekening.pdf"


# --- error branches -----------------------------------------------------


def test_predict_needs_no_credentials(client):
    """The endpoint is deliberately unauthenticated — no API key, no header, no 401."""
    response = post_pdf(client, make_pdf(1), headers={})
    assert response.status_code == 200
    assert response.json()["data"]["document"]["n_pages"] == 1


def test_a_stray_api_key_header_is_simply_ignored(client):
    """Old callers still sending X-API-Key must not be broken by its removal."""
    response = post_pdf(client, make_pdf(1), headers={"X-API-Key": "anything-at-all"})
    assert response.status_code == 200


def test_unsupported_upload_is_rejected(client):
    response = post_pdf(client, b"this is not a document at all", filename="notes.txt")
    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "INVALID_FILE_TYPE"
    assert body["errors"][0]["accepted"] == ["PDF", "PNG", "JPEG"]


# --- page images: PNG and JPEG ------------------------------------------


@pytest.mark.parametrize("fmt,filename", [("PNG", "page.png"), ("JPEG", "page.jpg"), ("JPEG", "page.jpeg")])
def test_page_image_is_scored_as_a_one_page_document(client, fmt, filename):
    response = post_pdf(client, make_image(fmt), filename=filename)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["document"]["input_type"] == "image"
    assert data["document"]["n_pages"] == 1
    assert len(data["pages"]) == 1
    assert data["pages"][0]["page"] == 1
    assert data["document"]["verdict"] == data["pages"][0]["verdict"]


def test_image_and_pdf_responses_have_the_same_shape(client):
    from_pdf = post_pdf(client, make_pdf(1), filename="a.pdf").json()["data"]
    from_png = post_pdf(client, make_image("PNG"), filename="a.png").json()["data"]
    assert from_pdf.keys() == from_png.keys()
    assert from_pdf["document"].keys() == from_png["document"].keys()
    assert from_pdf["pages"][0].keys() == from_png["pages"][0].keys()


def test_pdf_still_reports_input_type_pdf(client):
    data = post_pdf(client, make_pdf(2), filename="a.pdf").json()["data"]
    assert data["document"]["input_type"] == "pdf"
    assert data["document"]["n_pages"] == 2


def test_the_extension_does_not_decide_the_format(client):
    """A real PNG named .pdf is still scored as an image — magic bytes win."""
    data = post_pdf(client, make_image("PNG"), filename="misleading.pdf").json()["data"]
    assert data["document"]["input_type"] == "image"


def test_corrupt_image_is_rejected_not_crashed(client):
    truncated = make_image("PNG")[:120]
    response = post_pdf(client, truncated, filename="broken.png")
    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_FILE_TYPE"


def test_encrypted_pdf_is_refused_not_guessed(client):
    """The service must never try passwords: in this corpus approve/ and reject/
    use different ones, so a successful guess would leak the label."""
    response = post_pdf(client, make_pdf(1, encrypt_with="s3cret"))
    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "ENCRYPTED_PDF"
    assert "decrypted" in body["message"].lower()


def test_page_limit_is_enforced(client, monkeypatch):
    from src.core.config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg.request, "max_pages", 2)
    response = post_pdf(client, make_pdf(3))
    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "PAGE_LIMIT_EXCEEDED"
    assert body["errors"][0] == {"n_pages": 3, "limit": 2}


def test_oversized_upload_is_rejected(client, monkeypatch):
    from src.core.config import get_config

    monkeypatch.setitem(get_config().request, "max_upload_mb", 0.000001)
    response = post_pdf(client, make_pdf(1))
    assert response.status_code == 400
    assert response.json()["error_code"] == "FILE_TOO_LARGE"


def test_wrong_content_type_gets_the_service_envelope(client):
    response = client.post("/v1/predict", json={"file": "x"})
    assert response.status_code == 400
    body = response.json()
    assert set(body) == ENVELOPE_KEYS
    assert body["error_code"] == "INVALID_FILE_TYPE"
    assert body["request_id"], "content-type rejections must still carry a request id"


def test_request_id_is_echoed_when_supplied(client):
    response = client.post(
        "/v1/predict",
        headers={"x-request-id": "abc123"},
        files={"file": ("d.pdf", io.BytesIO(make_pdf(1)), "application/pdf")},
    )
    assert response.headers["x-request-id"] == "abc123"
    assert response.json()["request_id"] == "abc123"
