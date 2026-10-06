"""Predictor logic with the model stubbed, so probabilities are exact.

The HTTP tests can't steer a real backbone onto a chosen probability, so the
threshold and aggregation rules are pinned here instead.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.services.predictor import (
    EncryptedPdfError,
    InvalidPdfError,
    ModelNotLoadedError,
    PageLimitExceeded,
    Predictor,
)
from tests.conftest import make_image, make_pdf


class _StubProbe:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, embeddings):
        return self.probabilities[: len(embeddings)]


def build_predictor(probabilities, threshold=0.5):
    predictor = Predictor(backbone="efficientnet_b0", threshold=threshold, batch_size=4)
    predictor._probe = _StubProbe(probabilities)
    predictor._meta = {
        "backbone": "efficientnet_b0",
        "classifier": "logreg",
        "created_at": "TESTVERSION",
        "render_dpi": 200,
        "embedding_dim": 4,
    }

    class _StubFE:
        embedding_dim = 4

        @staticmethod
        def preprocess(image):
            import torch

            return torch.zeros(3, 4, 4)

        @staticmethod
        def embed_batch(batch):
            import torch

            return torch.zeros(len(batch), 4)

    predictor._fe = _StubFE()
    return predictor


def test_threshold_boundary_is_inclusive():
    predictor = build_predictor([0.5, 0.499999, 0.500001])
    pages = predictor.predict_pdf(make_pdf(3), max_pages=10)["pages"]
    assert [p["verdict"] for p in pages] == ["reject", "approve", "reject"]


def test_custom_threshold_shifts_the_boundary():
    predictor = build_predictor([0.55, 0.75], threshold=0.7)
    pages = predictor.predict_pdf(make_pdf(2), max_pages=10)["pages"]
    assert [p["verdict"] for p in pages] == ["approve", "reject"]


def test_all_approve_gives_an_approve_document():
    predictor = build_predictor([0.1, 0.2, 0.3])
    document = predictor.predict_pdf(make_pdf(3), max_pages=10)["document"]
    assert document["verdict"] == "approve"
    assert (document["n_approve"], document["n_reject"]) == (3, 0)


def test_one_bad_page_rejects_the_whole_document():
    predictor = build_predictor([0.1, 0.99, 0.2])
    document = predictor.predict_pdf(make_pdf(3), max_pages=10)["document"]
    assert document["verdict"] == "reject"
    assert (document["n_approve"], document["n_reject"]) == (2, 1)


def test_pages_are_batched_without_changing_results():
    """batch_size is a throughput knob only — 9 pages across batches of 4 must
    come back in page order with the probabilities they were given."""
    probabilities = [0.05, 0.95, 0.10, 0.90, 0.15, 0.85, 0.20, 0.80, 0.25]
    predictor = build_predictor(probabilities)
    pages = predictor.predict_pdf(make_pdf(9), max_pages=10)["pages"]
    assert [p["page"] for p in pages] == list(range(1, 10))
    assert [p["proba_reject"] for p in pages] == pytest.approx(probabilities)


def test_page_limit_raises_before_any_rendering():
    predictor = build_predictor([0.1] * 5)
    with pytest.raises(PageLimitExceeded) as excinfo:
        predictor.predict_pdf(make_pdf(5), max_pages=2)
    assert (excinfo.value.n_pages, excinfo.value.limit) == (5, 2)


def test_encrypted_pdf_raises_rather_than_guessing():
    predictor = build_predictor([0.1])
    with pytest.raises(EncryptedPdfError):
        predictor.predict_pdf(make_pdf(1, encrypt_with="pw"), max_pages=10)


def test_garbage_bytes_raise_invalid_pdf():
    predictor = build_predictor([0.1])
    with pytest.raises(InvalidPdfError):
        predictor.predict_pdf(b"%PDF-1.4 truncated garbage", max_pages=10)


def test_predicting_without_a_model_raises():
    predictor = Predictor(backbone="efficientnet_b0", threshold=0.5, batch_size=4)
    assert predictor.is_loaded is False
    with pytest.raises(ModelNotLoadedError):
        predictor.predict_pdf(make_pdf(1), max_pages=10)


def test_load_model_rejects_a_backbone_mismatch(tmp_path):
    from tests.conftest import ARTIFACT

    if not ARTIFACT.exists():
        pytest.skip("no artifact staged")
    predictor = Predictor(backbone="resnet50", threshold=0.5, batch_size=4)
    with pytest.raises(ModelNotLoadedError, match="backbone"):
        predictor.load_model(ARTIFACT)


def test_artifact_roundtrips_through_the_real_loader():
    """The shipped artifact must rebuild into a working model with no network."""
    from tests.conftest import ARTIFACT

    if not ARTIFACT.exists():
        pytest.skip("no artifact staged")
    predictor = Predictor(backbone="efficientnet_b0", threshold=0.5, batch_size=4)
    predictor.load_model(ARTIFACT)
    assert predictor.is_loaded
    info = predictor.info()
    assert info["embedding_dim"] == 1280
    assert info["render_dpi"] == 200
    # The metrics baked into the artifact are the published champion numbers.
    assert info["fitted_metrics"]["accuracy"] == pytest.approx(0.9140625)
    assert info["fitted_metrics"]["macro_f1"] == pytest.approx(0.9116853791632691)


# --- page images -----------------------------------------------------------
#
# _open_image and sniff are pure functions, so the pixel-level guarantees are
# asserted directly rather than through the stubbed model.


def test_sniff_identifies_each_accepted_format():
    from src.services.predictor import sniff

    assert sniff(make_pdf(1)) == "pdf"
    assert sniff(make_image("PNG")) == "png"
    assert sniff(make_image("JPEG")) == "jpeg"
    assert sniff(b"GIF89a not supported") == "unknown"
    assert sniff(b"") == "unknown"


def test_png_and_jpeg_are_scored_as_one_page():
    for fmt in ("PNG", "JPEG"):
        predictor = build_predictor([0.9])
        result = predictor.predict(make_image(fmt, size=(200, 300)), max_pages=100)
        assert result["document"]["n_pages"] == 1
        assert result["document"]["input_type"] == "image"
        assert result["pages"][0]["page"] == 1


def test_pdf_still_reports_input_type_pdf():
    predictor = build_predictor([0.9, 0.1])
    result = predictor.predict(make_pdf(2), max_pages=100)
    assert result["document"]["input_type"] == "pdf"
    assert result["document"]["n_pages"] == 2


def test_unsupported_bytes_raise_rather_than_guess():
    from src.services.predictor import UnsupportedFileTypeError

    predictor = build_predictor([0.9])
    with pytest.raises(UnsupportedFileTypeError):
        predictor.predict(b"GIF89a still not a document", max_pages=100)


def test_corrupt_image_raises_invalid_image():
    from src.services.predictor import InvalidImageError

    predictor = build_predictor([0.9])
    with pytest.raises(InvalidImageError):
        predictor.predict(make_image("PNG")[:120], max_pages=100)


def test_page_limit_does_not_apply_to_a_single_image():
    """max_pages guards PDF rasterization; one image is always one page."""
    predictor = build_predictor([0.9])
    result = predictor.predict(make_image("PNG", size=(200, 300)), max_pages=1)
    assert result["document"]["n_pages"] == 1


def test_transparent_png_is_flattened_onto_white_not_black():
    """A transparent PNG naively converted with .convert('RGB') goes black,
    the opposite of a document's ground. It must composite onto white."""
    from src.services.predictor import _open_image

    flattened = _open_image(make_image("PNG", size=(20, 20), mode="RGBA"))
    assert flattened.mode == "RGB"
    assert flattened.getpixel((0, 0)) == (255, 255, 255)


def test_exif_rotation_is_applied():
    """Scanner and phone JPEGs record rotation in EXIF, not in the pixels."""
    from src.services.predictor import _open_image

    upright = _open_image(make_image("JPEG", size=(400, 600)))
    rotated = _open_image(make_image("JPEG", size=(400, 600), exif_orientation=6))
    assert upright.size == (400, 600)
    assert rotated.size == (600, 400)  # transposed by exif_transpose


def test_absurdly_large_image_is_refused_before_embedding(monkeypatch):
    import src.services.predictor as module
    from src.services.predictor import InvalidImageError

    monkeypatch.setattr(module, "MAX_IMAGE_PIXELS", 1000)
    predictor = build_predictor([0.9])
    with pytest.raises(InvalidImageError, match="MP"):
        predictor.predict(make_image("PNG", size=(200, 200)), max_pages=100)
