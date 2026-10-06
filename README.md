# Guardrail Statement Classifier

Classifies a scanned bank-statement page (`mutasi rekening`) as **approve**
or **reject**, from the page image itself (layout/completeness cues — the
observed rejection reasons include things like `data_terpotong`, truncated
page content).

See `.claude/plans/shiny-watching-parnas.md` (or ask for a copy) for the full
design rationale, and `EXPERIMENTS.md` for the full log of every training
scenario run so far (backbones, probe vs fine-tune, 4-class relabeling,
resolution experiments) with results and what was learned from each. Summary
of what matters operationally:

## Data & encryption

- Source: `gs://gc-bribrain-dev-gcs-ocr-01/ocr-nilam/bank-statement/guardrail/{approve,reject}/*.pdf`
- Two independent encryption layers on every object:
  1. **GCS CMEK** (Cloud KMS) — transparent to any caller with object-read IAM, nothing to do here.
  2. **PDF-level password encryption** (AES-256, PDF /R 6) — `approve/` and
     `reject/` are each encrypted with their **own, different** password.
     Passwords are resolved at runtime from Secret Manager
     (`projects/edm-bribrain-dev-01/secrets/ocr-nilam-statement`, expected
     payload `{"approve": "...", "reject": "..."}`) — see `src/data/secrets.py`.
     Until that secret + an IAM grant (`roles/secretmanager.secretAccessor`)
     exist, set `GUARDRAIL_APPROVE_PASSWORD` / `GUARDRAIL_REJECT_PASSWORD` as
     env vars for local dev. **Never** hardcode a password in config or code.
- **Known open issue**: because approve/reject use different passwords, the
  password already reveals the label for this historical dataset. That's
  fine for building a training set, but it means this scheme **cannot**
  classify a genuinely new, not-yet-labeled document in production — you
  can't pick a password without already knowing the answer. `src/infer.py`
  requires the caller to supply the password explicitly rather than guessing,
  and this repo does not attempt to solve the "what password does a new
  document get" question — that needs a product/infra decision upstream
  (e.g. a third "pending" password, re-encrypted into approve/reject after a
  verdict is assigned).

## GCP credentials

Set `gcp.credentials_file` in `configs/train_config.yaml` to a service-account
JSON key path, or leave it `null` to use Application Default Credentials
(`gcloud auth application-default login`, or workload identity in a real
deployment). Whichever service account you use needs:
- `storage.objects.get` / `storage.objects.list` on the guardrail prefix
- `secretmanager.versions.access` on `ocr-nilam-statement` (once provisioned)

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Pull + decrypt + rasterize everything into a local manifest + image cache
scripts/build_manifest.sh

# 2. Train + compare: linear-probe baseline (CPU) then end-to-end fine-tune
#    (needs a working CUDA device — check `nvidia-smi` first)
scripts/run_train.sh
```

This trains and evaluates 4 combinations — {DiT, ResNet50} × {linear-probe,
fine-tune} — on the same held-out, group-stratified test split (split by
`source_document` so pages from one statement never straddle train/val/test).
Reports land in `runs/<name>/metrics.json`.

## Model

- **Backbones**: `microsoft/dit-base` (Document Image Transformer — pretrained
  on scanned documents, primary recommendation) and a ResNet50/EfficientNetV2-S
  baseline (ImageNet-pretrained, via `timm`).
- **Classifier heads**, both trained and compared per backbone:
  - **Linear probe**: frozen backbone embeddings → logistic regression /
    XGBoost. Fast, low overfitting risk at ~1,000 labeled pages — the floor
    every fine-tune has to beat.
  - **End-to-end fine-tune**: `Linear(features → 2)` + softmax on top of an
    unfrozen backbone, class-weighted cross-entropy loss.

## Serving API

A containerized HTTP service scores every page of an uploaded bank-statement
document — a **PDF**, or a single page image (**PNG**, **JPEG**). Source lives alongside the training code and reuses it directly
(`src/data/render.py`, `src/models/backbones.py`, `src/models/linear_probe.py`),
so the served path and the offline pipeline cannot drift apart.

```
main.py                  # uvicorn entrypoint shim -> src.main:app
config.yaml              # non-secret config, ${VAR:-default} interpolation
.env.template            # optional overrides only — no secrets
src/main.py              # app factory + lifespan
src/api/                 # routes, schemas
src/core/                # config, logging, startup validation
src/middleware/          # request id, content-type
src/services/            # predictor, GCS artifact download
tests/unit/              # contract + predictor tests (gate the image build)
```

### The model artifact

The champion probe used to exist only in memory — `train_probe()` returned a
live object that nothing saved, so every scoring run refit it from the labeled
manifest. `src/persist_probe.py` fixes that:

```bash
scripts/persist_probe.sh          # fit, verify, write, stage locally
```

It writes a single bundle holding the fitted logistic regression **and** the
frozen backbone's weights, so a container reproduces the exact model with no
HuggingFace hub access at runtime. Nothing is written unless (1) the refit
reproduces `runs/probe_efficientnet_b0_logreg/metrics.json` to 1e-9 and (2) the
reloaded bundle predicts identically to the in-memory model.

Published to
`gs://gc-bribrain-dev-gcs-ocr-01/ocr-nilam/bank-statement/guardrail/models/`;
the container downloads it at startup (weights are never baked into the image).

### Run it

```bash
scripts/nexus_build.sh dev-1 dev  # build (unit tests gate the image)
scripts/nexus_deploy.sh dev-1 dev # run on port 8090

curl -s -X POST http://localhost:8090/v1/predict -F "file=@statement.pdf" | jq
curl -s -X POST http://localhost:8090/v1/predict -F "file=@page.png"      | jq
curl -s -X POST http://localhost:8090/v1/predict -F "file=@page.jpg"      | jq
```

`GET /health/live`, `/health/ready` and `/health` cover liveness, readiness and
full status; `/docs` serves the OpenAPI page.

### Request and response

`POST /v1/predict`, `multipart/form-data`, field `file`. Accepts a **PDF**
(every page scored) or a **PNG/JPEG** page image (one page). One entry per page,
`page` 1-based to match the `source_page` column in `predictions.csv`. The
document verdict is **reject if any page rejects** — a guardrail exists to stop
unusable statements reaching OCR, and one bad page makes the statement unusable.

The response shape is identical for both input kinds; `document.input_type` is
`"pdf"` or `"image"`, and a page image always reports `n_pages: 1`.

```json
{
  "status_code": 200,
  "status_desc": "OK",
  "message": "Prediction successful",
  "data": {
    "document": {
      "filename": "PF24018IOR_mutasi_rekening.pdf", "input_type": "pdf",
      "n_pages": 7, "n_approve": 5, "n_reject": 2,
      "verdict": "reject", "verdict_rule": "any_page_reject"
    },
    "pages": [
      {"page": 1, "verdict": "approve", "proba_reject": 0.0421, "proba_approve": 0.9579},
      {"page": 2, "verdict": "reject",  "proba_reject": 0.8817, "proba_approve": 0.1183}
    ],
    "model": {
      "backbone": "efficientnet_b0", "head": "logreg",
      "artifact_version": "20260902T063954Z", "render_dpi": 200, "threshold": 0.5
    },
    "processing_time_ms": 1830.4
  },
  "error_code": null, "errors": null, "request_id": "9f2c1e4a7b3d4e8f"
}
```

Errors use the same envelope with `data: null` and an `error_code` of
`INVALID_FILE_TYPE`, `ENCRYPTED_PDF`, `FILE_TOO_LARGE`, `PAGE_LIMIT_EXCEEDED`
(400), `MODEL_NOT_LOADED` (503) or `INFERENCE_FAILED` (500).

### Accepted formats

The format is decided by **magic bytes**, never by the filename or the
client-supplied `Content-Type` — both are caller-controlled, and neither is
authenticated now that the endpoint is open. A PNG named `.pdf` is scored as an
image; a `.txt` gets `INVALID_FILE_TYPE` with the accepted list in `errors`.

A page image is not a second-class path: the cached PNGs in
`data_cache/test467_images/` *are* the offline pipeline's rasterization, so an
uploaded 200-dpi page image takes the same route the training data took. Over
25 real test_467 pages uploaded as PNG, labels agreed **25/25** with
`predictions.csv` at `max |delta P(reject)| = 2.6e-03` — tighter than the PDF path
(1.4e-02), because it skips the rasterization round-trip entirely.

Two things the image path handles that a rendered PDF page never needs:

- **EXIF orientation.** Phone and scanner JPEGs record rotation in metadata
  rather than in the pixels; without `exif_transpose` the model would score a
  sideways page.
- **Alpha.** A transparent PNG flattened with a plain `.convert("RGB")` turns
  transparent areas *black* — the opposite of a document's ground. It is
  composited onto white instead.

Images above 80 MP are refused (`INVALID_FILE_TYPE`) from the header, before
any pixels are decoded. For scale, a 200-dpi A4 page is about 3.7 MP.

### The endpoint is unauthenticated

`/v1/predict` takes no credentials — there is no API key, and any `X-API-Key`
header a caller still sends is ignored. Every endpoint is open to anyone who
can reach the port.

This was a deliberate decision, and it puts the access boundary entirely
outside this service. Two consequences to keep in view when deciding where to
run it: anyone who can reach the port can classify arbitrary PDFs, and each
request occupies an inference worker for roughly 600 ms per page, so an
unthrottled caller can saturate the service. Restrict it at the network layer
(bind to localhost, a private subnet, or an authenticating reverse proxy)
rather than relying on the application.

### Encrypted PDFs are refused, deliberately

The service accepts **plain, unencrypted PDFs only**. It never tries a password.
Per the *Known open issue* above, `approve/` and `reject/` are encrypted with
different passwords, so "which password opens this file" is a perfect label
oracle on historical documents — and useless on a genuinely new one. Callers
decrypt upstream and send plaintext.

### Parity with the offline pipeline

```bash
.venv/bin/python -m src.verify_parity --mode png            # no password needed
.venv/bin/python -m src.verify_parity --mode http --sample 20   # full path, needs the test467 password
```

Re-scores pages from `runs/probe_efficientnet_b0_logreg_test467/predictions.csv`
through the serving artifact and reports label agreement. Expect labels to
agree and probabilities to differ slightly — see EXPERIMENTS.md on the
probe-refit sensitivity.

## Security notes

- The service-account key used to explore this data during planning was
  pasted in plaintext in chat and should be rotated in IAM.
- Never commit real Secret Manager values, service-account keys, or PDF
  passwords to this repo. `data_cache/` and `runs/` are local-only artifacts
  and should stay out of version control.
