# Protean AI Services

In-house Python service that hosts AI/ML capabilities consumed by the
Protean orchestrated APIs (RISE platform). Currently exposes the
endpoints required by the **Image Matching Enrichment API** (PRD §4
in-house modules):

- `POST /api/v1/image/ai/face-detect-crop` — detect faces and return each cropped face (base64)
- `POST /api/v1/image/ai/age-predict`
- `POST /api/v1/image/ai/gender-predict`

The repository is structured so additional business modules
(e.g. signature-verification, document-quality, voice-biometrics) can be
added by dropping a new package under `app/modules/<name>/` — see
[`docs/ADDING_A_MODULE.md`](docs/ADDING_A_MODULE.md).

---

## Quick start

```bash
# 1. Create a venv
python -m venv .venv
source .venv/bin/activate            # POSIX
.venv\Scripts\activate               # Windows PowerShell

# 2. Install dependencies (stub backend — no ML libs required)
pip install -r requirements.txt

# 3. Copy and edit env
cp .env.example .env

# 4. Run
uvicorn app.main:app --reload --port 8000
```

OpenAPI docs are served at `http://localhost:8000/docs` (Swagger) and
`/redoc` once the server is up.

## Predictor backends

ML implementations are pluggable. The backend is selected by
`IMAGE_AI_BACKEND` in `.env`:

| Value      | Use when                                         | Extra deps                          |
|------------|--------------------------------------------------|-------------------------------------|
| `stub`     | Local dev, unit tests, CI — no GPU / no weights  | None                                |
| `deepface` | Production — real predictions for age/gender     | `deepface`, `tf-keras`, `opencv-python` |
| `mediapipe`| Production — fast CPU face-detect/crop           | `pip install mediapipe`             |

On the `deepface` backend, `IMAGE_FACE_DETECTOR` chooses how faces are
located. The default, `yunet`, is an OpenCV YuNet detector tuned for the
small inset photos on Indian ID cards (Aadhaar / PAN / voter / DL /
passport) — multi-scale + tiled so even low-resolution uploads work, and
it never mistakes a QR code for a face. See
[`app/modules/image/README.md`](app/modules/image/README.md#face-detector-deepface-backend).

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the layered
breakdown and [`app/modules/image/predictors/base.py`](app/modules/image/predictors/base.py)
for the abstract interface every backend implements.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

## Docker

```bash
docker compose up --build
```

## Integration with the Node orchestrator

The Node side (`protean-apigee-api`) calls this service via these env
vars:

```
IMAGE_AI_BASE_URL=http://protean-ai-services:8000
IMAGE_AI_API_KEY=<shared-secret>
```

The Node connectors are in
`src/modules/image/connectors/imageVendorConnectors.js`
(`FaceDetectCropConnector`, `AgePredictConnector`,
`GenderPredictConnector`). Face detection and cropping are now served by
a single `/ai/face-detect-crop` endpoint, so the Node side should call
that one path instead of the former `/ai/face-detect` + `/ai/face-crop`
pair.
