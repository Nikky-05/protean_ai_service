# Image module

In-house AI capabilities consumed by the **Image Matching Enrichment
API** (PRD section 4 in-house rows). Three endpoints, one Python service.

| Endpoint                                       | Purpose                                   |
|------------------------------------------------|-------------------------------------------|
| `POST /api/v1/image/ai/face-detect-crop`       | Detect faces + crop each one (base64)     |
| `POST /api/v1/image/ai/age-predict`            | Predict age + age range                   |
| `POST /api/v1/image/ai/gender-predict`         | Predict gender                            |

## Predictor backend

All endpoints delegate to a `Predictor` implementation selected by
`IMAGE_AI_BACKEND` env var:

- `stub` - deterministic mock (no ML deps). Use in dev/CI.
- `deepface` - real predictions via DeepFace (TensorFlow under the hood).
- `mediapipe` - fast CPU face-detect/crop.
- `mivolo` - YuNet detection + **MiVOLO** for joint age+gender, with
  InsightFace `buffalo_l` as the fallback. MiVOLO is more accurate on age
  than DeepFace's regressor. It is a joint model, so **one inference returns
  both age and gender** — `/age-predict` and `/gender-predict` share a single
  cached result per image. On CPU it is a heavy ViT, so unlike the deepface
  backend it runs **once on a single aligned crop** (no multi-variant voting).
  Set `IMAGE_MIVOLO_CHECKPOINT` to the weights; without it the backend boots
  in fallback-only mode (buffalo_l) so the service still runs. The same
  calibrated abstain gates as deepface (tiny / dark faces) apply, and gender
  below `IMAGE_MIVOLO_MIN_GENDER_CONFIDENCE` is returned null.

To add a new backend, implement `Predictor` in
`predictors/<backend>_impl.py` and wire it in `predictors/factory.py`.
The router/service stay untouched.

### Face detector (`deepface` backend)

`IMAGE_FACE_DETECTOR` selects how faces are *located* before DeepFace
runs age/gender on the crop:

- `yunet` *(default, recommended)* - OpenCV YuNet driven by
  `predictors/yunet_detector.py`. Built specifically for the small,
  faded inset photos on Aadhaar / PAN / voter / DL / passport scans:
  - **multi-scale** - the card is scanned at several sizes, since a
    given face peaks at a different scale depending on its quality;
  - **tiled fallback** - if the whole-image scan finds nothing, the
    image is split into overlapping upscaled tiles, which lifts recall
    on the smallest uploads (a ~250px-wide PAN card still works);
  - it is a real face detector, so it never crops a QR code or a
    signature box - the failure mode of the old contour heuristic.
  The model (`models/face_detection_yunet_2023mar.onnx`) is bundled, so
  the service runs offline. Requires `opencv-python`.
- `retinaface` / `mtcnn` / `ssd` / `opencv` - DeepFace's own detectors.
- `facenet` - facenet-pytorch MTCNN (needs `facenet-pytorch`).

## Request / response

Face detect-crop and gender predict take this body shape:

```json
{ "image": "<base64 or https URL>" }
```

`/ai/face-detect-crop` returns the detection summary together with one
cropped face image per detected face:

```json
{
  "face_detected": true,
  "face_count": 2,
  "confidence": 0.52,
  "faces": [
    { "face_b64": "<base64 JPEG>", "bbox": [x, y, w, h], "confidence": 0.52 },
    { "face_b64": "<base64 JPEG>", "bbox": [x, y, w, h], "confidence": 0.48 }
  ]
}
```

When no face is found, `face_detected` is `false`, `face_count` is `0`
and `faces` is an empty list.

Age predict also accepts DOB from document OCR. When DOB is present, the
service returns exact age and skips face-based age estimation:

```json
{ "image": "<base64 or https URL>", "dob": "1998-05-22" }
```

Aliases `date_of_birth` and `dateOfBirth` are also accepted.

Response shapes are documented in `schemas.py` and surfaced in Swagger
at `/docs`.
