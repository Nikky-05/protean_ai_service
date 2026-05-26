# Architecture

A four-layer FastAPI app, organised by module so new business
capabilities slot in without touching shared code.

```
HTTP request
   │
   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  app/main.py                          ← FastAPI app, lifespan       │
│    ├─ middleware/                                                   │
│    │     correlation.py  ← X-Correlation-Id bind to log context     │
│    │     auth.py         ← Authorization header check               │
│    │     error_handler.py← AppError → JSON                          │
│    └─ api/v1/router.py                ← mounts every module router  │
│                                                                     │
│  app/modules/<name>/                  ← ONE folder per capability   │
│    router.py    ← thin HTTP layer                                   │
│    service.py   ← business logic (no FastAPI imports)               │
│    schemas.py   ← Pydantic request / response models                │
│    predictors/  ← (image-only) pluggable ML backends                │
│       base.py        ← abstract Predictor                           │
│       stub.py        ← deterministic mock                           │
│       deepface_impl  ← real model (optional dep)                    │
│       factory.py     ← config-driven selection                      │
│                                                                     │
│  app/core/                            ← cross-cutting primitives    │
│    config.py        ← typed Settings via pydantic-settings          │
│    logging.py       ← structlog JSON                                │
│    exceptions.py    ← AppError hierarchy                            │
│                                                                     │
│  app/utils/                           ← reusable helpers            │
│    image_io.py      ← base64/URL → PIL.Image                        │
└─────────────────────────────────────────────────────────────────────┘
   │
   ▼
JSON response
```

## Design principles

1. **One folder per module.** Everything a module owns — routes,
   schemas, service, predictors, README — lives under
   `app/modules/<name>/`. Cross-module imports are forbidden in
   review; share via `app/core/` or `app/utils/` instead.

2. **Routers are thin.** They marshal HTTP and nothing else. Real
   work goes in `service.py` so unit tests don't need FastAPI.

3. **Predictors are pluggable.** Every ML capability defines an
   abstract `Predictor` interface. Concrete backends
   (`stub` / `deepface` / `mediapipe` / future) implement it. The
   factory picks the right one from config. Swapping a model never
   touches the router.

4. **Optional ML deps.** Heavy dependencies (TensorFlow, OpenCV,
   MediaPipe) are NOT in `requirements.txt`. They are imported
   lazily inside the backend that needs them so a `stub` deployment
   builds in seconds.

5. **Structured logs + correlation IDs.** Every request gets an
   `X-Correlation-Id` (echoed back); structlog binds it to the
   context so all log lines emitted during the request are
   stitchable with the Node orchestrator's audit log.

6. **All errors → AppError.** Modules never raise `HTTPException`.
   They raise `ValidationError`, `ModelInferenceError`, etc., which
   the central handler maps to a consistent JSON envelope:

   ```json
   {
     "success": false,
     "error": { "code": "MODEL_INFERENCE_ERROR", "message": "...", "details": {} }
   }
   ```

## Why four endpoints, not one

See the PRD §4 / §8 alignment in the parent Node README — short
version: separate endpoints give per-sub-API billing (PRD §11),
tier-gated pricing (PRD §8), independent scaling, and independent
model deploys. The cost (4 HTTP calls vs 1) is wall-clock free
because the Node pipeline runs them in parallel.
