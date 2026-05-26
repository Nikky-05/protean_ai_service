# Adding a new business module

The repo is structured so a new AI capability — say
`signature-verification` — slots in with no changes to existing modules
or shared code.

## Step-by-step

### 1. Copy the template

```bash
cp -r app/modules/_template app/modules/signature
```

Rename references inside the copied files (`example` → your verbs).

You'll have:

```
app/modules/signature/
├── __init__.py
├── README.md      ← what this module does, endpoints, deps
├── schemas.py     ← Pydantic request / response models
├── service.py     ← business logic (no FastAPI imports)
└── router.py      ← HTTP layer
```

### 2. Register the router

Open `app/api/v1/router.py` and add two lines:

```python
from app.modules.signature.router import router as signature_router
api_v1.include_router(signature_router, prefix="/signature", tags=["signature"])
```

### 3. (If ML-backed) add a predictor subpackage

For modules that need a swappable ML backend, mirror the image
module's layout:

```
app/modules/signature/predictors/
├── __init__.py
├── base.py            ← abstract Predictor
├── stub.py            ← deterministic mock for dev/CI
├── <real>_impl.py     ← optional, heavy dep
└── factory.py         ← reads backend choice from app.core.config
```

Add the env var (e.g. `SIGNATURE_AI_BACKEND`) to
`app/core/config.py` as a `Literal[...]` field.

### 4. Tests

```
tests/modules/signature/
├── __init__.py
└── test_signature_routes.py
```

Use the shared fixtures in `tests/conftest.py` (`client`, etc.).
Don't import anything from `tests/modules/image/`.

### 5. Document

Update `app/modules/signature/README.md` (endpoint list, request /
response samples). If the module is part of an orchestrated product,
also reference the relevant PRD section so reviewers can audit
coverage.

## Rules of the road

- **No cross-module imports.** Modules talk to each other only via
  HTTP, never by importing each other's `service.py`. Shared code
  goes in `app/core/` or `app/utils/`.
- **Errors raise `AppError`,** never `HTTPException`. The central
  handler in `app/middleware/error_handler.py` maps them.
- **Async first.** Routers and services are `async def`. Sync ML
  calls go through `asyncio.to_thread` (see DeepFace backend).
- **Heavy deps are lazy.** `import deepface` (or similar) happens
  inside the backend class, not at module top-level — keep the stub
  deployment lean.
