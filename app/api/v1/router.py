"""Top-level `/api/v1` router.

This file is the single registration point for every business module's
router. Adding a new module is a one-line change here (after creating
the module package under `app/modules/<name>/`).
"""

from fastapi import APIRouter, Depends

from app.middleware.auth import require_api_key
from app.modules.image.router import router as image_router

api_v1 = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])

# ---- Module registration ----
# Add new modules below. Each module owns its own URL sub-prefix.
api_v1.include_router(image_router, prefix="/image", tags=["image"])
