"""Backend factory — picks the concrete Predictor based on settings.

Cached so the heavy ML constructors only run once per process.
"""

from functools import lru_cache

from app.core.config import get_settings
from app.core.exceptions import ValidationError

from .base import Predictor


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    backend = get_settings().image_ai_backend
    if backend == "stub":
        from .stub import StubPredictor

        return StubPredictor()
    if backend == "deepface":
        from .deepface_impl import DeepFacePredictor

        return DeepFacePredictor()
    if backend == "mediapipe":
        from .mediapipe_impl import MediaPipePredictor

        return MediaPipePredictor()
    if backend == "mivolo":
        from .mivolo_impl import MiVOLOPredictor

        return MiVOLOPredictor()
    raise ValidationError(f"Unknown IMAGE_AI_BACKEND: {backend!r}")
