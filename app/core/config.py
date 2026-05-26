"""Typed application settings loaded from environment / .env."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    host: str = "0.0.0.0"  # noqa: S104 — bind-all is correct inside a container
    port: int = 8000
    workers: int = 1

    # Auth — shared secret expected on every /api/* request
    ai_api_key: str = Field(default="dev-only-key", min_length=1)

    # Image module
    image_ai_backend: Literal["stub", "deepface", "mediapipe"] = "stub"
    image_model_weights_dir: str | None = None

    # Face-detection quality gates (apply to deepface / mediapipe backends).
    # Predictions are returned as null when no face passes these thresholds —
    # better to skip than to hallucinate on background pixels.
    #   yunet — OpenCV YuNet, upscaled + tiled; best on small ID-card photos
    #   retinaface / mtcnn / ssd / opencv — DeepFace's own detectors
    #   facenet — facenet-pytorch MTCNN
    image_face_detector: Literal[
        "yunet", "opencv", "ssd", "mtcnn", "retinaface", "facenet"
    ] = "yunet"
    # 0.75 suits the YuNet detector: its multi-scale scan lifts every genuine
    # ID-card face to ~0.88+ while stray boxes stay below ~0.70, so this
    # threshold cleanly separates them. Lower it if you switch to a DeepFace
    # detector whose real-face confidences run lower.
    image_min_face_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    # Inset photos on low-resolution ID-card uploads are genuinely tiny (the
    # face can be ~30px on an 1000px-wide card), so the floor is deliberately
    # low — confidence, not size, is the real quality gate.
    image_min_face_size_px: int = Field(default=24, ge=0)

    # Performance — large images (multi-MP phone photos) are downscaled so
    # the long edge is at most this many pixels before face detection runs.
    # retinaface on a 4000px image is several times slower than on a 1600px
    # one with no meaningful accuracy loss for the inset photos on OVD cards;
    # bounding boxes are scaled back to original-image coordinates.
    image_max_detect_edge: int = Field(default=1600, ge=320)

    # /ai/face-detect-crop padding — fraction of the detected face's
    # width/height added on each side. The raw detector box is tight and clips
    # the forehead / chin / ears; 0.4 gives a natural head-and-shoulders crop.
    image_face_crop_margin: float = Field(default=0.4, ge=0.0, le=1.5)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
