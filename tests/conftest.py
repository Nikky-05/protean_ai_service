"""Shared pytest fixtures.

The auth dependency is overridden so tests don't need to send the
shared secret on every request. The predictor cache is reset between
tests so each test gets a fresh stub instance.
"""

import base64
import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Force the stub backend BEFORE importing the app — Settings is cached.
os.environ.setdefault("IMAGE_AI_BACKEND", "stub")
os.environ.setdefault("AI_API_KEY", "test-key")

from app.main import create_app  # noqa: E402
from app.middleware.auth import require_api_key  # noqa: E402
from app.modules.image.predictors.factory import get_predictor  # noqa: E402


@pytest.fixture
def app():
    get_predictor.cache_clear()
    application = create_app()
    application.dependency_overrides[require_api_key] = lambda: None
    return application


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def jpeg_b64() -> str:
    """A tiny solid-colour JPEG, base64-encoded — enough for the stub backend."""
    img = Image.new("RGB", (64, 64), color=(120, 200, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
