"""Unit tests for the stub predictor — no HTTP layer involved."""

import pytest
from PIL import Image

from app.modules.image.predictors.stub import StubPredictor


@pytest.fixture
def predictor() -> StubPredictor:
    return StubPredictor()


@pytest.fixture
def img() -> Image.Image:
    return Image.new("RGB", (128, 128), color=(50, 50, 200))


async def test_face_detect_crop(predictor, img):
    r = await predictor.face_detect_crop(img)
    assert r.face_count >= 0
    assert r.face_detected is (r.face_count > 0)
    # One cropped face per detected face; each carries a base64 crop + bbox.
    assert len(r.faces) == r.face_count
    for face in r.faces:
        assert face.face_b64
        assert len(face.bbox) == 4
        # centre-crop is 60% of the shortest side → side=76 on a 128px image
        assert face.bbox[2] == 76 and face.bbox[3] == 76


async def test_age_predict_within_bounds(predictor, img):
    r = await predictor.age_predict(img)
    assert 18 <= r.age <= 67
    assert r.age_range in {"18-25", "26-35", "36-50", "51+"}


async def test_gender_predict_valid_label(predictor, img):
    r = await predictor.gender_predict(img)
    assert r.gender in {"M", "F", "Other"}
