"""Tests for DeepFace predictor helpers that do not load ML models."""

import pytest
from PIL import Image

from app.modules.image.predictors.deepface_impl import (
    _is_probable_cropped_portrait,
    _select_gender_prediction,
    _select_insightface_gender,
)


def test_cropped_portrait_guard_accepts_small_visual_portrait():
    img = Image.new("RGB", (86, 120), color=(115, 100, 90))
    for x in range(20, 66):
        for y in range(18, 100):
            img.putpixel((x, y), (55 + (x % 25), 45 + (y % 35), 40))

    assert _is_probable_cropped_portrait(img) is True


def test_cropped_portrait_guard_rejects_landscape_document():
    img = Image.new("RGB", (700, 420), color=(245, 245, 245))

    assert _is_probable_cropped_portrait(img) is False


def test_cropped_portrait_guard_rejects_blank_crop():
    img = Image.new("RGB", (86, 120), color=(245, 245, 245))

    assert _is_probable_cropped_portrait(img) is False


def test_gender_selection_accepts_strong_fallback_signal():
    rows = [
        {"gender": {"Woman": 96.0, "Man": 4.0}, "dominant_gender": "Woman"},
        {"gender": {"Woman": 94.0, "Man": 6.0}, "dominant_gender": "Woman"},
    ]

    assert _select_gender_prediction(
        rows, min_model_conf=0.90, min_margin=0.25
    ) == ("F", 0.95)


def test_gender_selection_rejects_tiny_portrait_signal_under_90_percent():
    rows = [{"gender": {"Man": 83.74, "Woman": 16.26}, "dominant_gender": "Man"}]

    assert _select_gender_prediction(rows, min_model_conf=0.90, min_margin=0.25) == (
        None,
        0.0,
    )


def test_gender_selection_accepts_larger_portrait_signal_over_80_percent():
    rows = [{"gender": {"Woman": 83.74, "Man": 16.26}, "dominant_gender": "Woman"}]

    gender, confidence = _select_gender_prediction(
        rows, min_model_conf=0.80, min_margin=0.25
    )

    assert gender == "F"
    assert confidence == pytest.approx(0.8374)


def test_gender_selection_rejects_detected_face_below_70_percent():
    rows = [{"gender": {"Man": 69.0, "Woman": 31.0}, "dominant_gender": "Man"}]

    assert _select_gender_prediction(rows) == (None, 0.0)


def test_gender_selection_keeps_detected_face_signal():
    rows = [{"gender": {"Man": 83.74, "Woman": 16.26}, "dominant_gender": "Man"}]

    gender, confidence = _select_gender_prediction(rows)

    assert gender == "M"
    assert confidence == pytest.approx(0.8374)


def test_insightface_gender_selection_uses_weighted_agreement():
    votes = [("F", 0.92), ("F", 0.88), ("M", 0.20)]

    gender, confidence = _select_insightface_gender(votes)

    assert gender == "F"
    assert confidence == pytest.approx((1.8 / 2.0) * (2.0 / 3.0))


def test_insightface_gender_selection_rejects_split_votes():
    votes = [("F", 0.90), ("M", 0.88)]

    assert _select_insightface_gender(votes) == (None, 0.0)


def test_insightface_gender_selection_rejects_low_quality_weak_confidence():
    votes = [("M", 0.6085747480392456)]

    assert _select_insightface_gender(
        votes,
        min_confidence=0.75,
        min_agreement=0.75,
    ) == (None, 0.0)
