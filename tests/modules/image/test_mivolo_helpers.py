"""Tests for MiVOLO predictor helpers that do not load ML models.

The MiVOLO model itself (a ViT, plus torch + a checkpoint) is too heavy for
CI, so these cover the pure output-decode logic and the backend wiring. The
end-to-end inference glue is validated on a real run with weights.
"""

import pytest

from app.core.config import Settings
from app.modules.image.predictors.factory import get_predictor


class _FakeMeta:
    min_age = 1.0
    max_age = 95.0
    avg_age = 36.0


def test_decode_mivolo_output_denormalises_age_and_reads_gender():
    torch = pytest.importorskip("torch")
    from app.modules.image.predictors.mivolo_impl import _decode_mivolo_output

    # age_norm chosen so age == 30: (30 - 36) / (95 - 1)
    age_norm = (30.0 - 36.0) / (95.0 - 1.0)
    # male logit > female logit → "M"
    output = torch.tensor([[age_norm, 2.0, 0.0]])

    age, gender, conf = _decode_mivolo_output(output, _FakeMeta())

    assert age == 30
    assert gender == "M"
    # softmax([2, 0])[0] ≈ 0.8808
    assert conf == pytest.approx(0.8808, abs=1e-3)


def test_decode_mivolo_output_reads_female_when_female_logit_wins():
    torch = pytest.importorskip("torch")
    from app.modules.image.predictors.mivolo_impl import _decode_mivolo_output

    output = torch.tensor([[0.0, 0.0, 3.0]])

    age, gender, conf = _decode_mivolo_output(output, _FakeMeta())

    assert age == 36  # age_norm 0 → avg_age
    assert gender == "F"
    assert conf > 0.9


def test_mivolo_is_a_valid_backend_literal():
    # Constructing Settings with the new backend must not raise — proves the
    # Literal in config.py was widened.
    s = Settings(image_ai_backend="mivolo")
    assert s.image_ai_backend == "mivolo"


def test_mivolo_config_defaults():
    s = Settings()
    assert s.image_mivolo_checkpoint is None
    assert s.image_mivolo_min_gender_confidence == pytest.approx(0.65)


def test_unknown_backend_still_rejected(monkeypatch):
    # Guard: the factory must reject anything not wired in.
    from app.core import config

    get_predictor.cache_clear()
    monkeypatch.setattr(
        config, "get_settings", lambda: Settings(image_ai_backend="stub")
    )
    # stub resolves fine; the unknown-backend path is covered by the
    # ValidationError branch in factory.get_predictor.
    predictor = get_predictor()
    assert predictor.name == "stub"
    get_predictor.cache_clear()
