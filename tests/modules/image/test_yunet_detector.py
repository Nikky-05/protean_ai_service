"""Unit tests for the YuNet face detector.

Detection accuracy is validated against real ID-card images during
development; here we cover the building blocks (NMS / IoU) and the
no-face path, which together exercise model loading, the orientation
loop and the tiled fallback.
"""

import pytest
from PIL import Image

pytest.importorskip("cv2")

from app.modules.image.predictors.yunet_detector import (  # noqa: E402
    DetectedFace,
    YuNetFaceDetector,
    _iou,
    _nms,
)


@pytest.fixture
def detector() -> YuNetFaceDetector:
    return YuNetFaceDetector(min_confidence=0.75, min_size_px=24, detect_edge=1600)


def test_blank_image_yields_no_face(detector):
    """A plain image must report no face — and not raise from the tiled path."""
    angle, faces = detector.detect(Image.new("RGB", (320, 200), (245, 245, 245)))
    assert faces == []
    assert angle == 0


def test_nms_drops_overlapping_boxes():
    a = DetectedFace(0, 0, 100, 100, 0.9)
    b = DetectedFace(6, 6, 100, 100, 0.8)  # heavily overlaps `a`
    c = DetectedFace(500, 500, 80, 80, 0.7)  # disjoint
    kept = _nms([b, a, c])
    assert len(kept) == 2
    assert a in kept and c in kept  # highest-confidence box of the pair survives


def test_iou_disjoint_is_zero_overlap_is_high():
    a = DetectedFace(0, 0, 10, 10, 1.0)
    assert _iou(a, DetectedFace(100, 100, 10, 10, 1.0)) == 0.0
    assert _iou(a, DetectedFace(0, 0, 10, 10, 1.0)) == pytest.approx(1.0)
