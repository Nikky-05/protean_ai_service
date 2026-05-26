"""OpenCV YuNet face detector — tuned for Indian ID-card photos.

Why a dedicated detector
------------------------
On Aadhaar / PAN / voter / DL / passport scans the real photo of the
person is a small, often faded inset on a busy card. The DeepFace
detectors (retinaface / mtcnn / facenet / its own `yunet` wrapper) all
struggled here in testing:

  * they miss the small inset photo entirely on low-resolution uploads
    (a PAN card photographed at ~450px wide → a ~70px face), and
  * the contour-based "document photo" fallback that compensated for the
    misses fired on QR codes and signature boxes — cropping a QR block
    instead of the face.

This module drives OpenCV's YuNet model (`cv2.FaceDetectorYN`) directly.
It is fast (CPU, tens of milliseconds) and, being a real face detector,
never reports a QR code or a line of text as a face. Two techniques make
it reliable on cards:

  * **Upscaling.** Every image is scaled so its long edge is roughly
    ``detect_edge`` px before detection — a tiny inset face is far easier
    to find once it is a few hundred pixels tall.

  * **Tiled fallback.** When whole-image detection finds nothing, the
    image is split into overlapping tiles, each upscaled and re-scanned.
    A face that is ~6% of the whole card becomes ~30%+ of a tile, which
    lifts recall sharply on the hardest low-quality uploads. Because
    YuNet only ever reports real faces, tiling cannot reintroduce
    QR-code / signature false positives.

Orientation: scans / screenshots are often sideways or upside-down with
no EXIF tag, so detection is retried at 90° / 270° / 180° when the image
as-is yields nothing.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.core.exceptions import ModelInferenceError
from app.core.logging import get_logger

log = get_logger(__name__)

_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
# Repo-bundled copy so the service works with no network access.
_BUNDLED_MODEL = os.path.join(os.path.dirname(__file__), "models", _MODEL_FILENAME)

# Orientations tried, in order, when the image as-is yields no face:
# 0 first (the common case), then the two 90° rotations, then upside-down.
_ORIENTATIONS = (0, 90, 270, 180)

# Each tile is upscaled so its long edge is roughly this many pixels.
_TILE_EDGE = 800
# Target span of one (pre-overlap) tile cell in original-image pixels.
_TILE_SPAN = 380
# IoU above which two boxes are treated as the same face (NMS).
_NMS_IOU = 0.3


@dataclass(frozen=True)
class DetectedFace:
    """A face box in the coordinates of some oriented image."""

    x: int
    y: int
    w: int
    h: int
    confidence: float


def _resolve_model_path(weights_dir: str | None) -> str:
    """Locate the YuNet ONNX file, downloading it once if necessary."""
    if weights_dir:
        candidate = os.path.join(weights_dir, _MODEL_FILENAME)
        if os.path.isfile(candidate):
            return candidate
    if os.path.isfile(_BUNDLED_MODEL):
        return _BUNDLED_MODEL
    # Last resort: fetch into the bundled location so subsequent boots are offline.
    os.makedirs(os.path.dirname(_BUNDLED_MODEL), exist_ok=True)
    log.info("yunet_model_downloading", url=_MODEL_URL)
    try:
        urllib.request.urlretrieve(_MODEL_URL, _BUNDLED_MODEL)  # noqa: S310 — fixed HTTPS URL
    except Exception as e:  # noqa: BLE001
        raise ModelInferenceError(
            f"YuNet model is missing and could not be downloaded: {e}"
        ) from e
    return _BUNDLED_MODEL


def _iou(a: DetectedFace, b: DetectedFace) -> float:
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _nms(faces: list[DetectedFace]) -> list[DetectedFace]:
    """Drop overlapping boxes — keeps the highest-confidence one per face.

    Tiled detection sees the same face in several overlapping tiles, so
    de-duplication is required before the count is reported to callers.
    """
    kept: list[DetectedFace] = []
    for face in sorted(faces, key=lambda f: f.confidence, reverse=True):
        if all(_iou(face, k) < _NMS_IOU for k in kept):
            kept.append(face)
    return kept


class YuNetFaceDetector:
    """Whole-image + tiled YuNet detection with orientation retry."""

    def __init__(
        self,
        min_confidence: float,
        min_size_px: int,
        detect_edge: int,
        weights_dir: str | None = None,
    ) -> None:
        try:
            import cv2
        except ImportError as e:
            raise ModelInferenceError(
                "opencv-python is not installed. Run `pip install opencv-python` "
                "to use IMAGE_FACE_DETECTOR=yunet."
            ) from e
        if not hasattr(cv2, "FaceDetectorYN"):
            raise ModelInferenceError(
                "Installed OpenCV is too old for YuNet — upgrade to opencv-python>=4.8."
            )
        self._cv2 = cv2
        self._model_path = _resolve_model_path(weights_dir)
        self._min_conf = min_confidence
        self._min_px = min_size_px
        self._edge = detect_edge
        # Tiles run at a slightly stricter score: many small crops are scanned,
        # so a higher bar keeps stray low-confidence boxes out of the results.
        self._tile_conf = max(min_confidence, 0.7)
        log.info("yunet_ready", model=os.path.basename(self._model_path))

    # ─── Low-level YuNet call ─────────────────────────────────────────────

    def _run_yunet(self, image: Image.Image, score: float) -> list[DetectedFace]:
        """Run YuNet once on `image` (coordinates are `image`'s own pixels)."""
        w, h = image.size
        if w < 20 or h < 20:
            return []
        bgr = self._cv2.cvtColor(np.asarray(image), self._cv2.COLOR_RGB2BGR)
        detector = self._cv2.FaceDetectorYN.create(
            self._model_path, "", (w, h), score, 0.3, 5000
        )
        detector.setInputSize((w, h))
        _, faces = detector.detect(bgr)
        if faces is None:
            return []
        out: list[DetectedFace] = []
        for f in faces:
            x, y, fw, fh = (int(round(v)) for v in f[:4])
            x, y = max(0, x), max(0, y)
            fw, fh = min(fw, w - x), min(fh, h - y)
            if fw > 0 and fh > 0:
                out.append(DetectedFace(x, y, fw, fh, float(f[14])))
        return out

    # ─── Scaling helpers ──────────────────────────────────────────────────

    @staticmethod
    def _scale_to(image: Image.Image, target_edge: int) -> tuple[Image.Image, float]:
        """Resize so the long edge is ~`target_edge` (upscales small images too)."""
        w, h = image.size
        scale = target_edge / max(w, h)
        if abs(scale - 1.0) < 0.02:
            return image, 1.0
        resample = Image.LANCZOS if scale > 1 else Image.BILINEAR
        new = (max(1, round(w * scale)), max(1, round(h * scale)))
        return image.resize(new, resample), scale

    @staticmethod
    def _orient(image: Image.Image, angle: int) -> Image.Image:
        """Return the image rotated clockwise by `angle` (0/90/180/270)."""
        if angle == 0:
            return image
        return image.rotate(-angle, expand=True)

    # ─── Detection passes ─────────────────────────────────────────────────

    def _scales(self) -> tuple[int, ...]:
        """Long-edge targets for the multi-scale whole-image scan.

        YuNet's confidence for a given face is sensitive to how large that
        face is in the input: a faded inset photo may peak when the card is
        upscaled, while a noisy one peaks when it is scaled down. Scanning a
        few sizes and unioning the results catches both.
        """
        return (round(self._edge * 0.7), self._edge, round(self._edge * 1.4))

    def _detect_whole(self, image: Image.Image) -> list[DetectedFace]:
        """Scan the whole image at several scales; union + de-duplicate."""
        found: list[DetectedFace] = []
        for target in self._scales():
            scaled, scale = self._scale_to(image, target)
            inv = 1.0 / scale
            found.extend(
                DetectedFace(
                    round(f.x * inv), round(f.y * inv),
                    round(f.w * inv), round(f.h * inv), f.confidence,
                )
                for f in self._run_yunet(scaled, self._min_conf)
            )
        return _nms(found)

    def _tile_boxes(self, w: int, h: int) -> list[tuple[int, int, int, int]]:
        """Overlapping tile rectangles covering a `w`×`h` image."""
        cols = max(2, round(w / _TILE_SPAN))
        rows = max(2, round(h / _TILE_SPAN))
        tw, th = w / cols, h / rows
        boxes: list[tuple[int, int, int, int]] = []
        row = 0
        while row < rows:
            col = 0
            while col < cols:
                x, y = col * tw, row * th
                # 50% overlap so a face straddling a cell edge stays whole
                # inside at least one tile.
                boxes.append(
                    (int(x), int(y),
                     int(min(w, x + tw * 1.5)), int(min(h, y + th * 1.5)))
                )
                col += 1
            row += 1
        return boxes

    def _detect_tiled(self, image: Image.Image) -> list[DetectedFace]:
        """Fallback: scan overlapping upscaled tiles, then de-duplicate."""
        w, h = image.size
        found: list[DetectedFace] = []
        for x0, y0, x1, y1 in self._tile_boxes(w, h):
            region = image.crop((x0, y0, x1, y1))
            scaled, scale = self._scale_to(region, _TILE_EDGE)
            inv = 1.0 / scale
            for f in self._run_yunet(scaled, self._tile_conf):
                found.append(
                    DetectedFace(
                        x0 + round(f.x * inv), y0 + round(f.y * inv),
                        round(f.w * inv), round(f.h * inv), f.confidence,
                    )
                )
        return _nms(found)

    def _gate(self, faces: list[DetectedFace]) -> list[DetectedFace]:
        """Keep faces that clear the confidence + minimum-size thresholds."""
        return [
            f
            for f in faces
            if f.confidence >= self._min_conf
            and f.w >= self._min_px
            and f.h >= self._min_px
        ]

    # ─── Public API ───────────────────────────────────────────────────────

    def detect(self, image: Image.Image) -> tuple[int, list[DetectedFace]]:
        """Detect every face in `image`.

        Returns ``(angle, faces)`` — the orientation (0/90/180/270) the
        faces were found in, and their boxes in that oriented image's
        coordinates. ``(0, [])`` when no face is found.
        """
        # Round 1 — cheap whole-image scan across orientations.
        for angle in _ORIENTATIONS:
            faces = self._gate(self._detect_whole(self._orient(image, angle)))
            if faces:
                return angle, _nms(faces)
        # Round 2 — only the hardest uploads reach here: tile + upscale.
        for angle in _ORIENTATIONS:
            faces = self._gate(self._detect_tiled(self._orient(image, angle)))
            if faces:
                return angle, _nms(faces)
        return 0, []
