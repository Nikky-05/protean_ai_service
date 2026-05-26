"""DeepFace-backed predictor — production-grade.

Why this is structured the way it is
------------------------------------
On Indian OVDs (PAN, Aadhaar, DL) the actual photo of the person is a
small inset on a noisy background of text, logos, watermarks and color
banners. Running DeepFace's `analyze()` directly on the whole card has
two failure modes:

  1. The detector misses the inset photo, and with
     `enforce_detection=False` the classifier processes the whole card
     as if it were a face. Gender then reflects the *background* (orange
     banner, white margin, signature box) — not the person — giving
     confidently-wrong "M"/"F" labels.

  2. The detector finds a tiny low-quality crop. The age/gender models
     were trained on adult headshots and degrade sharply on sub-50px,
     B&W or faded photos.

To avoid both, every method here follows the same pipeline:

    detect once (configurable detector, default retinaface)
        → quality-gate (confidence ≥ MIN_CONF and w,h ≥ MIN_PX)
        → if no face survives: return null prediction with confidence 0.0
        → else: pass the aligned crop to analyze(detector_backend="skip")
          so DeepFace classifies the face directly without re-detecting

Robustness to imperfect documents
---------------------------------
OVD photos are routinely shot sideways or upside down. EXIF orientation
is already applied at decode time (`image_io._open_image`), but scans /
screenshots / re-saved images often have rotated *content* with no EXIF
tag. So when detection finds nothing in the image as-is, it retries at
90° / 270° / 180° before giving up.

Each /ai/face-detect-crop face image is taken from the full-resolution
image with a configurable margin (`IMAGE_FACE_CROP_MARGIN`) around the
detector box — the raw box is tight and clips the forehead / chin / ears.

Performance
-----------
The Node orchestrator fires three endpoints per image (face-detect-crop,
age-predict, gender-predict) and does NOT serialise them — they arrive
at this service within milliseconds of each other. Three measures keep
latency sane:

  * **Serialised inference.** TensorFlow grabs every CPU core per
    inference, so running several detections concurrently causes
    catastrophic context-switch thrashing (a 1-2s detection balloons to
    60s+). A single lock (`_ml_lock`) ensures only one model inference
    runs at a time — counter-intuitively much faster on a CPU box.
  * **Per-image detection cache.** Detection results are cached on pixel
    content. Combined with the lock + a double-checked re-read, the first
    of the concurrent calls for an image computes detection; the others
    wait on the lock and then hit the cache.
  * **Downscaling.** Images above `IMAGE_MAX_DETECT_EDGE` are shrunk
    before detection; bounding boxes are scaled back to original coords.

Models are pre-warmed on construction so the first user request doesn't
pay model-load time.

Install with `pip install deepface` and set `IMAGE_AI_BACKEND=deepface`.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import replace
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from app.core.config import get_settings
from app.core.exceptions import ModelInferenceError
from app.core.logging import get_logger
from app.utils.image_io import encode_image_b64

from .base import (
    AgePrediction,
    CroppedFace,
    FaceDetectCrop,
    GenderPrediction,
    Predictor,
    age_to_bucket,
)

log = get_logger(__name__)

_CACHE_MAX_ENTRIES = 32
_CROPPED_PORTRAIT_MIN_EDGE = 40
_CROPPED_PORTRAIT_MAX_EDGE = 512
# _CROPPED_PORTRAIT_CONFIDENCE = 0.40
_CROPPED_PORTRAIT_CONFIDENCE = 0.55
_GENDER_MIN_MODEL_CONF = 0.70
# _GENDER_CROPPED_PORTRAIT_MIN_MODEL_CONF = 0.80
# _GENDER_TINY_PORTRAIT_MIN_MODEL_CONF = 0.90
# _GENDER_LOW_QUALITY_MIN_MARGIN = 0.25
_GENDER_CROPPED_PORTRAIT_MIN_MODEL_CONF = 0.65
_GENDER_TINY_PORTRAIT_MIN_MODEL_CONF = 0.75
_GENDER_LOW_QUALITY_MIN_MARGIN = 0.15
_GENDER_TINY_PORTRAIT_MAX_SHORT_EDGE = 120
_INSIGHTFACE_MIN_CONFIDENCE = 0.55
_INSIGHTFACE_MIN_AGREEMENT = 0.60
# Below this *detected-face* edge (px), both gender models (insightface
# buffalo_l and DeepFace) classify ID-card photos unreliably and emit
# confident-but-wrong labels — e.g. a small Aadhaar portrait read as "M".
# Abstain (gender=None) instead of asserting a wrong gender. Observed on real
# Aadhaar samples: a ~30px detected face is misclassified, while a ~44px face
# is classified correctly once insightface gets the upscaled crop (see
# _insightface_input_variants). 40px is the cutoff between those. Does not
# apply to whole-image portrait uploads, whose bbox is the full image. Tune
# against more real ID samples.
_GENDER_MIN_FACE_EDGE_PX = 40
# Mean luminance (0-255) below which the face is too dark to gender reliably.
# Calibrated on real ID samples: a misclassified dark face measured ~45, a
# correctly-classified dim face ~60. 55 separates them. Only brightness is
# gated, not blur (the blurriest sample was classified correctly).
_GENDER_MIN_FACE_BRIGHTNESS = 55
# Upscale the detected-face crop to at least this short edge before handing it
# to insightface, so its detector/aligner get a usable face from tiny ID photos.
_INSIGHTFACE_CROP_MIN_EDGE = 224
# When the detected face's short edge is below this fraction of the image's
# short edge, treat it as a small inset (ID-card photo) and let insightface
# classify the upscaled crop alone, rather than mixing in a weak full-image
# detection that disagrees and forces an abstain.
_INSET_FACE_FRACTION = 0.33

# Orientations tried, in order, when the image as-is yields no face.
# 0 first (the common case), then the two 90° rotations, then upside-down.
_ORIENTATIONS = (0, 90, 270, 180)


def _is_probable_cropped_portrait(image: Image.Image) -> bool:
    """True when the input is likely already a face/portrait crop.

    The normal document pipeline intentionally returns null when no detector
    finds a face. Tiny uploaded profile photos are different: detectors can
    miss an 80-120px compressed face even though the whole image is the face.
    This guard keeps that fallback away from full-card/page images and blank
    crops.
    """
    w, h = image.size
    short_edge = min(w, h)
    long_edge = max(w, h)
    if short_edge < _CROPPED_PORTRAIT_MIN_EDGE or long_edge > _CROPPED_PORTRAIT_MAX_EDGE:
        return False

    aspect = w / max(h, 1)
    if not 0.50 <= aspect <= 1.35:
        return False

    gray = np.asarray(image.convert("L"))
    if float(gray.std()) < 8.0:
        return False

    gy, gx = np.gradient(gray.astype(np.float32))
    edge_density = float(((np.abs(gx) + np.abs(gy)) > 55).mean())
    # return edge_density <= 0.45
    return edge_density <= 0.55



def _gender_probabilities(row: dict[str, Any]) -> tuple[float, float]:
    probs = row.get("gender", {}) or {}
    man = float(probs.get("Man", probs.get("man", 0.0))) / 100.0
    woman = float(probs.get("Woman", probs.get("woman", 0.0))) / 100.0
    if man > 0.0 or woman > 0.0:
        return man, woman

    label = (row.get("dominant_gender") or "").lower()
    if label.startswith("m"):
        return 1.0, 0.0
    if label.startswith("w") or label.startswith("f"):
        return 0.0, 1.0
    return 0.0, 0.0


def _select_gender_prediction(
    rows: list[dict[str, Any]],
    *,
    min_model_conf: float = _GENDER_MIN_MODEL_CONF,
    min_margin: float = 0.0,
) -> tuple[str | None, float]:
    if not rows:
        return None, 0.0

    scores = [_gender_probabilities(row) for row in rows]
    man = sum(score[0] for score in scores) / len(scores)
    woman = sum(score[1] for score in scores) / len(scores)
    model_conf = max(man, woman)
    margin = abs(man - woman)

    if model_conf <= 0.0:
        return "Other", 0.0
    if model_conf < min_model_conf:
        return None, 0.0
    if margin < min_margin:
        return None, 0.0
    return ("M" if man >= woman else "F"), model_conf


def _select_insightface_gender(
    votes: list[tuple[str, float]],
    *,
    min_confidence: float = _INSIGHTFACE_MIN_CONFIDENCE,
    min_agreement: float = _INSIGHTFACE_MIN_AGREEMENT,
) -> tuple[str | None, float]:
    if not votes:
        return None, 0.0

    weighted = {"M": 0.0, "F": 0.0}
    total_weight = 0.0
    for label, score in votes:
        if label not in weighted:
            continue
        weight = max(0.0, min(1.0, float(score)))
        weighted[label] += weight
        total_weight += weight

    if total_weight <= 0.0:
        return None, 0.0

    gender = "M" if weighted["M"] >= weighted["F"] else "F"
    agreement = weighted[gender] / total_weight
    mean_detection_score = total_weight / len(votes)
    confidence = agreement * mean_detection_score

    if agreement < min_agreement or confidence < min_confidence:
        return None, 0.0
    return gender, confidence


class DeepFacePredictor(Predictor):
    name = "deepface"

    def __init__(self) -> None:
        try:
            from deepface import DeepFace
        except ImportError as e:
            raise ModelInferenceError(
                "deepface is not installed. Run `pip install deepface` or set "
                "IMAGE_AI_BACKEND=stub for local dev."
            ) from e

        s = get_settings()
        self._detector: str = s.image_face_detector
        self._min_conf: float = s.image_min_face_confidence
        self._min_px: int = s.image_min_face_size_px
        self._max_edge: int = s.image_max_detect_edge
        self._crop_margin: float = s.image_face_crop_margin
        self._facenet_mtcnn: Any | None = None
        self._yunet: Any | None = None
        self._insightface_gender_app: Any | None = None
        self._insightface_gender_failed = False

        if self._detector == "yunet":
            # YuNet (OpenCV) drives detection directly — far more reliable on
            # small printed ID-card photos than the DeepFace detectors, and it
            # never fires on QR codes / signatures. DeepFace is still used
            # below for the age / gender classifiers.
            from .yunet_detector import YuNetFaceDetector

            self._yunet = YuNetFaceDetector(
                min_confidence=self._min_conf,
                min_size_px=self._min_px,
                detect_edge=self._max_edge,
                weights_dir=s.image_model_weights_dir,
            )
        elif self._detector == "facenet":
            try:
                from facenet_pytorch import MTCNN
            except ImportError as e:
                raise ModelInferenceError(
                    "facenet_pytorch is not installed. Install it or use "
                    "IMAGE_FACE_DETECTOR=retinaface."
                ) from e
            self._facenet_mtcnn = MTCNN(
                image_size=224,
                margin=20,
                keep_all=True,
                post_process=False,
                select_largest=False,
                device="cpu",
            )

        # Per-image detection cache. The endpoint calls for one image
        # (face-detect-crop / age-predict / gender-predict) share a single
        # detection pass instead of running retinaface once per call.
        self._cache: OrderedDict[bytes, dict[str, Any]] = OrderedDict()
        self._cache_lock = threading.Lock()

        # Serialises ALL TensorFlow inference. TF uses every core per
        # inference; running them concurrently thrashes the CPU and turns a
        # 1-2s detection into 60s+. One inference at a time is far faster
        # here. Also makes the detection cache effective — concurrent calls
        # for the same image queue on this lock, then hit the warm cache.
        self._ml_lock = threading.Lock()

        self._warm_up(DeepFace)

    # ─── Internal helpers ─────────────────────────────────────────────────

    def _warm_up(self, deepface: Any) -> None:
        """Trigger weight downloads + model loads at startup, not on first request."""
        try:
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            # facenet and yunet drive detection outside DeepFace and load their
            # own weights in __init__ — only warm DeepFace's own detectors here.
            if self._detector not in ("facenet", "yunet"):
                deepface.extract_faces(
                    img_path=dummy, detector_backend=self._detector, enforce_detection=False
                )
            deepface.analyze(
                img_path=dummy,
                actions=["age", "gender"],
                enforce_detection=False,
                detector_backend="skip",
            )
            log.info("deepface_warmed", detector=self._detector)
        except Exception as e:  # noqa: BLE001 — warm-up is best-effort
            log.warning("deepface_warmup_failed", error=str(e))

    @staticmethod
    def _to_np(image: Image.Image) -> np.ndarray:
        return np.asarray(image)

    @staticmethod
    def _orient(image: Image.Image, angle: int) -> Image.Image:
        """Return the image rotated clockwise by `angle` degrees (0/90/180/270)."""
        if angle == 0:
            return image
        # PIL rotates counter-clockwise; negate so positive angle = clockwise.
        return image.rotate(-angle, expand=True)

    @staticmethod
    def _face_to_uint8(face_arr: np.ndarray) -> np.ndarray:
        """DeepFace returns float arrays in [0,1] — most analyzers want uint8."""
        if face_arr.dtype != np.uint8 and face_arr.max() <= 1.0:
            return (face_arr * 255).astype(np.uint8)
        return face_arr.astype(np.uint8) if face_arr.dtype != np.uint8 else face_arr

    def _prep_for_detection(self, image: Image.Image) -> tuple[np.ndarray, float]:
        """Downscale oversized images so detection stays fast.

        Returns (numpy array fed to the detector, scale) where
        scale = detection_size / original_size (≤ 1.0). Callers multiply
        detected coordinates by 1/scale to map back to original pixels.
        """
        w, h = image.size
        long_edge = max(w, h)
        if long_edge <= self._max_edge:
            return np.asarray(image), 1.0
        scale = self._max_edge / long_edge
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        return np.asarray(image.resize(new_size, Image.BILINEAR)), scale

    @staticmethod
    def _tensor_to_face_array(tensor: Any) -> np.ndarray:
        """Convert a facenet-pytorch MTCNN face tensor to uint8 HWC RGB."""
        arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
        return np.clip(arr, 0, 255).astype(np.uint8)

    @staticmethod
    def _fallback_face_array(image: Image.Image, x: int, y: int, w: int, h: int) -> np.ndarray:
        crop = image.crop((x, y, x + w, y + h)).resize((224, 224), Image.BILINEAR)
        return np.asarray(crop)

    def _detect_one_orientation_facenet(self, image: Image.Image) -> list[dict[str, Any]]:
        """Run facenet-pytorch MTCNN on one orientation.

        FaceNet/MTCNN tends to produce cleaner aligned crops than document-wide
        DeepFace detectors. We keep the bounding box for /face-detect-crop and
        the aligned 224px face tensor for age/gender analysis.
        """
        if self._facenet_mtcnn is None:
            raise ModelInferenceError("facenet detector was not initialised")

        try:
            boxes, probs = self._facenet_mtcnn.detect(image)
            aligned = self._facenet_mtcnn(image)
        except Exception as e:  # noqa: BLE001
            raise ModelInferenceError(f"facenet face detection failed: {e}") from e

        if boxes is None or probs is None:
            return []

        aligned_faces: list[np.ndarray] = []
        if aligned is not None:
            if getattr(aligned, "ndim", 0) == 3:
                aligned_faces = [self._tensor_to_face_array(aligned)]
            else:
                aligned_faces = [self._tensor_to_face_array(face) for face in aligned]

        iw, ih = image.size
        kept: list[dict[str, Any]] = []
        for idx, (box, prob) in enumerate(zip(boxes, probs, strict=False)):
            if prob is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            x = max(0, int(round(x1)))
            y = max(0, int(round(y1)))
            right = min(iw, int(round(x2)))
            lower = min(ih, int(round(y2)))
            w = max(0, right - x)
            h = max(0, lower - y)
            conf = float(prob)
            if conf < self._min_conf or w < self._min_px or h < self._min_px:
                continue
            face_arr = (
                aligned_faces[idx]
                if idx < len(aligned_faces)
                else self._fallback_face_array(image, x, y, w, h)
            )
            kept.append(
                {
                    "face": face_arr,
                    "confidence": conf,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
            )
        return kept

    def _detect_one_orientation(self, image: Image.Image) -> list[dict[str, Any]]:
        """Run the detector once on `image`; bounding boxes in `image`'s coords."""
        if self._detector == "facenet":
            return self._detect_one_orientation_facenet(image)

        from deepface import DeepFace

        arr, scale = self._prep_for_detection(image)
        try:
            faces = DeepFace.extract_faces(
                img_path=arr,
                detector_backend=self._detector,
                enforce_detection=False,
                align=True,
            )
        except Exception as e:  # noqa: BLE001
            raise ModelInferenceError(f"face detection failed: {e}") from e

        inv = 1.0 / scale if scale > 0 else 1.0
        kept: list[dict[str, Any]] = []
        for f in faces:
            conf = float(f.get("confidence", 0))
            region = f.get("facial_area", {}) or {}
            # Scale the box back to original-image coordinates and quality-gate
            # on the original size, not the downscaled size.
            w = int(round(int(region.get("w", 0)) * inv))
            h = int(round(int(region.get("h", 0)) * inv))
            if conf >= self._min_conf and w >= self._min_px and h >= self._min_px:
                kept.append(
                    {
                        "face": f["face"],
                        "confidence": conf,
                        "x": int(round(int(region.get("x", 0)) * inv)),
                        "y": int(round(int(region.get("y", 0)) * inv)),
                        "w": w,
                        "h": h,
                    }
                )
        return kept

    def _document_photo_face_candidates(self, image: Image.Image) -> list[dict[str, Any]]:
        """Find portrait-photo blocks on document images and infer a face box.

        Aadhaar/PAN-style documents often contain a tiny printed photo that
        generic face detectors miss, while QR codes and signatures produce
        false positives. This fallback searches the lower-left document region
        for a photo-like rectangle and returns a conservative upper-face crop
        from inside it.
        """
        try:
            import cv2
        except ImportError:
            return []

        arr = np.asarray(image)
        ih, iw = arr.shape[:2]
        if iw < 80 or ih < 80:
            return []

        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # Search the lower-left/left-middle region where document portrait
        # photos are usually printed. This intentionally excludes Aadhaar QR
        # blocks on the right side.
        y0 = int(ih * 0.45)
        x1 = int(iw * 0.55)
        roi_h = ih - y0
        roi_w = x1
        if roi_h <= 0 or roi_w <= 0:
            return []

        roi_gray = gray[y0:ih, 0:x1]
        roi_hsv = hsv[y0:ih, 0:x1]
        sat = roi_hsv[:, :, 1]
        val = roi_hsv[:, :, 2]

        # Non-white, non-background pixels. Printed photos usually form a
        # compact rectangle with dark/colored content; plain white paper does
        # not. Text alone is filtered later by size/fill/aspect constraints.
        mask = (((roi_gray < 205) | ((sat > 35) & (val < 245))).astype(np.uint8)) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[float, int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            abs_y = y + y0
            area = w * h
            if area <= 0:
                continue
            aspect = w / max(h, 1)
            fill = cv2.contourArea(contour) / area
            if not (0.06 * iw <= w <= 0.28 * iw):
                continue
            if not (0.055 * ih <= h <= 0.22 * ih):
                continue
            if not (0.50 <= aspect <= 1.35):
                continue
            if fill < 0.18:
                continue
            if abs_y < ih * 0.50:
                continue

            # Prefer lower-left rectangles with meaningful area. This nudges
            # the Aadhaar portrait over logos, dates, or small text clusters.
            lower_bonus = abs_y / ih
            left_bonus = 1.0 - (x / max(roi_w, 1))
            score = area * (0.55 + lower_bonus) * (0.75 + left_bonus) * min(fill, 0.8)
            candidates.append((score, x, abs_y, w, h))

        if not candidates:
            return []

        candidates.sort(reverse=True)
        faces: list[dict[str, Any]] = []
        for _, px, py, pw, ph in candidates[:2]:
            # The printed portrait includes shoulders/background. Infer a
            # tighter face/head box from its upper center, then let the common
            # margin cropper add natural context.
            fw = int(round(pw * 0.72))
            fh = int(round(ph * 0.72))
            fx = int(round(px + (pw - fw) / 2))
            fy = int(round(py + ph * 0.04))
            fw = min(fw, iw - fx)
            fh = min(fh, ih - fy)
            if fw < max(20, self._min_px // 2) or fh < max(20, self._min_px // 2):
                continue
            face_img = image.crop((fx, fy, fx + fw, fy + fh)).resize((224, 224), Image.BILINEAR)
            faces.append(
                {
                    "face": np.asarray(face_img),
                    "confidence": max(self._min_conf, 0.72),
                    "x": fx,
                    "y": fy,
                    "w": fw,
                    "h": fh,
                    "source": "document_photo",
                }
            )
        return faces

    def _run_detection_yunet(self, image: Image.Image) -> dict[str, Any]:
        """Detect via the YuNet detector and adapt to the common result shape.

        YuNet handles orientation / upscaling / tiling internally; here we
        just attach the aligned 224px crop each face needs for the downstream
        age / gender classifiers.
        """
        assert self._yunet is not None
        started = time.perf_counter()
        angle, detected = self._yunet.detect(image)
        oriented = self._orient(image, angle)
        iw, ih = oriented.size
        faces: list[dict[str, Any]] = []
        for f in detected:
            right, lower = min(iw, f.x + f.w), min(ih, f.y + f.h)
            crop = oriented.crop((f.x, f.y, right, lower)).resize((224, 224), Image.BILINEAR)
            faces.append(
                {
                    "face": np.asarray(crop),
                    "confidence": f.confidence,
                    "x": f.x,
                    "y": f.y,
                    "w": f.w,
                    "h": f.h,
                }
            )
        log.info(
            "face_detection_complete" if faces else "face_detection_no_face",
            detector="yunet",
            angle=angle,
            faces_kept=len(faces),
            ms=round((time.perf_counter() - started) * 1000),
        )
        return {"angle": angle, "faces": faces}

    def _run_detection(self, image: Image.Image) -> dict[str, Any]:
        """Detect faces, retrying rotations so sideways OVD scans still work.

        Returns {"angle": <orientation that worked>, "faces": [...]} with
        face boxes expressed in the coordinates of the rotated image.
        """
        if self._detector == "yunet":
            return self._run_detection_yunet(image)

        started = time.perf_counter()
        for angle in _ORIENTATIONS:
            oriented = self._orient(image, angle)
            kept = self._detect_one_orientation(oriented)
            doc_faces = self._document_photo_face_candidates(oriented)
            combined = kept + doc_faces
            if combined:
                log.info(
                    "face_detection_complete",
                    detector=self._detector,
                    angle=angle,
                    faces_kept=len(combined),
                    document_photo_candidates=len(doc_faces),
                    ms=round((time.perf_counter() - started) * 1000),
                )
                return {"angle": angle, "faces": combined}

        log.info(
            "face_detection_no_face",
            detector=self._detector,
            tried=list(_ORIENTATIONS),
            ms=round((time.perf_counter() - started) * 1000),
        )
        return {"angle": 0, "faces": []}

    def _cache_get(self, key: bytes) -> dict[str, Any] | None:
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
            return cached

    def _cache_put(self, key: bytes, value: dict[str, Any]) -> None:
        with self._cache_lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > _CACHE_MAX_ENTRIES:
                self._cache.popitem(last=False)

    def _detect_quality_gated(self, image: Image.Image) -> dict[str, Any]:
        """Detection with a per-image cache + serialised inference.

        The four endpoint calls for one image arrive near-simultaneously.
        Without the lock they would all miss the cache and run detection in
        parallel — which thrashes the CPU. With it: the first call computes
        and caches; the rest queue on `_ml_lock`, then the double-checked
        cache read returns the warm result instantly.
        """
        key = hashlib.sha1(np.asarray(image).tobytes(), usedforsecurity=False).digest()

        cached = self._cache_get(key)
        if cached is not None:
            return cached

        with self._ml_lock:
            # Re-check — another thread may have computed this while we waited.
            cached = self._cache_get(key)
            if cached is not None:
                return cached
            result = self._run_detection(image)
            self._cache_put(key, result)
            return result

    def _whole_image_portrait_fallback(self, image: Image.Image) -> dict[str, Any] | None:
        if not _is_probable_cropped_portrait(image):
            return None
        w, h = image.size
        face_img = image.convert("RGB").resize((224, 224), Image.LANCZOS)
        return {
            "face": np.asarray(face_img),
            "confidence": _CROPPED_PORTRAIT_CONFIDENCE,
            "x": 0,
            "y": 0,
            "w": w,
            "h": h,
            "angle": 0,
            "source": "whole_image_portrait",
        }

    def _top_face(self, image: Image.Image) -> dict[str, Any] | None:
        """Best quality-gated face, annotated with the orientation it was found in.

        Confidence alone can choose a tiny false-positive on document artwork.
        Blend confidence with face size so the dominant document photo wins
        when multiple detections are plausible.
        """
        det = self._detect_quality_gated(image)
        faces = det["faces"]
        if not faces:
            fallback = self._whole_image_portrait_fallback(image)
            if fallback is not None:
                log.info(
                    "face_detection_cropped_portrait_fallback",
                    detector=self._detector,
                    size=image.size,
                    confidence=fallback["confidence"],
                )
            return fallback
        oriented = self._orient(image, det["angle"])
        iw, ih = oriented.size

        def _rank(face: dict[str, Any]) -> float:
            area_ratio = (face["w"] * face["h"]) / max(1, iw * ih)
            area_score = min(1.0, (area_ratio**0.5) * 4.0)
            source_bonus = 0.35 if face.get("source") == "document_photo" else 0.0
            # QR/text false positives have very dense high-frequency edges.
            # Real printed portraits are softer, even when low quality.
            x, y, w, h = face["x"], face["y"], face["w"], face["h"]
            crop = np.asarray(oriented.crop((x, y, min(iw, x + w), min(ih, y + h))))
            if crop.size == 0:
                edge_penalty = 0.35
            else:
                gray = np.asarray(Image.fromarray(crop).convert("L"))
                gy, gx = np.gradient(gray.astype(np.float32))
                edge_density = float(((np.abs(gx) + np.abs(gy)) > 45).mean())
                edge_penalty = 0.35 if edge_density > 0.22 else 0.0
            return (face["confidence"] * 0.65) + (area_score * 0.20) + source_bonus - edge_penalty

        best = max(faces, key=_rank)
        return {**best, "angle": det["angle"]}

    def _crop_with_margin(self, image: Image.Image, face: dict[str, Any]) -> Image.Image:
        """Crop the face from `image` with `_crop_margin` padding on every side.

        Cropping from the full-resolution oriented image (rather than reusing
        the detector's tight, possibly-downscaled crop) gives a natural
        head-and-shoulders image that doesn't clip the forehead or chin.
        """
        iw, ih = image.size
        x, y, w, h = face["x"], face["y"], face["w"], face["h"]
        side_margin = w * self._crop_margin
        top_margin = h * self._crop_margin * 1.15
        bottom_margin = h * self._crop_margin * 1.45

        left = x - side_margin
        upper = y - top_margin
        right = x + w + side_margin
        lower = y + h + bottom_margin

        # Keep the crop portrait-ish/square-ish. Detector boxes can be narrow
        # on IDs; widening prevents ear/cheek clipping, while not growing so
        # much that the crop becomes the whole document.
        crop_w = right - left
        crop_h = lower - upper
        aspect = crop_w / max(crop_h, 1)
        if aspect < 0.70:
            extra = ((crop_h * 0.70) - crop_w) / 2
            left -= extra
            right += extra
        elif aspect > 1.20:
            extra = ((crop_w / 1.20) - crop_h) / 2
            upper -= extra
            lower += extra

        left = max(0, int(round(left)))
        upper = max(0, int(round(upper)))
        right = min(iw, int(round(right)))
        lower = min(ih, int(round(lower)))
        # Guard against a degenerate box collapsing to zero area.
        if right <= left or lower <= upper:
            return Image.fromarray(self._face_to_uint8(face["face"]))
        return image.crop((left, upper, right, lower))

    def _gender_input_variants(self, image: Image.Image, top: dict[str, Any]) -> list[np.ndarray]:
        face_arr = self._face_to_uint8(top["face"])
        variants: list[Image.Image] = [Image.fromarray(face_arr).convert("RGB")]

        oriented = self._orient(image, int(top.get("angle", 0)))
        source_crop = self._crop_with_margin(oriented, top).convert("RGB")
        variants.append(source_crop.resize((224, 224), Image.LANCZOS))

        if self._is_low_quality_gender_input(image, top):
            fitted = ImageOps.fit(
                image.convert("RGB"),
                (224, 224),
                method=Image.LANCZOS,
                centering=(0.5, 0.42),
            )
            variants.extend(
                [
                    ImageEnhance.Contrast(fitted).enhance(1.15),
                    ImageEnhance.Sharpness(fitted).enhance(1.25),
                ]
            )

        unique: list[np.ndarray] = []
        seen: set[bytes] = set()
        for variant in variants:
            arr = np.asarray(variant)
            key = arr.tobytes()
            if key not in seen:
                seen.add(key)
                unique.append(arr)
        return unique

    def _insightface_input_variants(
        self, image: Image.Image, top: dict[str, Any]
    ) -> list[np.ndarray]:
        """Inputs for insightface, which runs its OWN detection + alignment.

        Unlike the DeepFace classifiers (fed a tight 224px crop with
        detector_backend="skip"), insightface localises and aligns the face
        itself, so it works best on the natural image with surrounding
        context. Tight pre-crops squashed to 224px distort the face and flip
        the genderage label — so here we hand it the full oriented image plus
        an upscale when the source is small enough to starve the model of
        pixels.
        """
        oriented = self._orient(image, int(top.get("angle", 0))).convert("RGB")

        # An upscaled margin-crop of the detected face (natural context, not
        # squashed to 224) so insightface's detector/aligner get a usable face.
        crop = self._crop_with_margin(oriented, top).convert("RGB")
        short_edge = min(crop.size)
        if 0 < short_edge < _INSIGHTFACE_CROP_MIN_EDGE:
            scale = _INSIGHTFACE_CROP_MIN_EDGE / short_edge
            crop = crop.resize(
                (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                Image.LANCZOS,
            )

        # On ID cards the printed photo is a tiny region of a big card. When the
        # whole card is resized to det_size, that ~40-80px face becomes a weak,
        # often-misgendered detection. Feeding BOTH the full card and the crop
        # makes those two disagree, which collapses agreement and forces an
        # abstain. So for a small inset face, trust the upscaled crop alone; only
        # use the full natural image when the face already fills a fair share of
        # it (selfie / cropped portrait), where context helps insightface.
        face_short = min(int(top.get("w", 0)), int(top.get("h", 0)))
        img_short = max(1, min(oriented.size))
        is_inset_face = (face_short / img_short) < _INSET_FACE_FRACTION

        if is_inset_face:
            variants: list[Image.Image] = [crop]
        else:
            variants = [oriented, crop]
            if min(oriented.size) < 320:
                variants.append(
                    oriented.resize(
                        (oriented.width * 2, oriented.height * 2), Image.LANCZOS
                    )
                )

        unique: list[np.ndarray] = []
        seen: set[bytes] = set()
        for variant in variants:
            arr = np.asarray(variant)
            key = arr.tobytes()
            if key not in seen:
                seen.add(key)
                unique.append(arr)
        return unique

    @staticmethod
    def _is_low_quality_gender_input(image: Image.Image, top: dict[str, Any]) -> bool:
        if top.get("source") == "whole_image_portrait":
            return True
        if _is_probable_cropped_portrait(image):
            return True
        return min(int(top.get("w", 0)), int(top.get("h", 0))) < 80

    @staticmethod
    def _gender_thresholds(image: Image.Image, top: dict[str, Any]) -> tuple[float, float]:
        if not DeepFacePredictor._is_low_quality_gender_input(image, top):
            return _GENDER_MIN_MODEL_CONF, 0.0

        if _is_probable_cropped_portrait(image):
            short_edge = min(image.size)
            if short_edge < _GENDER_TINY_PORTRAIT_MAX_SHORT_EDGE:
                return _GENDER_TINY_PORTRAIT_MIN_MODEL_CONF, _GENDER_LOW_QUALITY_MIN_MARGIN
            return _GENDER_CROPPED_PORTRAIT_MIN_MODEL_CONF, _GENDER_LOW_QUALITY_MIN_MARGIN

        return _GENDER_TINY_PORTRAIT_MIN_MODEL_CONF, _GENDER_LOW_QUALITY_MIN_MARGIN

    @staticmethod
    def _analyze_rows(result: Any) -> list[dict[str, Any]]:
        if isinstance(result, list):
            return [row for row in result if isinstance(row, dict)]
        return [result] if isinstance(result, dict) else []

    def _get_insightface_gender_app(self) -> Any | None:
        if self._insightface_gender_failed:
            return None
        if self._insightface_gender_app is not None:
            return self._insightface_gender_app

        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            self._insightface_gender_failed = True
            log.warning(
                "insightface_gender_unavailable",
                reason="install insightface and onnxruntime to enable genderage",
            )
            return None

        try:
            root = get_settings().image_model_weights_dir or "~/.insightface"
            app = FaceAnalysis(
                name="buffalo_l",
                root=root,
                allowed_modules=["detection", "genderage"],
            )
            app.prepare(ctx_id=-1, det_size=(320, 320))
        except Exception as e:  # noqa: BLE001
            self._insightface_gender_failed = True
            log.warning("insightface_gender_init_failed", error=str(e))
            return None

        self._insightface_gender_app = app
        log.info("insightface_gender_ready", model="buffalo_l")
        return app

    @staticmethod
    def _largest_insightface_face(faces: list[Any]) -> Any | None:
        if not faces:
            return None

        def _rank(face: Any) -> float:
            bbox = getattr(face, "bbox", None)
            if bbox is None or len(bbox) < 4:
                area = 1.0
            else:
                x1, y1, x2, y2 = (float(v) for v in bbox[:4])
                area = max(1.0, (x2 - x1) * (y2 - y1))
            return area * float(getattr(face, "det_score", 0.0))

        return max(faces, key=_rank)

    def _insightface_gender_predict(self, variants: list[np.ndarray]) -> GenderPrediction | None:
        app = self._get_insightface_gender_app()
        if app is None:
            return None

        votes: list[tuple[str, float]] = []
        for variant in variants:
            try:
                bgr = np.ascontiguousarray(variant[:, :, ::-1])
                faces = app.get(bgr)
            except Exception as e:  # noqa: BLE001
                log.warning("insightface_gender_variant_failed", error=str(e))
                continue

            face = self._largest_insightface_face(faces)
            if face is None or not hasattr(face, "gender"):
                continue
            label = "M" if int(face.gender) == 1 else "F"
            votes.append((label, float(getattr(face, "det_score", 0.75))))

        gender, confidence = _select_insightface_gender(votes)
        if gender is None:
            log.info("insightface_gender_uncertain", votes=len(votes))
            return None
        log.info("insightface_gender_prediction", gender=gender, confidence=confidence)
        return GenderPrediction(gender=gender, confidence=confidence)

    # ─── Public predictor interface ───────────────────────────────────────

    async def face_detect_crop(self, image: Image.Image) -> FaceDetectCrop:
        def _run() -> FaceDetectCrop:
            det = self._detect_quality_gated(image)
            faces = det["faces"]
            if not faces:
                log.info("face_detect_crop_no_face", detector=self._detector)
                return FaceDetectCrop(
                    face_detected=False, face_count=0, confidence=0.0, faces=[]
                )
            # Crop every detected face from the orientation it was found in, so
            # each output is upright even when the source card was photographed
            # sideways. Margin is applied against the full-resolution image.
            oriented = self._orient(image, det["angle"])
            cropped: list[CroppedFace] = []
            for f in sorted(faces, key=lambda x: x["confidence"], reverse=True):
                face_img = self._crop_with_margin(oriented, f)
                cropped.append(
                    CroppedFace(
                        face_b64=encode_image_b64(face_img),
                        bbox=[f["x"], f["y"], f["w"], f["h"]],
                        confidence=float(f["confidence"]),
                    )
                )
            return FaceDetectCrop(
                face_detected=True,
                face_count=len(cropped),
                confidence=max(c.confidence for c in cropped),
                faces=cropped,
            )

        return await asyncio.to_thread(_run)

    async def age_predict(self, image: Image.Image) -> AgePrediction:
        from deepface import DeepFace

        def _run() -> AgePrediction:
            top = self._top_face(image)
            if not top:
                log.info("age_predict_skipped_no_face", detector=self._detector)
                return AgePrediction(age=None, age_range=None, confidence=0.0)
            # Crop the detected face from the full-resolution oriented image with
            # the same margin /ai/face-detect-crop uses, so the caller gets the
            # cropped Aadhaar photo back alongside the predicted age.
            oriented = self._orient(image, int(top.get("angle", 0)))
            face_crop_b64 = encode_image_b64(self._crop_with_margin(oriented, top))
            bbox = [int(top["x"]), int(top["y"]), int(top["w"]), int(top["h"])]
            face_arr = self._face_to_uint8(top["face"])
            try:
                # `_top_face` already released `_ml_lock`; re-acquire it just
                # for the classifier so analyses don't thrash each other.
                with self._ml_lock:
                    result = DeepFace.analyze(
                        img_path=face_arr,
                        actions=["age"],
                        enforce_detection=False,
                        detector_backend="skip",
                    )
            except Exception as e:  # noqa: BLE001
                raise ModelInferenceError(f"age_predict failed: {e}") from e
            row = result[0] if isinstance(result, list) else result
            raw_age = row.get("age")
            age = int(raw_age) if raw_age is not None else None
            return AgePrediction(
                age=age,
                age_range=age_to_bucket(age),
                # Tie reported confidence to detection — the age regressor has no
                # native per-prediction confidence, so we expose the only honest
                # signal we have (how well we localised the face).
                confidence=top["confidence"],
                face_b64=face_crop_b64,
                bbox=bbox,
            )

        return await asyncio.to_thread(_run)

    async def gender_predict(self, image: Image.Image) -> GenderPrediction:
        # def _run() -> GenderPrediction:
        #     top = self._top_face(image)
        #     if not top:
        #         log.info("gender_predict_skipped_no_face", detector=self._detector)
        #         return GenderPrediction(gender=None, confidence=0.0)
        def _run() -> GenderPrediction:
            top = self._top_face(image)
            if not top and _is_probable_cropped_portrait(image):
                log.info("gender_predict_portrait_bypass", detector=self._detector, size=image.size)
                w, h = image.size
                top = {
                    "face": np.asarray(image.convert("RGB").resize((224, 224), Image.LANCZOS)),
                    "confidence": _CROPPED_PORTRAIT_CONFIDENCE,
                    "x": 0, "y": 0, "w": w, "h": h,
                    "angle": 0,
                    "source": "whole_image_portrait",
                }
            if not top:
                log.info("gender_predict_skipped_no_face", detector=self._detector)
                return GenderPrediction(gender=None, confidence=0.0)

            # Crop the detected face from the full-resolution oriented image with
            # the same margin /ai/face-detect-crop uses, so the caller gets the
            # cropped Aadhaar photo back alongside the predicted gender.
            oriented_for_crop = self._orient(image, int(top.get("angle", 0)))
            crop_img = self._crop_with_margin(oriented_for_crop, top)
            face_crop_b64 = encode_image_b64(crop_img)
            bbox = [int(top["x"]), int(top["y"]), int(top["w"]), int(top["h"])]

            # Abstain when the detected face is too small to gender reliably
            # (see _GENDER_MIN_FACE_EDGE_PX). The crop is still returned so the
            # caller can see what was detected; only the label is withheld.
            is_portrait_input = (
                top.get("source") == "whole_image_portrait"
                or _is_probable_cropped_portrait(image)
            )
            if not is_portrait_input and min(int(top["w"]), int(top["h"])) < _GENDER_MIN_FACE_EDGE_PX:
                log.info(
                    "gender_predict_face_too_small",
                    detector=self._detector,
                    w=int(top["w"]),
                    h=int(top["h"]),
                    min_edge=_GENDER_MIN_FACE_EDGE_PX,
                )
                return GenderPrediction(
                    gender=None, confidence=0.0, face_b64=face_crop_b64, bbox=bbox
                )

            # Abstain on too-dark faces. Both gender models confidently
            # misclassify underexposed ID photos (calibrated on real samples: a
            # wrong dark face scored mean-luminance ~45, while a correct dim face
            # scored ~60). Blur is NOT gated — the blurriest sample was the one
            # classified correctly, so a blur gate would suppress good answers.
            mean_luminance = float(np.asarray(crop_img.convert("L")).mean())
            if mean_luminance < _GENDER_MIN_FACE_BRIGHTNESS:
                log.info(
                    "gender_predict_face_too_dark",
                    detector=self._detector,
                    mean_luminance=round(mean_luminance, 1),
                    floor=_GENDER_MIN_FACE_BRIGHTNESS,
                )
                return GenderPrediction(
                    gender=None, confidence=0.0, face_b64=face_crop_b64, bbox=bbox
                )

            # insightface re-detects + aligns internally, so give it the
            # natural-context image; DeepFace below needs the tight 224 crops.
            insightface_inputs = self._insightface_input_variants(image, top)
            with self._ml_lock:
                insightface_result = self._insightface_gender_predict(insightface_inputs)
            if insightface_result is not None:
                return replace(insightface_result, face_b64=face_crop_b64, bbox=bbox)

            variants = self._gender_input_variants(image, top)

            from deepface import DeepFace

            try:
                # `_top_face` already released `_ml_lock`; re-acquire it just
                # for the classifier so analyses don't thrash each other.
                with self._ml_lock:
                    rows: list[dict[str, Any]] = []
                    for variant in variants:
                        result = DeepFace.analyze(
                            img_path=variant,
                            actions=["gender"],
                            enforce_detection=False,
                            detector_backend="skip",
                        )
                        rows.extend(self._analyze_rows(result))
            except Exception as e:  # noqa: BLE001
                raise ModelInferenceError(f"gender_predict failed: {e}") from e
            min_model_conf, min_margin = self._gender_thresholds(image, top)
            gender, model_conf = _select_gender_prediction(
                rows,
                min_model_conf=min_model_conf,
                min_margin=min_margin,
            )
            if gender is None:
                log.info(
                    "gender_predict_uncertain",
                    detector=self._detector,
                    source=top.get("source"),
                    variants=len(variants),
                )
                return GenderPrediction(
                    gender=None, confidence=0.0, face_b64=face_crop_b64, bbox=bbox
                )
            # Gender confidence is the classifier probability after the face
            # has passed localisation gates. M/F is only returned when this
            # clears the configured threshold; uncertain cases stay null.
            return GenderPrediction(
                gender=gender, confidence=model_conf, face_b64=face_crop_b64, bbox=bbox
            )

        return await asyncio.to_thread(_run)
