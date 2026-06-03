"""MiVOLO-backed predictor — joint age + gender, tuned for CPU.

Why this backend exists
-----------------------
The deepface backend predicts age with DeepFace's VGG-based age regressor,
which is the weakest link in that stack. MiVOLO (https://github.com/
WildChlamydia/MiVOLO) is a transformer model that jointly predicts age and
gender and is markedly more accurate on age. This backend swaps the age /
gender classifiers for MiVOLO while keeping the rest of the pipeline that
already works well on Indian OVDs (Aadhaar / PAN / DL).

CPU discipline
--------------
On a CPU box MiVOLO is a heavy ViT. Three rules keep latency sane and are
the whole reason this is a separate backend rather than another variant of
the deepface code path:

  * **One model, run once.** MiVOLO is *joint* — a single inference returns
    age AND gender. The /age-predict and /gender-predict endpoints both read
    from one cached MiVOLO result per image (`_attr_cache`), so the model
    runs once even though the orchestrator fires both endpoints.
  * **One variant, not many.** The deepface backend votes MiVOLO-equivalent
    classifiers over several augmented crops. Multiplying a ViT over 3-4
    variants on CPU is exactly what hurts, so here MiVOLO sees a single
    well-aligned crop.
  * **Serialised inference.** A single `_ml_lock` ensures one model runs at a
    time — TensorFlow/Torch grab every core per inference, so concurrency
    thrashes the CPU. Same rationale as the deepface backend.

Detection stays YuNet
---------------------
YuNet (the same detector the deepface backend defaults to) localises the
inset face — it is reliable on small printed ID-card photos and never fires
on QR codes or signatures. MiVOLO's own advantage is body/person context,
which an ID-card headshot does not provide, so MiVOLO runs face-only here.

Fallback
--------
InsightFace `buffalo_l` (genderage) also yields age + gender. It is the
fallback when MiVOLO is unavailable (no checkpoint, import error) or when
MiVOLO abstains on gender. Calibrated abstain gates (tiny / dark faces) are
shared with the deepface backend so a noisy ID photo returns null rather
than a confidently-wrong label.

MiVOLO is not on PyPI — install it from source:
``pip install "git+https://github.com/WildChlamydia/MiVOLO.git"`` (plus
``torch``, and ``insightface`` + ``onnxruntime`` for the fallback). Download
the ``mivolo_imbd.pth.tar`` checkpoint, point ``IMAGE_MIVOLO_CHECKPOINT`` at
it, and set ``IMAGE_AI_BACKEND=mivolo``. Detection uses YuNet, so MiVOLO's
own yolov8 person-face detector is not needed here.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import get_settings
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

# These pure helpers are already unit-tested against the deepface backend;
# reuse them so the two backends abstain on the same calibrated signals
# instead of drifting apart.
from .deepface_impl import (
    _CROPPED_PORTRAIT_CONFIDENCE,
    _GENDER_MIN_FACE_BRIGHTNESS,
    _GENDER_MIN_FACE_EDGE_PX,
    _INSIGHTFACE_CROP_MIN_EDGE,
    _is_probable_cropped_portrait,
    _select_insightface_gender,
)
from .yunet_detector import YuNetFaceDetector

log = get_logger(__name__)

_CACHE_MAX_ENTRIES = 32
# Orientations are handled inside YuNetFaceDetector; this backend only needs
# the angle it reports back to crop upright.


@dataclass(frozen=True)
class _Attributes:
    """One image's age + gender, computed once and shared by both endpoints."""

    detected: bool
    age: int | None
    age_confidence: float
    gender: str | None
    gender_confidence: float
    face_b64: str | None
    bbox: list[int] | None


def _decode_mivolo_output(output: Any, meta: Any) -> tuple[int | None, str | None, float]:
    """Turn a MiVOLO forward pass into (age, gender, gender_confidence).

    Mirrors ``MiVOLO.fill_in_results`` exactly so this stays correct against
    the model. For the joint model each output row is
    ``[male_logit, female_logit, age_norm]``: age is column 2, denormalised
    with the checkpoint's ``(age_norm * (max_age - min_age) + avg_age)``, and
    gender is the softmax over columns 0:2 (index 0 = male, 1 = female).
    only-age checkpoints emit just the normalised age and no gender.
    """
    min_age = float(meta.min_age)
    max_age = float(meta.max_age)
    avg_age = float(meta.avg_age)

    if bool(getattr(meta, "only_age", False)):
        age_norm = float(output.reshape(-1)[0].item())
        age = int(round(age_norm * (max_age - min_age) + avg_age))
        return age, None, 0.0

    row = output[0]
    age = int(round(float(row[2].item()) * (max_age - min_age) + avg_age))

    gender_probs = output[:, :2].softmax(-1)[0]
    male_p = float(gender_probs[0].item())
    female_p = float(gender_probs[1].item())
    gender = "M" if male_p >= female_p else "F"
    gender_conf = max(male_p, female_p)
    return age, gender, gender_conf


class MiVOLOPredictor(Predictor):
    name = "mivolo"

    def __init__(self) -> None:
        s = get_settings()
        self._min_conf: float = s.image_min_face_confidence
        self._min_px: int = s.image_min_face_size_px
        self._max_edge: int = s.image_max_detect_edge
        self._crop_margin: float = s.image_face_crop_margin
        self._mivolo_ckpt: str | None = s.image_mivolo_checkpoint
        self._min_gender_conf: float = s.image_mivolo_min_gender_confidence

        # YuNet drives detection — reliable on small printed ID-card faces and
        # immune to QR-code / signature false positives.
        self._yunet = YuNetFaceDetector(
            min_confidence=self._min_conf,
            min_size_px=self._min_px,
            detect_edge=self._max_edge,
            weights_dir=s.image_model_weights_dir,
        )

        self._mivolo: Any | None = None
        self._mivolo_failed = False
        self._insightface_app: Any | None = None
        self._insightface_failed = False

        # Per-image detection cache + per-image attribute cache. The endpoint
        # calls for one image share a single detection pass AND a single
        # MiVOLO inference instead of recomputing per call.
        self._cache: OrderedDict[bytes, dict[str, Any]] = OrderedDict()
        self._attr_cache: OrderedDict[bytes, _Attributes] = OrderedDict()
        self._cache_lock = threading.Lock()
        # Serialises ALL model inference — see module docstring.
        self._ml_lock = threading.Lock()
        # Serialises per-image attribute computation so concurrent /age-predict
        # and /gender-predict calls for one image run MiVOLO exactly once: the
        # first computes + caches, the rest wait then hit the cache. Distinct
        # from _ml_lock — _compute_attributes re-acquires _ml_lock internally,
        # and threading.Lock is not reentrant.
        self._attr_lock = threading.Lock()

        self._warm_up()

    # ─── Lazy model loaders ───────────────────────────────────────────────

    def _get_mivolo(self) -> Any | None:
        if self._mivolo_failed:
            return None
        if self._mivolo is not None:
            return self._mivolo
        if not self._mivolo_ckpt:
            self._mivolo_failed = True
            log.warning(
                "mivolo_unavailable",
                reason="set IMAGE_MIVOLO_CHECKPOINT to the MiVOLO weights; "
                "running gender/age via buffalo_l fallback only",
            )
            return None
        try:
            from mivolo.model.mi_volo import MiVOLO

            model = MiVOLO(
                self._mivolo_ckpt,
                half=False,
                use_persons=False,  # ID-card headshots have no body context
                disable_faces=False,
                verbose=False,
                device="cpu",
            )
        except Exception as e:  # noqa: BLE001 — degrade to fallback, never crash boot
            self._mivolo_failed = True
            log.warning("mivolo_init_failed", error=str(e))
            return None
        self._mivolo = model
        log.info("mivolo_ready", checkpoint=self._mivolo_ckpt)
        return model

    def _get_insightface(self) -> Any | None:
        if self._insightface_failed:
            return None
        if self._insightface_app is not None:
            return self._insightface_app
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            self._insightface_failed = True
            log.warning(
                "insightface_unavailable",
                reason="install insightface + onnxruntime to enable the buffalo_l fallback",
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
            self._insightface_failed = True
            log.warning("insightface_init_failed", error=str(e))
            return None
        self._insightface_app = app
        log.info("insightface_ready", model="buffalo_l")
        return app

    def _warm_up(self) -> None:
        """Trigger weight loads at startup, not on the first user request."""
        try:
            self._get_mivolo()
            self._get_insightface()
        except Exception as e:  # noqa: BLE001 — warm-up is best-effort
            log.warning("mivolo_warmup_failed", error=str(e))

    # ─── Geometry helpers ─────────────────────────────────────────────────

    @staticmethod
    def _orient(image: Image.Image, angle: int) -> Image.Image:
        """Return the image rotated clockwise by `angle` (0/90/180/270)."""
        if angle == 0:
            return image
        return image.rotate(-angle, expand=True)

    def _crop_with_margin(self, image: Image.Image, face: dict[str, Any]) -> Image.Image:
        """Crop the face with `_crop_margin` padding on every side.

        Mirrors the deepface backend so /face-detect-crop, /age-predict and
        /gender-predict all return the same natural head-and-shoulders crop.
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
        if right <= left or lower <= upper:
            return image.crop((x, y, min(iw, x + w), min(ih, y + h)))
        return image.crop((left, upper, right, lower))

    # ─── Detection (cached + serialised) ──────────────────────────────────

    def _run_detection(self, image: Image.Image) -> dict[str, Any]:
        started = time.perf_counter()
        # YuNet handles orientation internally and returns the angle its boxes
        # are expressed in; we carry that angle through so crops come out upright.
        angle, detected = self._yunet.detect(image)
        faces: list[dict[str, Any]] = [
            {"confidence": f.confidence, "x": f.x, "y": f.y, "w": f.w, "h": f.h}
            for f in detected
        ]
        log.info(
            "face_detection_complete" if faces else "face_detection_no_face",
            detector="yunet",
            angle=angle,
            faces_kept=len(faces),
            ms=round((time.perf_counter() - started) * 1000),
        )
        return {"angle": angle, "faces": faces}

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
        key = hashlib.sha1(np.asarray(image).tobytes(), usedforsecurity=False).digest()
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        with self._ml_lock:
            cached = self._cache_get(key)
            if cached is not None:
                return cached
            result = self._run_detection(image)
            self._cache_put(key, result)
            return result

    def _top_face(self, image: Image.Image) -> dict[str, Any] | None:
        """Best detected face annotated with the orientation it was found in.

        Blend confidence with size so the dominant ID photo wins over a tiny
        false positive. Falls back to treating the whole image as a face when
        it looks like an already-cropped portrait.
        """
        det = self._detect_quality_gated(image)
        faces = det["faces"]
        if not faces:
            if _is_probable_cropped_portrait(image):
                w, h = image.size
                log.info("mivolo_whole_image_portrait_fallback", size=image.size)
                return {
                    "confidence": _CROPPED_PORTRAIT_CONFIDENCE,
                    "x": 0,
                    "y": 0,
                    "w": w,
                    "h": h,
                    "angle": 0,
                    "source": "whole_image_portrait",
                }
            return None
        oriented = self._orient(image, det["angle"])
        iw, ih = oriented.size

        def _rank(face: dict[str, Any]) -> float:
            area_ratio = (face["w"] * face["h"]) / max(1, iw * ih)
            area_score = min(1.0, (area_ratio**0.5) * 4.0)
            return (face["confidence"] * 0.7) + (area_score * 0.3)

        best = max(faces, key=_rank)
        return {**best, "angle": det["angle"]}

    # ─── Inference back-ends ──────────────────────────────────────────────

    def _mivolo_infer(self, face_rgb: np.ndarray) -> tuple[int | None, str | None, float] | None:
        """Run MiVOLO once on a single aligned face crop (face-only mode).

        Returns (age, gender, gender_confidence), or None when MiVOLO is
        unavailable or the inference fails — the caller then uses buffalo_l.

        Builds the model input the same way ``MiVOLO.prepare_crops`` /
        ``predict`` do (the module-level ``prepare_classification_images`` with
        the checkpoint's mean/std), bypassing MiVOLO's own YOLO detection since
        YuNet already located the face. Two details that mirror MiVOLO exactly:

          * ``prepare_classification_images`` does a BGR→RGB swap internally,
            so the crop is handed over as BGR.
          * The public IMDB checkpoint is a *with-persons* model that expects a
            6-channel input (face + body). With no body crop, MiVOLO feeds a
            zero person tensor — ``prepare_classification_images([None])`` — and
            concatenates it; the model was trained with person-dropout so this
            is its legitimate face-only path. We do the same.

        Wrapped defensively: any mismatch disables MiVOLO and falls back to
        buffalo_l.
        """
        model = self._get_mivolo()
        if model is None:
            return None
        try:
            import cv2
            import torch
            from mivolo.model.mi_volo import prepare_classification_images

            face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
            data_config = getattr(model, "data_config", {}) or {}
            mean = data_config.get("mean")
            std = data_config.get("std")
            # prepare_classification_images defaults to IMAGENET mean/std when
            # these are None, which matches MiVOLO's own data_config.
            kwargs = {}
            if mean is not None:
                kwargs["mean"] = mean
            if std is not None:
                kwargs["std"] = std
            size = int(model.input_size)
            faces_input = prepare_classification_images([face_bgr], size, device="cpu", **kwargs)
            if getattr(model.meta, "with_persons_model", False):
                # Zero (absent) body crop — the model's face-only path.
                person_input = prepare_classification_images([None], size, device="cpu", **kwargs)
                model_input = torch.cat((faces_input, person_input), dim=1)
            else:
                model_input = faces_input
            with torch.no_grad():
                output = model.inference(model_input)
            age, gender, gender_conf = _decode_mivolo_output(output, model.meta)
        except Exception as e:  # noqa: BLE001 — disable + fall back, never crash
            self._mivolo_failed = True
            log.warning("mivolo_inference_failed", error=str(e))
            return None
        log.info("mivolo_prediction", age=age, gender=gender, gender_conf=round(gender_conf, 3))
        return age, gender, gender_conf

    def _buffalo_infer(
        self, image: Image.Image, top: dict[str, Any]
    ) -> tuple[int | None, str | None, float] | None:
        """Fallback: InsightFace buffalo_l genderage — also joint age+gender."""
        app = self._get_insightface()
        if app is None:
            return None

        oriented = self._orient(image, int(top.get("angle", 0))).convert("RGB")
        crop = self._crop_with_margin(oriented, top).convert("RGB")
        short_edge = min(crop.size)
        if 0 < short_edge < _INSIGHTFACE_CROP_MIN_EDGE:
            scale = _INSIGHTFACE_CROP_MIN_EDGE / short_edge
            crop = crop.resize(
                (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                Image.LANCZOS,
            )

        try:
            bgr = np.ascontiguousarray(np.asarray(crop)[:, :, ::-1])
            faces = app.get(bgr)
        except Exception as e:  # noqa: BLE001
            log.warning("buffalo_inference_failed", error=str(e))
            return None
        if not faces:
            return None

        def _area(face: Any) -> float:
            bbox = getattr(face, "bbox", None)
            if bbox is None or len(bbox) < 4:
                return 1.0
            x1, y1, x2, y2 = (float(v) for v in bbox[:4])
            return max(1.0, (x2 - x1) * (y2 - y1))

        face = max(faces, key=_area)
        age = int(face.age) if getattr(face, "age", None) is not None else None
        gender: str | None = None
        gender_conf = 0.0
        if hasattr(face, "gender"):
            label = "M" if int(face.gender) == 1 else "F"
            score = float(getattr(face, "det_score", 0.75))
            gender, gender_conf = _select_insightface_gender([(label, score)])
        log.info("buffalo_prediction", age=age, gender=gender, gender_conf=round(gender_conf, 3))
        return age, gender, gender_conf

    # ─── Shared per-image computation ─────────────────────────────────────

    def _attr_cache_get(self, key: bytes) -> _Attributes | None:
        with self._cache_lock:
            cached = self._attr_cache.get(key)
            if cached is not None:
                self._attr_cache.move_to_end(key)
            return cached

    def _attr_cache_put(self, key: bytes, value: _Attributes) -> None:
        with self._cache_lock:
            self._attr_cache[key] = value
            self._attr_cache.move_to_end(key)
            while len(self._attr_cache) > _CACHE_MAX_ENTRIES:
                self._attr_cache.popitem(last=False)

    def _gender_allowed(self, image: Image.Image, top: dict[str, Any], crop: Image.Image) -> bool:
        """Apply the calibrated abstain gates (shared with the deepface backend).

        Tiny detected faces and underexposed ID photos are confidently
        misclassified by every gender model, so withhold the label there. The
        gates do not apply to whole-image portrait uploads (bbox = full image).
        """
        is_portrait = (
            top.get("source") == "whole_image_portrait"
            or _is_probable_cropped_portrait(image)
        )
        if not is_portrait and min(int(top["w"]), int(top["h"])) < _GENDER_MIN_FACE_EDGE_PX:
            log.info("gender_too_small", w=int(top["w"]), h=int(top["h"]))
            return False
        mean_luminance = float(np.asarray(crop.convert("L")).mean())
        if mean_luminance < _GENDER_MIN_FACE_BRIGHTNESS:
            log.info("gender_too_dark", mean_luminance=round(mean_luminance, 1))
            return False
        return True

    def _compute_attributes(self, image: Image.Image) -> _Attributes:
        """Detect → crop → MiVOLO (joint, once) → buffalo_l fallback.

        The result is cached per image so /age-predict and /gender-predict
        trigger only one MiVOLO inference between them.
        """
        top = self._top_face(image)
        if not top:
            log.info("mivolo_skipped_no_face")
            return _Attributes(False, None, 0.0, None, 0.0, None, None)

        oriented = self._orient(image, int(top.get("angle", 0)))
        crop_img = self._crop_with_margin(oriented, top)
        face_b64 = encode_image_b64(crop_img)
        bbox = [int(top["x"]), int(top["y"]), int(top["w"]), int(top["h"])]
        det_conf = float(top["confidence"])

        # One inference for both attributes. MiVOLO is fed a single aligned
        # crop resized to its input size; the prep inside _mivolo_infer
        # handles normalisation.
        face_arr = np.asarray(crop_img.convert("RGB"))
        with self._ml_lock:
            result = self._mivolo_infer(face_arr)
            if result is None:
                result = self._buffalo_infer(image, top)

        if result is None:
            # No model could classify — return the crop with null attributes.
            return _Attributes(True, None, det_conf, None, 0.0, face_b64, bbox)

        age, gender, gender_conf = result

        # Gender abstain gates + confidence floor.
        if gender is not None:
            if not self._gender_allowed(image, top, crop_img):
                gender, gender_conf = None, 0.0
            elif gender_conf < self._min_gender_conf:
                log.info("gender_uncertain", confidence=round(gender_conf, 3))
                gender, gender_conf = None, 0.0

        return _Attributes(
            detected=True,
            age=age,
            # The age regressor has no native per-prediction confidence, so we
            # expose detection confidence — the only honest signal we have.
            age_confidence=det_conf,
            gender=gender,
            gender_confidence=gender_conf,
            face_b64=face_b64,
            bbox=bbox,
        )

    def _attributes(self, image: Image.Image) -> _Attributes:
        key = hashlib.sha1(np.asarray(image).tobytes(), usedforsecurity=False).digest()
        cached = self._attr_cache_get(key)
        if cached is not None:
            return cached
        with self._attr_lock:
            # Re-check — another thread may have computed this while we waited.
            cached = self._attr_cache_get(key)
            if cached is not None:
                return cached
            attrs = self._compute_attributes(image)
            self._attr_cache_put(key, attrs)
            return attrs

    # ─── Public predictor interface ───────────────────────────────────────

    async def face_detect_crop(self, image: Image.Image) -> FaceDetectCrop:
        def _run() -> FaceDetectCrop:
            det = self._detect_quality_gated(image)
            faces = det["faces"]
            if not faces:
                log.info("face_detect_crop_no_face", detector="yunet")
                return FaceDetectCrop(
                    face_detected=False, face_count=0, confidence=0.0, faces=[]
                )
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
        def _run() -> AgePrediction:
            attrs = self._attributes(image)
            if not attrs.detected:
                return AgePrediction(age=None, age_range=None, confidence=0.0)
            return AgePrediction(
                age=attrs.age,
                age_range=age_to_bucket(attrs.age),
                confidence=attrs.age_confidence,
                face_b64=attrs.face_b64,
                bbox=attrs.bbox,
            )

        return await asyncio.to_thread(_run)

    async def gender_predict(self, image: Image.Image) -> GenderPrediction:
        def _run() -> GenderPrediction:
            attrs = self._attributes(image)
            if not attrs.detected:
                return GenderPrediction(gender=None, confidence=0.0)
            return GenderPrediction(
                gender=attrs.gender,
                confidence=attrs.gender_confidence,
                face_b64=attrs.face_b64,
                bbox=attrs.bbox,
            )

        return await asyncio.to_thread(_run)
