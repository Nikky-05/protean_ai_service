"""MediaPipe-backed predictor — CPU-friendly face detection / crop.

MediaPipe gives us fast, lightweight face detection without TensorFlow.
It does NOT provide age or gender — those endpoints raise
`ModelInferenceError` so the operator can configure `deepface` for them
while keeping MediaPipe for the cheap detect/crop path.

Install with:

    pip install mediapipe opencv-python-headless

and set `IMAGE_AI_BACKEND=mediapipe` in `.env`.
"""

from __future__ import annotations

import asyncio

import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.core.exceptions import ModelInferenceError
from app.utils.image_io import encode_image_b64

from .base import (
    AgePrediction,
    CroppedFace,
    FaceDetectCrop,
    GenderPrediction,
    Predictor,
)

_ORIENTATIONS = (0, 90, 270, 180)


class MediaPipePredictor(Predictor):
    name = "mediapipe"

    def __init__(self) -> None:
        try:
            import mediapipe as mp
        except ImportError as e:
            raise ModelInferenceError(
                "mediapipe is not installed. Run `pip install mediapipe` or pick another backend."
            ) from e
        self._mp = mp
        s = get_settings()
        self._min_conf = s.image_min_face_confidence
        self._min_px = s.image_min_face_size_px
        self._crop_margin = s.image_face_crop_margin

    @staticmethod
    def _orient(image: Image.Image, angle: int) -> Image.Image:
        if angle == 0:
            return image
        return image.rotate(-angle, expand=True)

    def _detect(self, image: Image.Image):
        arr = np.asarray(image)
        with self._mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=self._min_conf
        ) as det:
            return arr, det.process(arr)

    def _detections_for_best_orientation(self, image: Image.Image):
        for angle in _ORIENTATIONS:
            oriented = self._orient(image, angle)
            arr, res = self._detect(oriented)
            faces = []
            h, w, _ = arr.shape
            for det in res.detections or []:
                rb = det.location_data.relative_bounding_box
                x = max(int(rb.xmin * w), 0)
                y = max(int(rb.ymin * h), 0)
                bw = min(int(rb.width * w), w - x)
                bh = min(int(rb.height * h), h - y)
                conf = float(det.score[0])
                if conf >= self._min_conf and bw >= self._min_px and bh >= self._min_px:
                    faces.append({"x": x, "y": y, "w": bw, "h": bh, "confidence": conf})
            if faces:
                return oriented, faces
        return image, []

    def _crop_with_margin(self, image: Image.Image, face: dict[str, float]) -> Image.Image:
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
        return image.crop((left, upper, right, lower))

    async def face_detect_crop(self, image: Image.Image) -> FaceDetectCrop:
        def _run() -> FaceDetectCrop:
            oriented, faces = self._detections_for_best_orientation(image)
            if not faces:
                return FaceDetectCrop(
                    face_detected=False, face_count=0, confidence=0.0, faces=[]
                )
            # Crop every detected face, highest-confidence first, from the
            # orientation it was found in so each crop comes out upright.
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

    async def age_predict(self, image: Image.Image) -> AgePrediction:  # noqa: ARG002
        raise ModelInferenceError(
            "MediaPipe backend does not implement age_predict. Use IMAGE_AI_BACKEND=deepface "
            "or implement a dedicated age model here."
        )

    async def gender_predict(self, image: Image.Image) -> GenderPrediction:  # noqa: ARG002
        raise ModelInferenceError(
            "MediaPipe backend does not implement gender_predict. Use IMAGE_AI_BACKEND=deepface "
            "or implement a dedicated gender model here."
        )
