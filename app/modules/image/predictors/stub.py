"""Stub predictor — deterministic mock, zero ML deps.

Used in local dev and CI to keep tests fast and hermetic. Outputs are
derived from the image's pixel data so the same input gives the same
result on every call (deterministic) while different inputs give
different results (testable).
"""

import hashlib

from PIL import Image

from app.utils.image_io import encode_image_b64

from .base import (
    AgePrediction,
    CroppedFace,
    FaceDetectCrop,
    GenderPrediction,
    Predictor,
    age_to_bucket,
)


def _digest(img: Image.Image) -> int:
    """Stable per-image int derived from a tiny thumbnail's bytes."""
    thumb = img.copy()
    thumb.thumbnail((32, 32))
    return int.from_bytes(hashlib.sha256(thumb.tobytes()).digest()[:8], "big")


class StubPredictor(Predictor):
    name = "stub"

    async def face_detect_crop(self, image: Image.Image) -> FaceDetectCrop:
        d = _digest(image)
        count = (d % 3)  # 0, 1, or 2 — exercises no-face / single / multi paths
        confidence = 0.5 + (d % 50) / 100
        w, h = image.size
        # Centre-crop a square 60% of the shortest side as a stand-in face,
        # nudging each extra face so multiple crops differ deterministically.
        side = int(min(w, h) * 0.6)
        faces: list[CroppedFace] = []
        for i in range(count):
            x = min(max((w - side) // 2 + i * 8, 0), max(w - side, 0))
            y = min(max((h - side) // 2 + i * 8, 0), max(h - side, 0))
            face = image.crop((x, y, x + side, y + side))
            faces.append(
                CroppedFace(
                    face_b64=encode_image_b64(face),
                    bbox=[x, y, side, side],
                    confidence=confidence,
                )
            )
        return FaceDetectCrop(
            face_detected=count > 0,
            face_count=count,
            confidence=confidence if count > 0 else 0.0,
            faces=faces,
        )

    async def age_predict(self, image: Image.Image) -> AgePrediction:
        d = _digest(image)
        age = 18 + (d % 50)  # 18..67
        # Mirror the real backends: return the (stand-in) cropped face the age
        # was predicted from — a centre square 60% of the shortest side.
        w, h = image.size
        side = int(min(w, h) * 0.6)
        x = max((w - side) // 2, 0)
        y = max((h - side) // 2, 0)
        face = image.crop((x, y, x + side, y + side))
        return AgePrediction(
            age=age,
            age_range=age_to_bucket(age),
            confidence=0.7 + (d % 30) / 100,
            face_b64=encode_image_b64(face),
            bbox=[x, y, side, side],
        )

    async def gender_predict(self, image: Image.Image) -> GenderPrediction:
        d = _digest(image)
        gender = ["M", "F", "Other"][d % 3]
        # Mirror the real backends: return the (stand-in) cropped face the
        # gender was predicted from — a centre square 60% of the shortest side.
        w, h = image.size
        side = int(min(w, h) * 0.6)
        x = max((w - side) // 2, 0)
        y = max((h - side) // 2, 0)
        face = image.crop((x, y, x + side, y + side))
        return GenderPrediction(
            gender=gender,
            confidence=0.7 + (d % 30) / 100,
            face_b64=encode_image_b64(face),
            bbox=[x, y, side, side],
        )
