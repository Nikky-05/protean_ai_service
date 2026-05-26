"""Service layer for the image module.

Keeps the router thin — anything beyond request/response wiring lives
here so it stays unit-testable without spinning up FastAPI. The service
delegates the actual ML work to whichever `Predictor` the factory
returns.
"""

from datetime import date

from app.core.logging import get_logger
from app.utils.image_io import load_image

from .predictors.base import (
    AgePrediction,
    FaceDetectCrop,
    GenderPrediction,
    Predictor,
    age_to_bucket,
)

log = get_logger(__name__)


def _age_on(today: date, dob: date) -> int:
    age = today.year - dob.year
    if (today.month, today.day) < (dob.month, dob.day):
        age -= 1
    return age


class ImageService:
    def __init__(self, predictor: Predictor) -> None:
        self._predictor = predictor

    async def face_detect_crop(self, image: str) -> FaceDetectCrop:
        img = await load_image(image)
        log.info("face_detect_crop", backend=self._predictor.name, size=img.size)
        return await self._predictor.face_detect_crop(img)

    async def age_predict(self, image: str | None, dob: date | None = None) -> AgePrediction:
        if dob is not None:
            age = _age_on(date.today(), dob)
            log.info("age_predict_from_dob", age=age)
            return AgePrediction(age=age, age_range=age_to_bucket(age), confidence=1.0)
        if image is None:
            # Should be prevented by request validation, but keep the service
            # defensive for direct callers and tests.
            from app.core.exceptions import ValidationError

            raise ValidationError("Either `image` or `dob` is required")
        img = await load_image(image)
        log.info("age_predict", backend=self._predictor.name, size=img.size)
        return await self._predictor.age_predict(img)

    async def gender_predict(
        self, image: str, gender: str | None = None
    ) -> GenderPrediction:
        img = await load_image(image)
        if gender is not None:
            # Gender is known from the document text (e.g. MALE/FEMALE printed on
            # an Aadhaar card). Trust it instead of guessing from a tiny face:
            # return it with confidence 1.0. We still detect + crop the face so
            # the caller gets the photo back in the same response.
            det = await self._predictor.face_detect_crop(img)
            top = det.faces[0] if det.faces else None
            log.info("gender_from_document", gender=gender, face_found=bool(top))
            return GenderPrediction(
                gender=gender,
                confidence=1.0,
                face_b64=top.face_b64 if top else None,
                bbox=top.bbox if top else None,
            )
        log.info("gender_predict", backend=self._predictor.name, size=img.size)
        return await self._predictor.gender_predict(img)
