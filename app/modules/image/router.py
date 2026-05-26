"""HTTP routes for the image module.

Routes consumed by the Node `protean-apigee-api` connectors:
  - /ai/face-detect-crop   (detect faces + return cropped face images)
  - /ai/age-predict
  - /ai/gender-predict

The router only marshals input/output. All real work lives in
`service.py` / the `Predictor` backend.
"""

from fastapi import APIRouter, Depends

from .predictors.factory import get_predictor
from .schemas import (
    AgePredictRequest,
    AgePredictResponse,
    CroppedFace,
    FaceDetectCropResponse,
    GenderPredictRequest,
    GenderPredictResponse,
    ImageRequest,
)
from .service import ImageService

router = APIRouter()


def _service() -> ImageService:
    return ImageService(predictor=get_predictor())


@router.post("/ai/face-detect-crop", response_model=FaceDetectCropResponse)
async def face_detect_crop(
    body: ImageRequest, svc: ImageService = Depends(_service)
) -> FaceDetectCropResponse:
    """Detect faces in an image and return each one cropped as base64.

    Single endpoint covering both detection and cropping: the response
    reports `face_detected` / `face_count` / `confidence`, and `faces`
    carries one cropped face image (base64 JPEG) per detected face.
    """
    r = await svc.face_detect_crop(body.image)
    return FaceDetectCropResponse(
        face_detected=r.face_detected,
        face_count=r.face_count,
        confidence=r.confidence,
        faces=[
            CroppedFace(face_b64=f.face_b64, bbox=f.bbox, confidence=f.confidence)
            for f in r.faces
        ],
    )


@router.post(
    "/ai/age-predict",
    response_model=AgePredictResponse,
    response_model_exclude_none=True,
)
async def age_predict(
    body: AgePredictRequest, svc: ImageService = Depends(_service)
) -> AgePredictResponse:
    """Predict age from an image or DOB.

    When given an image (e.g. an Aadhaar card), the face is detected and
    cropped, the age is predicted from that crop, and the crop is returned as
    `face_b64` (with its `bbox`). When `dob` is supplied, age is computed
    exactly and no face fields are returned.
    """
    r = await svc.age_predict(body.image, body.dob)
    return AgePredictResponse(
        age=r.age,
        age_range=r.age_range,
        confidence=r.confidence,
        face_b64=r.face_b64,
        bbox=r.bbox,
    )


@router.post(
    "/ai/gender-predict",
    response_model=GenderPredictResponse,
    response_model_exclude_none=True,
)
async def gender_predict(
    body: GenderPredictRequest, svc: ImageService = Depends(_service)
) -> GenderPredictResponse:
    """Return gender for an image (e.g. an Aadhaar card).

    If `gender` is supplied (read from the document text, e.g. the MALE/FEMALE
    printed on an Aadhaar), it is returned authoritatively with `confidence`
    1.0 and the model is skipped. Otherwise gender is predicted from the
    detected face. Either way the detected face is cropped and returned as
    `face_b64` (with its `bbox`); face fields are omitted when no face is found.
    """
    r = await svc.gender_predict(body.image, body.gender)
    return GenderPredictResponse(
        gender=r.gender,
        confidence=r.confidence,
        face_b64=r.face_b64,
        bbox=r.bbox,
    )
