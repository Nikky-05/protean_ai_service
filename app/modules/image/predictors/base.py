"""Abstract Predictor interface.

Each backend (stub / deepface / mediapipe / future) implements this
interface. The service layer (`service.py`) and router (`router.py`)
talk only to this interface — swapping backends is a one-line config
change, not a code change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class CroppedFace:
    """A single detected face: its cropped image plus where it was found."""

    face_b64: str
    bbox: list[int]  # [x, y, w, h] in the orientation-corrected image
    confidence: float


@dataclass(frozen=True)
class FaceDetectCrop:
    """Combined detection + crop result.

    `face_detected` / `face_count` / `confidence` answer "is there a face?",
    while `faces` carries one cropped image (base64) per detected face.
    """

    face_detected: bool
    face_count: int
    confidence: float  # highest per-face detection confidence
    faces: list[CroppedFace]


@dataclass(frozen=True)
class AgePrediction:
    age: int | None
    age_range: str | None
    confidence: float
    # The face the age was predicted from, cropped (with margin) from the
    # input. Populated when a face was detected on the image — e.g. the small
    # printed photo on an Aadhaar card — so callers get the crop and the age
    # in one call. None when age came from DOB or no face was found.
    face_b64: str | None = None
    bbox: list[int] | None = None  # [x, y, w, h] in the orientation-corrected image


@dataclass(frozen=True)
class GenderPrediction:
    gender: str | None  # "M" | "F" | "Other"
    confidence: float
    # The face the gender was predicted from, cropped (with margin) from the
    # input — e.g. the printed photo on an Aadhaar card — so callers get the
    # crop and the gender in one call. None when no face was found.
    face_b64: str | None = None
    bbox: list[int] | None = None  # [x, y, w, h] in the orientation-corrected image


class Predictor(ABC):
    """Implement this to add a new ML backend."""

    name: str = "abstract"

    @abstractmethod
    async def face_detect_crop(self, image: Image.Image) -> FaceDetectCrop: ...

    @abstractmethod
    async def age_predict(self, image: Image.Image) -> AgePrediction: ...

    @abstractmethod
    async def gender_predict(self, image: Image.Image) -> GenderPrediction: ...


def age_to_bucket(age: int | None) -> str | None:
    if age is None or age < 0:
        return None
    if age < 18:
        return "0-17"
    if age <= 25:
        return "18-25"
    if age <= 35:
        return "26-35"
    if age <= 50:
        return "36-50"
    return "51+"
