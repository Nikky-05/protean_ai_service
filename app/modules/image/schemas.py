"""Pydantic request / response models for the image module."""

from datetime import date, datetime
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


class ImageRequest(BaseModel):
    image: str = Field(
        ...,
        description="Base64-encoded image (with or without data URI prefix) OR an https URL",
        examples=["data:image/jpeg;base64,/9j/4AAQSkZJ...", "https://cdn.example.com/x.jpg"],
    )


class AgePredictRequest(BaseModel):
    image: str | None = Field(
        None,
        description=(
            "Base64-encoded image (with or without data URI prefix) OR an https URL. "
            "Required when DOB is not available."
        ),
        examples=["data:image/jpeg;base64,/9j/4AAQSkZJ...", "https://cdn.example.com/x.jpg"],
    )
    dob: date | None = Field(
        None,
        validation_alias=AliasChoices("dob", "date_of_birth", "dateOfBirth"),
        description=(
            "Date of birth from the document. If provided, age is calculated exactly "
            "and face-based age prediction is skipped."
        ),
        examples=["1998-05-22", "22/05/1998"],
    )

    @field_validator("dob", mode="before")
    @classmethod
    def _parse_dob(cls, value):
        if value is None or value == "":
            return None
        if isinstance(value, date):
            return value
        if not isinstance(value, str):
            return value
        raw = value.strip()
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(raw, fmt).date()
            except ValueError:
                continue
        raise ValueError("dob must be a valid date: YYYY-MM-DD, DD-MM-YYYY, or DD/MM/YYYY")

    @model_validator(mode="after")
    def _require_age_source(self):
        if not self.image and self.dob is None:
            raise ValueError("Either `image` or `dob` is required")
        if self.dob is not None:
            today = date.today()
            if self.dob > today:
                raise ValueError("dob cannot be in the future")
            age = today.year - self.dob.year
            if (today.month, today.day) < (self.dob.month, self.dob.day):
                age -= 1
            if age > 120:
                raise ValueError("dob implies an age greater than 120")
        return self


class GenderPredictRequest(BaseModel):
    image: str = Field(
        ...,
        description="Base64-encoded image (with or without data URI prefix) OR an https URL",
        examples=["data:image/jpeg;base64,/9j/4AAQSkZJ...", "https://cdn.example.com/x.jpg"],
    )
    gender: Literal["M", "F", "Other"] | None = Field(
        None,
        validation_alias=AliasChoices("gender", "sex"),
        description=(
            "Known gender read from the document (e.g. the MALE/FEMALE printed on "
            "an Aadhaar card). When provided, it is returned authoritatively with "
            "confidence 1.0 and face-based gender prediction is skipped — the face "
            "is still cropped and returned. Accepts M/F/Other or full words like "
            "MALE/FEMALE/TRANSGENDER (any case). Unrecognised values are ignored "
            "and the model predicts instead."
        ),
        examples=["FEMALE", "M"],
    )

    @field_validator("gender", mode="before")
    @classmethod
    def _normalize_gender(cls, value):
        # Map whatever the document/OCR yields onto M/F/Other. Anything we don't
        # recognise (including blank) becomes None so the caller falls back to the
        # model rather than erroring the whole request.
        if not isinstance(value, str):
            return value
        v = value.strip().lower()
        if v in ("m", "male", "man"):
            return "M"
        if v in ("f", "female", "woman", "w"):
            return "F"
        if v in ("o", "other", "transgender", "trans", "t", "third gender"):
            return "Other"
        return None


class CroppedFace(BaseModel):
    """One detected face: its cropped image plus where it was found."""

    face_b64: str = Field(..., description="Cropped face as base64 JPEG")
    bbox: list[int] = Field(
        ..., description="[x, y, w, h] of the face in the orientation-corrected image"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)


class FaceDetectCropResponse(BaseModel):
    """Combined face detection + crop result."""

    face_detected: bool
    face_count: int = Field(..., ge=0)
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Highest per-face detection confidence"
    )
    faces: list[CroppedFace] = Field(
        default_factory=list,
        description="One entry per detected face, cropped with margin; "
        "empty when no face was detected",
    )


class AgePredictResponse(BaseModel):
    age: int | None = Field(None, ge=0, le=120)
    age_range: Literal["0-17", "18-25", "26-35", "36-50", "51+"] | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    face_b64: str | None = Field(
        None,
        description=(
            "The detected face, cropped from the image as base64 JPEG. Present "
            "when age was predicted from an image with a detectable face (e.g. the "
            "photo on an Aadhaar card); omitted when age came from DOB or no face "
            "was found."
        ),
    )
    bbox: list[int] | None = Field(
        None, description="[x, y, w, h] of the detected face in the orientation-corrected image"
    )


class GenderPredictResponse(BaseModel):
    gender: Literal["M", "F", "Other"] | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    face_b64: str | None = Field(
        None,
        description=(
            "The detected face, cropped from the image as base64 JPEG. Present "
            "when a detectable face was found (e.g. the photo on an Aadhaar card); "
            "omitted when no face was found."
        ),
    )
    bbox: list[int] | None = Field(
        None, description="[x, y, w, h] of the detected face in the orientation-corrected image"
    )
