"""End-to-end tests for the image module against the stub predictor."""

from datetime import date

import pytest


def _post(client, path, body):
    return client.post(f"/api/v1/image{path}", json=body)


def test_face_detect_crop_returns_expected_shape(client, jpeg_b64):
    res = _post(client, "/ai/face-detect-crop", {"image": jpeg_b64})
    assert res.status_code == 200
    body = res.json()
    assert set(body.keys()) == {"face_detected", "face_count", "confidence", "faces"}
    assert isinstance(body["face_detected"], bool)
    assert isinstance(body["face_count"], int)
    assert 0.0 <= body["confidence"] <= 1.0
    # `faces` carries one cropped face per detected face — count must agree,
    # and detection presence must match whether any face was returned.
    assert body["face_count"] == len(body["faces"])
    assert body["face_detected"] is (body["face_count"] > 0)
    for face in body["faces"]:
        assert face["face_b64"]  # non-empty base64 crop
        assert len(face["bbox"]) == 4
        assert 0.0 <= face["confidence"] <= 1.0


def test_age_predict_returns_age_and_bucket(client, jpeg_b64):
    res = _post(client, "/ai/age-predict", {"image": jpeg_b64})
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body["age"], int)
    assert body["age_range"] in {"0-17", "18-25", "26-35", "36-50", "51+"}
    assert 0.0 <= body["confidence"] <= 1.0


def test_age_predict_returns_cropped_face_for_image(client, jpeg_b64):
    """Image path detects + crops the face and returns it alongside the age."""
    res = _post(client, "/ai/age-predict", {"image": jpeg_b64})
    assert res.status_code == 200
    body = res.json()
    assert body["face_b64"]  # non-empty base64 crop
    assert len(body["bbox"]) == 4


def test_age_predict_uses_dob_when_present(client, jpeg_b64):
    today = date.today()
    dob = today.replace(year=today.year - 30)
    res = _post(client, "/ai/age-predict", {"image": jpeg_b64, "dob": dob.isoformat()})
    assert res.status_code == 200
    body = res.json()
    assert body == {"age": 30, "age_range": "26-35", "confidence": 1.0}


def test_age_predict_allows_dob_without_image(client):
    today = date.today()
    dob = today.replace(year=today.year - 17)
    res = _post(client, "/ai/age-predict", {"date_of_birth": dob.strftime("%d/%m/%Y")})
    assert res.status_code == 200
    body = res.json()
    assert body == {"age": 17, "age_range": "0-17", "confidence": 1.0}


def test_gender_predict_returns_valid_label(client, jpeg_b64):
    res = _post(client, "/ai/gender-predict", {"image": jpeg_b64})
    assert res.status_code == 200
    body = res.json()
    assert body["gender"] in {"M", "F", "Other"}
    assert 0.0 <= body["confidence"] <= 1.0


def test_gender_predict_returns_cropped_face(client, jpeg_b64):
    """Gender path detects + crops the face and returns it alongside the label."""
    res = _post(client, "/ai/gender-predict", {"image": jpeg_b64})
    assert res.status_code == 200
    body = res.json()
    assert body["face_b64"]  # non-empty base64 crop
    assert len(body["bbox"]) == 4


def test_stub_is_deterministic(client, jpeg_b64):
    """Same input → same prediction. Lets the Node side cache responses."""
    r1 = _post(client, "/ai/age-predict", {"image": jpeg_b64}).json()
    r2 = _post(client, "/ai/age-predict", {"image": jpeg_b64}).json()
    assert r1 == r2


@pytest.mark.parametrize("path", [
    "/ai/face-detect-crop", "/ai/age-predict", "/ai/gender-predict"
])
def test_invalid_base64_rejected(client, path):
    res = _post(client, path, {"image": "not-base64-and-not-a-url-!!"})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.parametrize("path", [
    "/ai/face-detect-crop", "/ai/age-predict", "/ai/gender-predict"
])
def test_missing_image_field_rejected(client, path):
    res = _post(client, path, {})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "VALIDATION_ERROR"
