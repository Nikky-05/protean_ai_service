"""Image input decoding — base64 / data-URI / HTTP(S) URL → PIL.Image.

The Node orchestrator may send either a base64 string or a public URL
(per PRD §5). Both paths converge on a `PIL.Image` so predictors stay
agnostic of input form.
"""

import base64
import io
import re
from typing import Final

import httpx
from PIL import Image, ImageOps

from app.core.exceptions import ValidationError

_URL_RE: Final = re.compile(r"^https?://", re.IGNORECASE)
_DATA_URI_RE: Final = re.compile(r"^data:image/[a-z]+;base64,", re.IGNORECASE)

_MAX_BYTES: Final = 15 * 1024 * 1024  # 15 MB hard cap
_HTTP_TIMEOUT: Final = 10.0


def _strip_data_uri(b64: str) -> str:
    return _DATA_URI_RE.sub("", b64, count=1)


async def _fetch_url(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
        try:
            r = await client.get(url)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise ValidationError(f"Failed to fetch image URL: {e}") from e
        if len(r.content) > _MAX_BYTES:
            raise ValidationError(f"Image exceeds {_MAX_BYTES} bytes")
        return r.content


def _decode_b64(b64: str) -> bytes:
    cleaned = _strip_data_uri(b64).strip()
    try:
        raw = base64.b64decode(cleaned, validate=True)
    except (ValueError, base64.binascii.Error) as e:
        raise ValidationError("Invalid base64 image payload") from e
    if len(raw) > _MAX_BYTES:
        raise ValidationError(f"Image exceeds {_MAX_BYTES} bytes")
    return raw


def _open_image(raw: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (OSError, ValueError) as e:
        raise ValidationError("Could not decode image bytes") from e
    # Honour EXIF orientation. Phone cameras record the sensor rotation as an
    # EXIF tag rather than rotating the pixels — so a photo that looks upright
    # in a gallery app is actually stored sideways. Without this transpose the
    # face detector sees a rotated face and silently fails to detect / crop.
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:  # noqa: BLE001 — corrupt EXIF must never break decoding
        pass
    return img.convert("RGB")


async def load_image(image: str) -> Image.Image:
    if not image or not isinstance(image, str):
        raise ValidationError("`image` must be a non-empty string (base64 or URL)")
    raw = await _fetch_url(image) if _URL_RE.match(image) else _decode_b64(image)
    return _open_image(raw)


def encode_image_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")
