"""A separate call path that sends one image to Jev, for the multimodal test only.

TypeSafe documents PNG and JPEG input for Jev 1.13 and 1.14 and labels it "Dreamer-ready", not production
ready: the request format, answers and limits can change between releases (docs.typesafe.ai/images, read
2026-11-17). This module treats it as a preview. The Jev adapter in src/ is not changed and still sends text
only.

    POST {base_url}/systemone
    TypeSafe-Danger-Zone-Images: true
    {"model": ..., "state": {"record": <audit record JSON text>, "report": <ref>},
     "assets": {<ref>: {"type": "image", "format": "png", "data_base64": ...}}, "questions": {...}}
"""

from __future__ import annotations

import base64
import struct
import time
from collections.abc import Mapping

import httpx
from pydantic import JsonValue, ValidationError

from paved_gate.evaluator.base import EvaluationResult, EvaluatorError, MediaPart, Question
from paved_gate.evaluator.jev import JevAPIError, JevEvaluator, _to_answer
from paved_gate.evaluator.jev_models import JevResponse

IMAGES_HEADER = "TypeSafe-Danger-Zone-Images"
FORMATS = {"image/png": "png", "image/jpeg": "jpg"}
# Documented limits: 16 MiB per file, 280 to 1568 px per side, 78,400 to 2,458,624 pixels.
MAX_BYTES = 16 * 1024 * 1024
SIDE = (280, 1568)
PIXELS = (78_400, 2_458_624)
REF = "img-ref-7c1e9a"  # replaced by the image wherever it appears; must not occur in the record text


def png_size(data: bytes) -> tuple[int, int]:
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def check_image(image: MediaPart) -> None:
    if image.mime_type != "image/png":
        raise EvaluatorError(f"{image.name}: this test sends PNG only, got {image.mime_type}")
    if len(image.data) > MAX_BYTES:
        raise EvaluatorError(f"{image.name} is {len(image.data):,} bytes, over Jev's documented 16 MiB")
    w, h = png_size(image.data)
    if not (SIDE[0] <= w <= SIDE[1] and SIDE[0] <= h <= SIDE[1] and PIXELS[0] <= w * h <= PIXELS[1]):
        raise EvaluatorError(f"{image.name} is {w}x{h}, outside Jev's documented image limits")


def build_body(
    model: str, record_text: str, image: MediaPart, questions: Mapping[str, Question]
) -> dict[str, JsonValue]:
    if REF in record_text:
        raise EvaluatorError("the record text contains the image reference string")
    check_image(image)
    # Reuse the adapter's own request model for the questions, then add the documented assets map.
    body: dict[str, JsonValue] = (
        JevEvaluator(mode="mock", model=model).build_request("", questions).model_dump(mode="json")
    )
    body["state"] = {"record": record_text, "report": REF}
    body["assets"] = {
        REF: {
            "type": "image",
            "format": FORMATS[image.mime_type],
            "data_base64": base64.b64encode(image.data).decode("ascii"),
        }
    }
    return body


async def evaluate_jev_image(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    model: str,
    record_text: str,
    image: MediaPart,
    questions: Mapping[str, Question],
    base_url: str = "https://api.typesafe.ai/v1",
) -> EvaluationResult:
    """One call, no retries: a failure is reported as it is."""
    body = build_body(model, record_text, image, questions)
    headers = {"Authorization": f"Bearer {api_key}", IMAGES_HEADER: "true"}
    started = time.perf_counter()
    try:
        resp = await client.post(f"{base_url.rstrip('/')}/systemone", json=body, headers=headers)
    except httpx.HTTPError as exc:
        raise EvaluatorError(f"Jev image request failed: {exc!r}") from exc
    latency_ms = (time.perf_counter() - started) * 1000
    if resp.status_code != 200:
        raise JevAPIError(resp.status_code, resp.text)
    payload = resp.json()
    if not isinstance(payload, dict):
        raise EvaluatorError("Jev response body is not a JSON object")
    try:
        response = JevResponse.model_validate(payload)
    except ValidationError as exc:
        raise EvaluatorError(f"Jev response did not match the expected schema: {exc}") from exc
    missing = set(questions) - set(response.answers)
    if missing:
        raise EvaluatorError(f"Jev response is missing answers for {sorted(missing)}")
    return EvaluationResult(
        evaluator="jev_image_preview",
        model=response.model,
        mode="live",
        answers={qid: _to_answer(qid, q, response.answers[qid]) for qid, q in questions.items()},
        raw=payload,
        latency_ms=round(latency_ms, 2),
        input_tokens=response.usage.input_tokens if response.usage else None,
        output_tokens=response.usage.output_tokens if response.usage else None,
        retries=0,
    )
