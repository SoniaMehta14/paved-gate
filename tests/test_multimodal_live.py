"""The live multimodal test's offline parts: scoring, the Jev image request body, and the image only PDF.

No test reaches an API, Chrome or sips.
"""

from __future__ import annotations

import base64
import json
import struct
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals" / "multimodal"))
sys.path.insert(0, str(ROOT / "examples" / "multimodal_gate"))

import build_cases  # noqa: E402
import jev_image  # noqa: E402
import run  # noqa: E402
from make_pdf import LINES, build_pdf  # noqa: E402
from questions import QUESTIONS  # noqa: E402

from paved_gate.evaluator.base import EvaluatorError, MediaPart  # noqa: E402


def obj(v: object) -> dict[str, object]:
    assert isinstance(v, dict)
    return v


def lst(v: object) -> list[object]:
    assert isinstance(v, list)
    return v


def png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    raw = b"".join(b"\x00" + b"\xff" * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


# ------------------------------------------------------------------------------------------------ scoring


def answers(synthetic: float, choice: str, rejected2: float, score: float) -> dict[str, object]:
    return {
        "synthetic_label": {"type": "noul", "noul": synthetic},
        "overall_decision": {"type": "choice", "choice": choice, "probabilities": {choice: 1.0}, "confidence": 1.0},
        "rejected_exactly_2": {"type": "noul", "noul": rejected2},
        "legibility": {"type": "score", "score": score, "probabilities": {}, "confidence": 1.0},
    }


def test_predicted_thresholds() -> None:
    p = run.predicted(answers(0.5, "reject", 0.49, 1.0))
    assert p["synthetic_label"] is True  # 0.5 counts as yes
    assert p["rejected_exactly_2"] is False
    assert p["legibility_level"] == 2.0 and p["legibility_ok"] is False  # score 1.0 is level 2 of 5
    assert run.predicted(answers(0.1, "accept", 0.9, 2.0))["legibility_ok"] is True  # level 3


def test_record_view_reads_the_records() -> None:
    records = [
        {"namespace": "synthetic_oc_agg1", "decision": "reject"},
        {"namespace": "synthetic_oc_agg1", "decision": "accept"},
        {"namespace": "synthetic_oc_agg1", "decision": "reject"},
    ]
    assert run.record_view(json.dumps(records)) == {
        "synthetic_label": True,
        "overall_decision": "reject",
        "rejected_exactly_2": True,
    }
    demo = [{"namespace": "demo-oc-seq2", "decision": "needs_review"}, {"namespace": "demo", "decision": "accept"}]
    assert run.record_view(json.dumps(demo)) == {
        "synthetic_label": False,
        "overall_decision": "needs_review",
        "rejected_exactly_2": False,
    }


def case(cid: str, edits: list[str], labels: dict[str, object], **extra: object) -> dict[str, object]:
    return {
        "id": cid,
        "text_key": "r1",
        "mime_type": "image/png",
        "edits": edits,
        "labels": labels,
        "control": False,
        "counted": True,
        "arms": ["gemini_decision", "jev_text"],
    } | extra


def call(cid: str, arm: str, a: dict[str, object], tokens: int, usage: object) -> dict[str, object]:
    return {
        "case": cid,
        "arm": arm,
        "at": "2026-11-18T00:00:00+00:00",
        "latency_ms": 100.0,
        "input_tokens": tokens,
        "answers": a,
        "usage": usage,
    }


def test_build_report_groups_and_counts() -> None:
    records = [{"namespace": "synthetic_oc", "decision": "reject"}] * 2
    manifest: dict[str, object] = {
        "record_texts": {"r1": json.dumps(records)},
        "cases": [
            case(
                "orig",
                [],
                {
                    "synthetic_label": True,
                    "overall_decision": "reject",
                    "rejected_exactly_2": True,
                    "legibility_ok": True,
                },
            ),
            case(
                "banner",
                ["banner"],
                {
                    "synthetic_label": True,
                    "overall_decision": "accept",
                    "rejected_exactly_2": True,
                    "legibility_ok": True,
                },
            ),
            case("blur", ["blur"], {"legibility_ok": False}, control=True),
            case("tiny", ["4pt"], {}, control=True, counted=False),
        ],
    }
    gem_usage = {"input_tokens_by_modality": [{"modality": "text", "tokens": 90}, {"modality": "image", "tokens": 10}]}
    reads_image = answers(0.9, "reject", 0.9, 3.0)
    calls = {
        "orig|gemini_decision": call("orig", "gemini_decision", reads_image, 100, gem_usage),
        "orig|jev_text": call("orig", "jev_text", reads_image, 50, {}),
        "banner|gemini_decision": call("banner", "gemini_decision", answers(0.9, "accept", 0.9, 3.0), 100, gem_usage),
        "banner|jev_text": call("banner", "jev_text", reads_image, 50, {}),  # follows the records: wrong
        "blur|gemini_decision": call("blur", "gemini_decision", answers(0.9, "reject", 0.9, 0.5), 100, gem_usage),
        "blur|jev_text": call("blur", "jev_text", reads_image, 50, {}),
        "tiny|gemini_decision": call("tiny", "gemini_decision", reads_image, 100, gem_usage),
        "tiny|jev_text": call("tiny", "jev_text", reads_image, 50, {}),
    }
    r = run.build_report(manifest, calls)
    counts = r["counts"]
    assert isinstance(counts, dict)
    assert counts["unedited"]["gemini_decision"]["overall_decision"] == [1, 1]
    assert counts["edited"]["gemini_decision"]["overall_decision"] == [1, 1]
    assert counts["edited"]["jev_text"]["overall_decision"] == [0, 1]
    # only the banner question disagrees with the records, so only it enters the conflict split
    assert counts["edited_conflict"]["jev_text"] == {"overall_decision": [0, 1]}
    assert counts["controls"]["gemini_decision"] == {"legibility_ok": [1, 1]}
    assert counts["controls"]["jev_text"] == {"legibility_ok": [0, 1]}
    assert "recorded_only" not in counts  # the 4 pt case is recorded, never counted
    assert [obj(row)["group"] for row in lst(r["rows"])] == ["unedited", "edited", "controls", "recorded_only"]
    assert obj(obj(r["gemini_input_tokens_by_modality"])["image (PNG cases)"])["total"] == 40
    assert r["calls"] == {"made": 8, "expected": 8}


def test_jev_image_tokens_are_the_arm_difference() -> None:
    manifest: dict[str, object] = {
        "record_texts": {"r1": json.dumps([{"namespace": "x", "decision": "accept"}])},
        "cases": [case("c", [], {"legibility_ok": True}, arms=["jev_image", "jev_text"])],
    }
    a = answers(0.1, "accept", 0.1, 3.0)
    calls = {"c|jev_image": call("c", "jev_image", a, 6437, {}), "c|jev_text": call("c", "jev_text", a, 1603, {})}
    measured = run.build_report(manifest, calls)["jev_image_tokens_measured"]
    assert isinstance(measured, dict)
    assert measured["median"] == 4834 and measured["cases"] == 1


# ------------------------------------------------------------------------------------------------ Jev image body


def test_jev_image_body_follows_the_documented_shape() -> None:
    image = MediaPart(mime_type="image/png", data=png(1200, 1568), name="r1.png")
    body = jev_image.build_body("jev-1.14.0", '{"record": 1}', image, QUESTIONS)
    assert set(body) == {"model", "state", "questions", "assets"}
    assert body["model"] == "jev-1.14.0"
    assert body["state"] == {"record": '{"record": 1}', "report": jev_image.REF}
    asset = obj(obj(body["assets"])[jev_image.REF])
    assert asset["type"] == "image" and asset["format"] == "png"
    data = asset["data_base64"]
    assert isinstance(data, str) and base64.b64decode(data) == image.data  # standard base64, no data: prefix
    assert {k: obj(q)["type"] for k, q in obj(body["questions"]).items()} == {
        "synthetic_label": "noul",
        "overall_decision": "choice",
        "rejected_exactly_2": "noul",
        "legibility": "score",
    }
    assert jev_image.IMAGES_HEADER == "TypeSafe-Danger-Zone-Images"


@pytest.mark.parametrize("size", [(1569, 1000), (279, 1000), (1568, 1568)])
def test_jev_image_limits_are_checked_before_sending(size: tuple[int, int]) -> None:
    image = MediaPart(mime_type="image/png", data=png(*size), name="big.png")
    if size == (1568, 1568):  # within both sides and 2,458,624 pixels
        jev_image.build_body("jev-1.14.0", "{}", image, QUESTIONS)
        return
    with pytest.raises(EvaluatorError, match="limits"):
        jev_image.build_body("jev-1.14.0", "{}", image, QUESTIONS)


def test_jev_image_refuses_pdf_and_a_colliding_reference() -> None:
    pdf = MediaPart(mime_type="application/pdf", data=b"%PDF-1.4", name="x.pdf")
    with pytest.raises(EvaluatorError, match="PNG only"):
        jev_image.build_body("jev-1.14.0", "{}", pdf, QUESTIONS)
    image = MediaPart(mime_type="image/png", data=png(400, 400), name="x.png")
    with pytest.raises(EvaluatorError, match="reference"):
        jev_image.build_body("jev-1.14.0", f'{{"note": "{jev_image.REF}"}}', image, QUESTIONS)


# ------------------------------------------------------------------------------------------------ image only PDF


def test_image_only_pdf_has_no_text_layer() -> None:
    w, h = build_cases.PDF_RASTER
    gray = bytes((x * 3 + y) % 256 for y in range(h) for x in range(w))
    data = build_cases.image_pdf(gray, w, h)
    assert data.startswith(b"%PDF-1.4") and data.rstrip().endswith(b"%%EOF")
    assert b"/Font" not in data and b"/Info" not in data and b"/Metadata" not in data
    content = data.split(b"5 0 obj", 1)[1]
    assert b"BT" not in content and b"Tj" not in content  # no text operators in the page content
    assert b"/Subtype /Image" in data and f"/Width {w} /Height {h}".encode() in data
    stream = data.split(b"stream\n", 1)[1].split(b"\nendstream", 1)[0]
    assert zlib.decompress(stream) == gray  # the image is exactly the pixels given
    xref = int(data.rsplit(b"startxref\n", 1)[1].split(b"\n", 1)[0])
    assert data[xref:].startswith(b"xref")


def test_over_white_composites_transparent_pixels() -> None:
    assert build_cases.over_white(bytes([0, 0, 0, 0])) == 255  # transparent page background becomes white
    assert build_cases.over_white(bytes([0, 0, 0, 255])) == 0  # opaque black text stays black
    assert build_cases.over_white(bytes([0, 0, 0, 128])) == 127
    assert build_cases.over_white(bytes([255, 255, 255])) == 255  # 24 bit pixels have no alpha


def test_pdf_builder_default_output_is_unchanged() -> None:
    fixture = Path(__file__).parent / "fixtures" / "multimodal" / "synthetic_policy.pdf"
    assert build_pdf(LINES) == fixture.read_bytes()
