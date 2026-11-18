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
import pdf_png  # noqa: E402
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


# ------------------------------------------------------------------------------------------------ PDF as PNG


def full_answers(
    synthetic: float, probs: dict[str, float], rejected2: float, levels: dict[str, float]
) -> dict[str, object]:
    choice = max(probs, key=probs.__getitem__)
    score = sum(int(k) * p for k, p in levels.items())
    return {
        "synthetic_label": {"type": "noul", "noul": synthetic},
        "overall_decision": {"type": "choice", "choice": choice, "probabilities": probs, "confidence": probs[choice]},
        "rejected_exactly_2": {"type": "noul", "noul": rejected2},
        "legibility": {"type": "score", "score": score, "probabilities": levels, "confidence": max(levels.values())},
    }


def test_probabilities_put_every_answer_on_the_half_cutoff_scale() -> None:
    a = full_answers(
        0.55,
        {"accept": 0.1, "needs_review": 0.7, "reject": 0.2},
        0.05,
        {"0": 0.1, "1": 0.3, "2": 0.25, "3": 0.25, "4": 0.1},
    )
    p = pdf_png.probabilities(a)
    assert p["synthetic_label"] == 0.55
    assert p["overall_decision"] == 0.7  # the chosen option's probability
    assert p["rejected_exactly_2"] == 0.05
    assert p["legibility_ok"] == pytest.approx(0.6)  # levels 3, 4 and 5 are indexes 2, 3 and 4


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_pdf_png_report_marks_close_calls_and_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cases_dir, out = tmp_path / "cases", tmp_path / "pdf_png"
    manifest: dict[str, object] = {
        "record_texts": {"pdf": json.dumps([{"namespace": "synthetic_oc", "decision": "needs_review"}])},
        "cases": [
            case(
                "pdf_a",
                ["decision_accept"],
                {
                    "synthetic_label": True,
                    "overall_decision": "accept",
                    "rejected_exactly_2": False,
                    "legibility_ok": True,
                },
                text_key="pdf",
                mime_type="application/pdf",
            ),
            case("pdf_tiny", ["4pt"], {}, text_key="pdf", mime_type="application/pdf", control=True, counted=False),
        ],
    }
    cases_dir.mkdir()
    (cases_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(run, "CASES", cases_dir)
    monkeypatch.setattr(pdf_png, "OUT", out)
    monkeypatch.setattr(pdf_png, "MAIN_CALLS", tmp_path / "run" / "calls.jsonl")
    main_settings = run.settings_hash((cases_dir / "manifest.json").read_bytes())
    rendered: list[dict[str, object]] = [{"id": "pdf_a", "sha256": "1"}, {"id": "pdf_tiny", "sha256": "2"}]
    settings = pdf_png.digest(rendered)
    sure = full_answers(0.95, {"accept": 0.98, "needs_review": 0.01, "reject": 0.01}, 0.02, {"4": 1.0})
    near = full_answers(0.55, {"accept": 0.02, "needs_review": 0.97, "reject": 0.01}, 0.45, {"1": 0.5, "2": 0.5})

    def rec(cid: str, arm: str, a: dict[str, object], s: str, tokens: int) -> dict[str, object]:
        return {
            "settings": s,
            "case": cid,
            "arm": arm,
            "at": "2026-11-18T00:00:00+00:00",
            "latency_ms": 100.0,
            "input_tokens": tokens,
            "answers": a,
            "usage": {"input_tokens_by_modality": []},
        }

    write_jsonl(
        tmp_path / "run" / "calls.jsonl",
        [
            rec("pdf_a", "gemini_decision", sure, main_settings, 2114),
            rec("pdf_a", "jev_text", near, main_settings, 1603),
            rec("pdf_tiny", "gemini_decision", near, main_settings, 2114),
            rec("pdf_tiny", "jev_text", near, main_settings, 1603),
        ],
    )
    write_jsonl(
        out / "calls.jsonl",
        [
            rec("pdf_a", "gemini_png", sure, settings, 2573),
            rec("pdf_a", "jev_png", sure, settings, 6549),
            rec("pdf_tiny", "gemini_png", near, settings, 2573),
            rec("pdf_tiny", "jev_png", near, settings, 6549),
        ],
    )
    r = pdf_png.build_report(manifest, rendered, settings)
    counts = obj(r["counts"])
    assert obj(obj(counts["edited"])["jev_text"])["overall_decision"] == [0, 1]  # follows the record
    assert obj(obj(counts["edited"])["jev_png"])["overall_decision"] == [1, 1]
    assert "recorded_only" not in counts
    # near: synthetic 0.55, rejected 0.45 and legibility 0.5 are close; the choice at 0.97 is not
    assert obj(r["close_calls"]) == {"gemini_pdf": 0, "gemini_png": 0, "jev_png": 0, "jev_text": 3}
    row = obj(lst(r["rows"])[0])
    cell = obj(obj(obj(row["columns"])["jev_text"])["synthetic_label"])
    assert cell == {"answer": True, "p": 0.55, "close": True, "correct": True}
    assert "(close)" in pdf_png.cell(cell)
    assert obj(r["jev_image_tokens"])["median"] == 4946
    assert "none was refused" in str(r["limit_note"])


def test_close_call_boundary() -> None:
    def close(p: float) -> bool:
        return abs(p - 0.5) <= pdf_png.CLOSE

    assert close(0.4) and close(0.6) and close(0.5)
    assert not close(0.39) and not close(0.61)
