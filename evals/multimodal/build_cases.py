"""Build the synthetic cases for the live multimodal test, with labels known by construction.

    uv run python evals/multimodal/build_cases.py            # writes results/multimodal/cases/

Image cases are re-rendered at 1200 x 1568 (within Jev's documented image limits) from the published
Harness² report HTML in docs/harness2/, with one edit applied by a script after the page renders.
PDF cases come from examples/multimodal_gate/make_pdf.py with edited lines. Each case carries the audit
records the source report was built from, as JSON text. Labels describe what the artifact shows,
not what the records say.

Needs Google Chrome (set CHROME to override the path). Makes no network requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "multimodal_gate"))
from make_pdf import LINES, build_pdf  # noqa: E402

OUT = ROOT / "results" / "multimodal" / "cases"
RECORDS = ROOT / "results" / "harness2" / "records"
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
WIDTH, HEIGHT = 1200, 1568
DECISION_TEXT = {"accept": "Accept", "needs_review": "Needs review", "reject": "Reject"}

DEMO_RECORDS = [
    "demo-oc-agg2__ledger-1__agg1.json",
    "demo-oc-agg2__ledger-2__agg2.json",
    "demo-oc-seq2__ledger-1__c1.json",
    "demo-oc-seq2__ledger-2__c2.json",
]
# report key: (HTML file, record file names the published page was built from, overall decision, rejected count)
REPORTS: dict[str, tuple[str, list[str], str, int]] = {
    "r1": (
        "synthetic_fixture.html",
        [
            "synthetic_fixture/synthetic_oc_agg1__guardrail_edit__agg1.json",
            "synthetic_fixture/synthetic_oc_agg1__prompt_edit__agg1.json",
            "synthetic_fixture/synthetic_oc_agg1__skill_with_trigger__agg1.json",
            "synthetic_fixture/synthetic_oc_agg1__subagent_edit__agg1.json",
            "synthetic_fixture/synthetic_oc_agg1__trigger_with_extra_instruction__agg1.json",
        ],
        "reject",
        2,
    ),
    "r2": ("default_policy.html", [f"default_policy/{n}" for n in DEMO_RECORDS], "reject", 4),
    "r3": ("example_policy.html", [f"example_policy/{n}" for n in DEMO_RECORDS], "accept", 0),
}
PDF_RECORD = "synthetic_fixture/synthetic_oc_agg1__skill_with_trigger__agg1.json"

# Run after the report's own script has rendered the page. Each edit changes exactly what its name says.
EDIT_JS = """
const tileBy = (label) => [...document.querySelectorAll(".tile")]
  .find((t) => t.querySelector(".l")?.textContent.trim() === label);
const EDITS = {
  banner: (d) => { const t = tileBy("overall decision"); t.querySelector(".v").textContent = d.text;
    t.className = t.className.replace(/\\bt-[a-z_]+/, "t-" + d.value); },
  rejected: (d) => { tileBy("rejected").querySelector(".v").textContent = String(d.value); },
  unmark: () => {
    document.querySelectorAll(".badge").forEach((b) => {
      if (/^synthetic fixture$/i.test(b.textContent.trim())) b.remove();
    });
    document.querySelectorAll(".banner").forEach((b) => b.remove());
    const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    for (let n = walk.nextNode(); n; n = walk.nextNode())
      n.nodeValue = n.nodeValue.replace(/Synthetic guardrail fixture/g, "Guardrail fixture")
        .replace(/synthetic_/gi, "fixture_");
  },
  blur: () => { document.body.style.filter = "blur(10px)"; },
};
for (const [name, arg] of __EDITS__) EDITS[name](arg);
"""


@dataclass
class Case:
    id: str
    source: str
    edits: list[str]
    artifact: str
    mime_type: str
    sha256: str
    bytes: int
    records: list[str]
    labels: dict[str, object]
    control: bool = False
    arms: list[str] = field(default_factory=list)
    counted: bool = True  # False: answers are recorded but not scored


def labels(synthetic: bool, decision: str, rejected: int | None, legible: bool = True) -> dict[str, object]:
    return {
        "synthetic_label": synthetic,
        "overall_decision": decision,
        "rejected_exactly_2": rejected == 2,
        "legibility_ok": legible,
    }


def png_text_chunks(data: bytes) -> list[str]:
    """Names of metadata chunks in a PNG (text, EXIF, time); a clean render has none."""
    found, pos = [], 8
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos : pos + 8])
        if kind in (b"tEXt", b"iTXt", b"zTXt", b"eXIf", b"tIME"):
            found.append(kind.decode())
        pos += 12 + length
    return found


def chrome(args: list[str], url: str) -> str:
    with tempfile.TemporaryDirectory() as profile:
        cmd = [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1"]
        cmd += [f"--user-data-dir={profile}", f"--window-size={WIDTH},{HEIGHT}", *args, url]
        return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60).stdout


def render(html_file: str, edits: list[tuple[str, object]], out: Path, work: Path) -> str:
    """Write the edited page, screenshot it, and return its visible text (scripts and styles dropped)."""
    page = (ROOT / "docs" / "harness2" / html_file).read_text(encoding="utf-8")
    script = EDIT_JS.replace("__EDITS__", json.dumps(edits))
    edited = work / f"{out.stem}.html"
    edited.write_text(page.replace("</body>", f"<script>{script}</script>\n</body>"), encoding="utf-8")
    url = edited.resolve().as_uri() + "#shot-light"
    chrome([f"--screenshot={out}"], url)
    dom = chrome(["--dump-dom"], url)
    body = re.sub(r"<(script|style)\b.*?</\1>", "", dom, flags=re.S)
    return re.sub(r"<[^>]+>", " ", body)


def record_text(names: list[str]) -> str:
    return json.dumps([json.loads((RECORDS / n).read_text(encoding="utf-8")) for n in names], sort_keys=True)


def build(out: Path) -> list[Case]:
    out.mkdir(parents=True, exist_ok=True)
    cases: list[Case] = []
    texts: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        image_specs: list[tuple[str, str, list[tuple[str, object]], dict[str, object], bool]] = []
        for key, (_, _, decision, rejected) in REPORTS.items():
            synthetic = key == "r1"
            image_specs.append((f"{key}_original", key, [], labels(synthetic, decision, rejected), False))
            for d in [x for x in DECISION_TEXT if x != decision]:
                image_specs.append(
                    (
                        f"{key}_banner_{d}",
                        key,
                        [("banner", {"value": d, "text": DECISION_TEXT[d]})],
                        labels(synthetic, d, rejected),
                        False,
                    )
                )
            new_count = 3 if rejected == 2 else 2
            image_specs.append(
                (
                    f"{key}_rejected_{new_count}",
                    key,
                    [("rejected", {"value": new_count})],
                    labels(synthetic, decision, new_count),
                    False,
                )
            )
            image_specs.append((f"{key}_blur", key, [("blur", None)], {"legibility_ok": False}, True))
        image_specs.append(("r1_unmarked", "r1", [("unmark", None)], labels(False, "reject", 2), False))
        image_specs.append(
            (
                "r1_unmarked_banner_accept",
                "r1",
                [("unmark", None), ("banner", {"value": "accept", "text": "Accept"})],
                labels(False, "accept", 2),
                False,
            )
        )
        for case_id, key, edits, lab, control in image_specs:
            html_file, record_names, _, _ = REPORTS[key]
            path = out / f"{case_id}.png"
            text = render(html_file, edits, path, work)
            check_render(case_id, edits, lab, text)
            data = path.read_bytes()
            meta = png_text_chunks(data)
            if meta:
                raise SystemExit(f"{case_id}: PNG carries metadata chunks {meta}")
            w, h = struct.unpack(">II", data[16:24])
            if (w, h) != (WIDTH, HEIGHT):
                raise SystemExit(f"{case_id}: rendered {w}x{h}, expected {WIDTH}x{HEIGHT}")
            texts[key] = record_text(record_names)
            cases.append(
                Case(
                    case_id,
                    html_file,
                    [e[0] for e in edits],
                    path.name,
                    "image/png",
                    hashlib.sha256(data).hexdigest(),
                    len(data),
                    record_names,
                    lab,
                    control,
                    ["gemini_decision", "jev_image", "jev_text"],
                )
            )

    pdf_specs: list[tuple[str, list[str], dict[str, object], float, bool]] = [
        ("pdf_original", LINES, labels(True, "needs_review", None), 14, False),
        ("pdf_decision_accept", swap(LINES, "needs review", "accept"), labels(True, "accept", None), 14, False),
        ("pdf_decision_reject", swap(LINES, "needs review", "reject"), labels(True, "reject", None), 14, False),
        ("pdf_unmarked", LINES[2:], labels(False, "needs_review", None), 14, False),
        (
            "pdf_trigger_count",
            swap(LINES, "one trigger line", "two trigger lines"),
            labels(True, "needs_review", None),
            14,
            False,
        ),
        ("pdf_4pt", LINES, {}, 4, True),  # recorded, not counted: vector text stays sharp when zoomed
    ]
    texts["pdf"] = record_text([PDF_RECORD])
    for case_id, lines, lab, size, control in pdf_specs:
        data = build_pdf(lines, font_size=size, leading=size * 18 / 14)
        if b"/Info" in data or b"/Metadata" in data:
            raise SystemExit(f"{case_id}: PDF carries metadata")
        path = out / f"{case_id}.pdf"
        path.write_bytes(data)
        cases.append(
            Case(
                case_id,
                "make_pdf.py",
                [] if case_id == "pdf_original" else [case_id[4:]],
                path.name,
                "application/pdf",
                hashlib.sha256(data).hexdigest(),
                len(data),
                [PDF_RECORD],
                lab,
                control,
                ["gemini_decision", "jev_text"],
                counted=case_id != "pdf_4pt",
            )
        )

    # The illegible PDF control: the original page as a low resolution image, with no text layer.
    data = image_pdf(*page_pixels(out / "pdf_original.pdf", PDF_RASTER))
    if b"/Font" in data or b"BT" in data.split(b"stream", 1)[0]:
        raise SystemExit("pdf_page_image: the PDF must not carry text")
    path = out / "pdf_page_image.pdf"
    path.write_bytes(data)
    cases.append(
        Case(
            "pdf_page_image",
            "make_pdf.py",
            [f"page image {PDF_RASTER[0]}x{PDF_RASTER[1]} px, no text layer"],
            path.name,
            "application/pdf",
            hashlib.sha256(data).hexdigest(),
            len(data),
            [PDF_RECORD],
            {"legibility_ok": False},
            True,
            ["gemini_decision", "jev_text"],
        )
    )

    for text in texts.values():
        if re.search(r"/Users/|/home/|/private/|[A-Za-z]:\\\\", text):
            raise SystemExit("a record holds an absolute path; not sending it")
    manifest = {
        "size": [WIDTH, HEIGHT],
        "record_texts": texts,
        "cases": [asdict(c) | {"text_key": "pdf" if c.mime_type == "application/pdf" else c.id[:2]} for c in cases],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return cases


PDF_RASTER = (77, 99)  # the 612 x 792 pt page at about 9 dpi; 14 pt text becomes under 2 px tall


def sips(*args: str) -> None:
    subprocess.run(["sips", *args], check=True, capture_output=True, timeout=60)


def page_pixels(pdf: Path, size: tuple[int, int]) -> tuple[bytes, int, int]:
    """Rasterise page 1 with macOS sips, shrink it to `size`, and return 8 bit grey pixels, top row first."""
    with tempfile.TemporaryDirectory() as tmp:
        png, bmp = Path(tmp) / "page.png", Path(tmp) / "page.bmp"
        sips("-s", "format", "png", str(pdf), "--out", str(png))
        sips("-z", str(size[1]), str(size[0]), str(png))
        sips("-s", "format", "bmp", str(png), "--out", str(bmp))
        raw = bmp.read_bytes()
    (offset,) = struct.unpack("<I", raw[10:14])
    width, height = struct.unpack("<ii", raw[18:26])
    (bpp,) = struct.unpack("<H", raw[28:30])
    if bpp not in (24, 32) or (width, abs(height)) != size:
        raise SystemExit(f"unexpected page raster: {width}x{height} at {bpp} bits")
    step = bpp // 8
    stride = (width * step + 3) & ~3
    rows = []
    for r in range(abs(height)):
        row = raw[offset + r * stride : offset + r * stride + width * step]
        rows.append(bytes(over_white(row[i : i + step]) for i in range(0, len(row), step)))
    if height > 0:  # bottom up
        rows.reverse()
    return b"".join(rows), width, abs(height)


def over_white(px: bytes) -> int:
    """Grey value of one BGR or BGRA pixel composited over a white page (sips leaves the page transparent)."""
    grey = 0.114 * px[0] + 0.587 * px[1] + 0.299 * px[2]
    alpha = px[3] / 255 if len(px) == 4 else 1.0
    return round(grey * alpha + 255 * (1 - alpha))


def image_pdf(gray: bytes, width: int, height: int) -> bytes:
    """One page PDF whose only content is a greyscale image scaled to the page; no fonts and no text."""
    image = zlib.compress(gray, 9)
    draw = b"q 612 0 0 792 0 0 cm /Im1 Do Q"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /XObject << /Im1 4 0 R >> >> "
        b"/Contents 5 0 R >>",
        f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} /ColorSpace /DeviceGray "
        f"/BitsPerComponent 8 /Filter /FlateDecode /Length {len(image)} >>\nstream\n".encode()
        + image
        + b"\nendstream",
        b"<< /Length " + str(len(draw)).encode() + b" >>\nstream\n" + draw + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def swap(lines: list[str], old: str, new: str) -> list[str]:
    out = [line.replace(old, new) for line in lines]
    if out == lines:
        raise SystemExit(f"edit {old!r} matched no line")
    return out


def check_render(case_id: str, edits: list[tuple[str, object]], lab: dict[str, object], text: str) -> None:
    """Confirm the rendered page shows what the labels say; the build stops on any mismatch."""
    words = " ".join(text.split())
    names = [e[0] for e in edits]
    if "blur" in names:
        return
    marked = "synthetic fixture" in words.lower()
    if lab["synthetic_label"] != marked:
        raise SystemExit(f"{case_id}: synthetic marking on page is {marked}, label says {lab['synthetic_label']}")
    if not lab["synthetic_label"] and re.search(r"synthetic", words, re.I):
        raise SystemExit(f"{case_id}: the word synthetic is still on the page")
    shown = re.search(r"(Accept|Needs review|Reject) overall decision", words)
    if not shown or shown.group(1) != DECISION_TEXT[str(lab["overall_decision"])]:
        raise SystemExit(f"{case_id}: banner shows {shown and shown.group(1)!r}")
    rejected = re.search(r"(\d+) rejected", words)
    if not rejected or (int(rejected.group(1)) == 2) != lab["rejected_exactly_2"]:
        raise SystemExit(f"{case_id}: rejected tile shows {rejected and rejected.group(1)!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    cases = build(args.out)
    print(f"{len(cases)} cases in {args.out}")
    for c in cases:
        print(f"  {c.id:28} {c.mime_type:16} {c.bytes:>8} B  {json.dumps(c.labels)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
