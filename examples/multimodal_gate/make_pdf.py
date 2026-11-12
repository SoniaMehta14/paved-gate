"""Write the synthetic one page PDF used by the multimodal gate tests.

    uv run python examples/multimodal_gate/make_pdf.py            # writes tests/fixtures/multimodal/synthetic_policy.pdf

The PDF is assembled by hand: one page of Helvetica text, no Info dictionary and no XMP metadata, so it
carries no producer, author, creation date or file path. The output is byte for byte reproducible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "tests" / "fixtures" / "multimodal" / "synthetic_policy.pdf"
LINES = [
    "SYNTHETIC TEST DOCUMENT: invented content, not real data",
    "",
    "Harness edit review summary",
    "Overall decision: needs review",
    "",
    "1. A skill was added: skills/ledger_check/SKILL.md",
    "2. guardrails.md gained one trigger line for that skill.",
    "3. The policy holds trigger lines for human review.",
    "",
    "Written for the Paved Gate multimodal gate tests.",
]


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(lines: list[str]) -> bytes:
    ops = ["BT", "/F1 14 Tf", "72 740 Td", "18 TL"]
    for line in lines:
        ops.append(f"({_escape(line)}) Tj T*")
    ops.append("ET")
    stream = "\n".join(ops).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> "
        b"/Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, nargs="?", default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(build_pdf(LINES))
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
