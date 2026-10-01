"""Build the fixed, versioned support-ticket sample from the Bitext dataset.

    uv run python evals/support_tickets/build_sample.py                  # 40 per intent (default)
    uv run python evals/support_tickets/build_sample.py --per-intent 30

Downloads the full CSV at a pinned Hugging Face revision into data/cache/ (gitignored),
verifies its sha256, draws a stratified sample (the same number of tickets per intent,
seeded), and writes data/eval/<name>.jsonl plus a manifest recording every choice.
The eval refuses to run if the sample file no longer matches its manifest hash.

Dataset: Bitext customer support LLM chatbot training dataset, (c) Bitext Innovations, 2024,
licensed CDLA-Sharing-1.0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from intents import require_descriptions

ROOT = Path(__file__).resolve().parents[2]
DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
REVISION = "430d1a89bd93bd1fa23c16f29dd53e73f0087443"
FILENAME = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
SOURCE_URL = f"https://huggingface.co/datasets/{DATASET}"
DOWNLOAD_URL = f"{SOURCE_URL}/resolve/{REVISION}/{FILENAME}"
CSV_SHA256 = "6f81102b0100b97b8468eb04368033a23206bf1fde9d53500d5806ec1001a434"
LICENSE = "CDLA-Sharing-1.0"
ATTRIBUTION = "(c) Bitext Innovations, 2024"

DEFAULT_CACHE = ROOT / "data" / "cache" / "bitext_27k_v11.csv"
DEFAULT_SAMPLE = ROOT / "data" / "eval" / "bitext_support_sample_v1.jsonl"


@dataclass(frozen=True)
class Ticket:
    id: str
    row_index: int
    text: str
    intent: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(sample: Path) -> Path:
    return sample.with_name(sample.stem + ".manifest.json")


def ensure_dataset(cache: Path) -> Path:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {DOWNLOAD_URL}")
        with httpx.stream("GET", DOWNLOAD_URL, follow_redirects=True, timeout=120) as resp:
            resp.raise_for_status()
            with cache.open("wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
    digest = sha256_file(cache)
    if digest != CSV_SHA256:
        raise ValueError(f"{cache} sha256 {digest} does not match pinned {CSV_SHA256}; delete it and re-run")
    return cache


def read_rows(csv_path: Path) -> list[Ticket]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return [
            Ticket(id=f"bitext-{i}", row_index=i, text=row["instruction"], intent=row["intent"])
            for i, row in enumerate(csv.DictReader(f))
        ]


def stratified_sample(rows: list[Ticket], per_intent: int, seed: int) -> list[Ticket]:
    """Up to `per_intent` tickets from every intent, chosen with a seeded RNG (deterministic)."""
    if per_intent < 1:
        raise ValueError("per_intent must be at least 1")
    by_intent: dict[str, list[Ticket]] = defaultdict(list)
    for t in rows:
        by_intent[t.intent].append(t)
    rng = random.Random(seed)
    picked: list[Ticket] = []
    for intent in sorted(by_intent):
        group = sorted(by_intent[intent], key=lambda t: t.row_index)
        picked.extend(rng.sample(group, min(per_intent, len(group))))
    return sorted(picked, key=lambda t: (t.intent, t.row_index))


def write_sample(tickets: list[Ticket], sample: Path, *, per_intent: int, seed: int) -> dict[str, object]:
    sample.parent.mkdir(parents=True, exist_ok=True)
    with sample.open("w", encoding="utf-8", newline="\n") as f:
        for t in tickets:
            f.write(json.dumps(asdict(t), ensure_ascii=False, sort_keys=True) + "\n")
    counts: dict[str, int] = defaultdict(int)
    for t in tickets:
        counts[t.intent] += 1
    manifest: dict[str, object] = {
        "dataset": DATASET,
        "source_url": SOURCE_URL,
        "revision": REVISION,
        "file": FILENAME,
        "file_sha256": CSV_SHA256,
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "per_intent": per_intent,
        "seed": seed,
        "n": len(tickets),
        "intents": len(counts),
        "counts": dict(sorted(counts.items())),
        "sample_file": sample.name,
        "sample_sha256": sha256_file(sample),
    }
    manifest_path(sample).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_sample(sample: Path) -> tuple[list[Ticket], dict[str, object]]:
    """Load the versioned sample, refusing if it no longer matches its manifest."""
    manifest = json.loads(manifest_path(sample).read_text(encoding="utf-8"))
    digest = sha256_file(sample)
    if digest != manifest["sample_sha256"]:
        raise ValueError(f"{sample.name} sha256 {digest} does not match its manifest; rebuild the sample")
    with sample.open(encoding="utf-8") as f:
        tickets = [Ticket(**json.loads(line)) for line in f if line.strip()]
    return tickets, manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-intent", type=int, default=40, help="tickets per intent (30-50 recommended)")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = ap.parse_args()

    rows = read_rows(ensure_dataset(args.cache))
    require_descriptions({t.intent for t in rows})
    tickets = stratified_sample(rows, args.per_intent, args.seed)
    manifest = write_sample(tickets, args.out, per_intent=args.per_intent, seed=args.seed)
    print(f"Wrote {args.out} ({manifest['n']} tickets, {manifest['intents']} intents)")
    print(f"sha256 {manifest['sample_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
