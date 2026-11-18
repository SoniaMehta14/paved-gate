# Evaluator benchmark results

Results of the Jev and Gemini Decision 1 comparison run by
[`scripts/benchmark_evaluators.py`](../../scripts/benchmark_evaluators.py). The numbers come from a single live run on
November 16, 2026, against the real Jev and Gemini Decisions APIs. Gemini was on the free tier and its calls were paced
to 45,000 input tokens per minute; Jev was not paced. The run was interrupted once by a Gemini rate limit and resumed
from its checkpoint, so some calls were timed in the first session and the rest in the second.

| File | What it is |
|---|---|
| [`evaluators-live-20261116T192356Z.md`](evaluators-live-20261116T192356Z.md) | The report: tables for support ticket classification, Harness² edit adoption and high fanout, with the stamps and notes |
| [`evaluators-live-20261116T192356Z.json`](evaluators-live-20261116T192356Z.json) | The same report as JSON, with every stamp (mode, date, model IDs, SDK version, settings, retry policies, pacing, tier, resume) |
| [`evaluators-live-20261116T192356Z.calls.jsonl`](evaluators-live-20261116T192356Z.calls.jsonl) | One record per API call: workload, evaluator, item ID, latency, retries, reported tokens, and the predicted and true labels or scores. It holds no ticket text, keys or paths |

| [`index.html`](index.html) | The Pages dashboard for both runs, built by [`scripts/evaluators_dashboard.py`](../../scripts/evaluators_dashboard.py) from the JSON files here |
| [`multimodal-live-20261118.json`](multimodal-live-20261118.json) | The live multimodal test of November 18, 2026, sanitized: case IDs, edit names, file hashes, labels, what the records say, every answer with its probability, latency, tokens and cost. It holds no images, PDFs, audit records, keys or paths |
| [`multimodal-live-20261118.md`](multimodal-live-20261118.md) | The same multimodal results as Markdown, with the caveats |

The multimodal files are written by [`evals/multimodal/publish.py`](../../evals/multimodal/publish.py) from the raw run results,
which stay out of git. The edited report images and PDF variants are not committed either:
[`evals/multimodal/build_cases.py`](../../evals/multimodal/build_cases.py) regenerates them from the published report
pages and [`examples/multimodal_gate/make_pdf.py`](../../examples/multimodal_gate/make_pdf.py), and the `sha256` of
each case in the JSON lets you check a rebuild.

The ticket items are IDs into the public Bitext sample already in [`data/eval/`](../../data/eval/) (CDLA Sharing 1.0).
The Harness² cases and the fanout payload are synthetic.

To rebuild the report from the calls file, with no API calls:

```bash
uv sync
uv run python scripts/benchmark_evaluators.py --rebuild docs/benchmarks/evaluators-live-20261116T192356Z.json
```

This rewrites the JSON and Markdown next to the calls file. Every value comes out the same; only the rebuild stamp
changes.
