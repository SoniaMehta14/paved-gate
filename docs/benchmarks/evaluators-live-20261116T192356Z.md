# Jev vs Gemini Decision 1 on governance workloads

**results (live)**

* Mode: live
* Date (UTC): 2026-11-16T19:23:56+00:00
* Models: Jev `jev-1.14.0`, Gemini Decision 1 `gemini-decision-1`
* Settings: {"mode": "live", "workloads": ["tickets", "harness2", "fanout"], "limit": null, "fanout_runs": 20, "fanout_sizes": [1, 10, 25, 50], "harness2_cases": null, "gemini_tpm": 45000, "gemini_tier": "free", "resumed_from_checkpoint": true, "calls_restored": 1180}
* Rebuilt: 2026-11-16T20:01:35+00:00 from evaluators-live-20261116T192356Z.calls.jsonl (2380 recorded calls, no API calls)

| Arm | Status | Note |
| :- | :- | :- |
| Jev | ok | real Jev API |
| Gemini Decision 1 | ok | real Gemini Decisions API |

## Support ticket classification

| Arm | Calls | Errors | Accuracy | Accuracy 95% CI | Macro F1 | ECE (top class probability) | ECE 95% CI | ECE (own confidence field) | p50 | p95 | Latency runs | Excluded (retried) |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| Jev | 1080 | 0 | 0.9222 | 0.9065 to 0.938 | 0.922 | 0.019 | 0.0124 to 0.0357 | 0.0196 | 89.24 ms | 183.47 ms | 1080 | 0 |
| Gemini Decision 1 | 1080 | 0 | 0.9398 | 0.925 to 0.9537 | 0.9389 | 0.0267 | 0.0188 to 0.0411 | 0.0267 | 256.79 ms | 371.37 ms | 1080 | 0 |

Decision agreement between arms: 1007 of 1080 tickets answered by both (0.9324).

Paired accuracy, exact two sided McNemar test over 1080 tickets answered by both arms:

| Outcome | Tickets |
| :- | -: |
| Both correct | 970 |
| Only Jev correct | 26 |
| Only Gemini Decision 1 correct | 45 |
| Both wrong | 39 |
| p value (discordant pairs only) | 0.031927 |

## Harness² edit adoption (synthetic, labelled by construction)

| Arm | Guardrail weakening caught | Empty output caught | Benign edits flagged | Errors | p50 | p95 | Latency runs | Excluded (retried) |
| :- | -: | -: | -: | -: | -: | -: | -: | -: |
| Jev | 10 of 10 | 10 of 10 | 0 of 10 | 0 | 79.46 ms | 141.05 ms | 30 | 0 |
| Gemini Decision 1 | 10 of 10 | 10 of 10 | 0 of 10 | 0 | 249.53 ms | 336.91 ms | 30 | 0 |

Agreement on flag or no flag: 30 of 30 cases.

## High fanout policy check

| Arm | Questions per call | Calls | p50 per call | p95 per call | p50 per decision | Latency runs | Excluded (retried) | Median input tokens | Cost per 1,000 decisions |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| Jev | 1 | 20 | 75.54 ms | 142.66 ms | 75.54 ms | 20 | 0 | 591 | 0.024822 USD |
| Jev | 10 | 20 | 88.34 ms | 177.96 ms | 8.83 ms | 20 | 0 | 1087 | 0.004565 USD |
| Jev | 25 | 20 | 95.01 ms | 162.01 ms | 3.8 ms | 20 | 0 | 2037 | 0.003422 USD |
| Jev | 50 | 20 | 102.91 ms | 184.48 ms | 2.06 ms | 20 | 0 | 3645 | 0.003062 USD |
| Gemini Decision 1 | 1 | 20 | 254.19 ms | 290.25 ms | 254.19 ms | 20 | 0 | 452 | 0.0452 USD |
| Gemini Decision 1 | 10 | 20 | 275.87 ms | 307.01 ms | 27.59 ms | 20 | 0 | 1044 | 0.01044 USD |
| Gemini Decision 1 | 25 | 20 | 317.47 ms | 425.82 ms | 12.7 ms | 20 | 0 | 2124 | 0.008496 USD |
| Gemini Decision 1 | 50 | 20 | 420.88 ms | 597.96 ms | 8.42 ms | 20 | 0 | 3927 | 0.007854 USD |

## Notes

* Timing method: 3 warmup calls per arm, not counted; requests sent sequentially, one at a time; 1 run per item for tickets and harness2, --fanout-runs runs per cell for fanout; wall clock time around the adapter's evaluate call, so client overhead (serialization, TLS, response parsing) is included; 10 s timeout for both arms. Calls that needed a retry are excluded from p50 and p95 (they still count for accuracy, F1, ECE and agreement), and the excluded count is shown.
* Calibration: ECE over the top class probability (the largest value in the answer's probabilities), 10 equal width bins on [0, 1], weighted by the share of answers in each bin: sum of |accuracy minus mean top probability| times bin share. Computed the same way for both arms. Each API's own confidence field is reported separately and is not comparable: Jev normalizes choice confidence as (p_max minus 1/n) / (1 minus 1/n), Gemini reports p_max.
* Retry policies differ: Jev: up to 1 retry on HTTP 429 and 529, jittered backoff capped at a quarter of the timeout; Gemini Decision 1: up to 2 retries on 429 and 5xx, exponential backoff 0.5 s doubling up to 8 s, plus up to 10% jitter; SDK retries off.
* Documented limits: Jev: no question cap; the request must fit 64k tokens per request and 32k per question, up to 255 options per choice; Gemini Decision 1: 1 to 1,000 questions; 262,144 tokens across input and all questions, up to 255 options per choice.
* Harness² cases are 30 small synthetic edits labelled by construction. Read them as counts; they are not a basis for ranking the evaluators.
* Live runs send only the public Bitext sample (CDLA Sharing 1.0) and synthetic payloads. Gemini free tier content may be used by Google to improve its products.
* Gemini calls were paced to 45,000 input tokens per minute over a sliding 60 second window of the tokens each response reported. Waits happen before a call, outside the timed section; they totalled 672 s. Jev calls were not paced.
* Gemini usage tier: free. The free tier allows 50,000 input tokens per minute, and Google may use free tier content to improve its products.
* Resumed from a checkpoint: 1180 calls came from an earlier session with the same settings and were not repeated, so latency for those calls was measured in that session.
* Costs use the list prices in scripts/pricing.py and the input tokens each API reports. Jev also reports output tokens, but TypeSafe bills input only, so they are not priced; Gemini reports 0 output tokens.
* Statistics: accuracy and ECE intervals are percentile bootstrap 95% intervals over 2,000 resamples of tickets with replacement (seed 0). The paired accuracy comparison is an exact two sided McNemar test on the tickets both arms answered; only the discordant pairs enter the p value.
* This run used the Gemini free tier, so its Gemini calls cost nothing. The cost column uses the paid tier list price (0.10 USD per million input tokens).
