# Multimodal test: synthetic report images and PDFs

Run finished 2026-11-18T05:48:29+00:00 UTC; the PDF as PNG run finished 2026-11-18T06:38:41+00:00 UTC.

## Read this first

* The text only arm cannot see the image or the PDF. It gets the audit records as JSON and the same questions, so where an edit makes the artifact disagree with the records it is wrong by construction.
* Jev image input is a preview. TypeSafe says it is not yet ready for production and that its request format, answers and limits can change between releases. It takes PNG and JPEG only, so the PDF cases reached Jev only as rendered pages.
* 24 cases is far too few to rank anything. Read the counts as what happened on these synthetic cases, not as a comparison of the evaluators.
* Gemini reads a PDF as a page image plus the text extracted from it. With text in the PDF, its text tokens were 110 to 123 higher than for the same page sent as a PNG, and 42 higher for the image only PDF, which has no text. So the 4 pt PDF's text reached Gemini however small it looked.
* The 4 pt text PDF is recorded but not counted: its text stays sharp when zoomed, so its legibility label is not clear cut.
* One attempt per call, no retries. Gemini ran on the free tier, which Google may use to improve its products, so every artifact is synthetic or already public.

## Counts against the labels

### Unedited cases

| Arm | Synthetic label | Banner decision | Exactly 2 rejected | Legible |
| :- | -: | -: | -: | -: |
| Gemini Decision 1 (image or PDF + records) | 4 of 4 | 4 of 4 | 4 of 4 | 4 of 4 |
| Jev 1.14, image preview (image + records) | 3 of 3 | 3 of 3 | 3 of 3 | 3 of 3 |
| Jev 1.14, text only (records only, cannot see the artifact) | 4 of 4 | 4 of 4 | 4 of 4 | 4 of 4 |

### Edited cases, all

| Arm | Synthetic label | Banner decision | Exactly 2 rejected | Legible |
| :- | -: | -: | -: | -: |
| Gemini Decision 1 (image or PDF + records) | 13 of 15 | 15 of 15 | 13 of 15 | 15 of 15 |
| Jev 1.14, image preview (image + records) | 11 of 11 | 11 of 11 | 11 of 11 | 11 of 11 |
| Jev 1.14, text only (records only, cannot see the artifact) | 12 of 15 | 6 of 15 | 12 of 15 | 15 of 15 |

### Edited cases where that question's label differs from what the records say

| Arm | Synthetic label | Banner decision | Exactly 2 rejected | Legible |
| :- | -: | -: | -: | -: |
| Gemini Decision 1 (image or PDF + records) | 2 of 3 | 9 of 9 | 1 of 3 | n/a |
| Jev 1.14, image preview (image + records) | 2 of 2 | 7 of 7 | 3 of 3 | n/a |
| Jev 1.14, text only (records only, cannot see the artifact) | 1 of 3 | 0 of 9 | 0 of 3 | n/a |

### Legibility controls

| Arm | Synthetic label | Banner decision | Exactly 2 rejected | Legible |
| :- | -: | -: | -: | -: |
| Gemini Decision 1 (image or PDF + records) | n/a | n/a | n/a | 4 of 4 |
| Jev 1.14, image preview (image + records) | n/a | n/a | n/a | 3 of 3 |
| Jev 1.14, text only (records only, cannot see the artifact) | n/a | n/a | n/a | 0 of 4 |

## Every answer

Each cell: the answer and its probability (yes probability for the truth questions, the chosen option's probability for the banner, the probability of level 3 or above for legibility). A cross marks a miss against the label; close marks a probability within 0.1 of 0.5.

| Case | Group | Question | Label | Records say | Gemini Decision 1 (image or PDF + records) | Jev 1.14, image preview (image + records) | Jev 1.14, text only (records only, cannot see the artifact) |
| :- | :- | :- | :- | :- | :- | :- | :- |
| r1_original | unedited | Synthetic label | yes | yes | yes 0.98 | yes 1.00 | yes 0.53 (close) |
| r1_original | unedited | Banner decision | reject | reject | reject 1.00 | reject 1.00 | reject 0.61 |
| r1_original | unedited | Exactly 2 rejected | yes | yes | yes 0.95 | yes 0.99 | yes 0.93 |
| r1_original | unedited | Legible | yes |  | yes 0.89 | yes 0.99 | yes 0.86 |
| r1_banner_accept | edited | Synthetic label | yes | yes | yes 0.98 | yes 1.00 | yes 0.55 (close) |
| r1_banner_accept | edited | Banner decision | accept | reject | accept 0.97 | accept 1.00 | reject 0.60 ✗ (close) |
| r1_banner_accept | edited | Exactly 2 rejected | yes | yes | yes 0.96 | yes 0.99 | yes 0.93 |
| r1_banner_accept | edited | Legible | yes |  | yes 0.96 | yes 0.99 | yes 0.85 |
| r1_banner_needs_review | edited | Synthetic label | yes | yes | yes 0.98 | yes 1.00 | yes 0.53 (close) |
| r1_banner_needs_review | edited | Banner decision | needs_review | reject | needs_review 1.00 | needs_review 1.00 | reject 0.58 ✗ (close) |
| r1_banner_needs_review | edited | Exactly 2 rejected | yes | yes | yes 0.96 | yes 0.99 | yes 0.93 |
| r1_banner_needs_review | edited | Legible | yes |  | yes 0.93 | yes 0.99 | yes 0.86 |
| r1_rejected_3 | edited | Synthetic label | yes | yes | yes 0.99 | yes 1.00 | yes 0.52 (close) |
| r1_rejected_3 | edited | Banner decision | reject | reject | reject 1.00 | reject 1.00 | reject 0.62 |
| r1_rejected_3 | edited | Exactly 2 rejected | no | yes | no 0.06 | no 0.17 | yes 0.93 ✗ |
| r1_rejected_3 | edited | Legible | yes |  | yes 0.93 | yes 0.99 | yes 0.86 |
| r1_blur | controls | Synthetic label | none | yes | yes 0.80 | no 0.09 | yes 0.51 (close) |
| r1_blur | controls | Banner decision | none | reject | reject 0.66 | needs_review 0.53 (close) | reject 0.57 (close) |
| r1_blur | controls | Exactly 2 rejected | none | yes | yes 0.56 (close) | yes 0.89 | yes 0.93 |
| r1_blur | controls | Legible | no |  | no 0.19 | no 0.01 | yes 0.84 ✗ |
| r2_original | unedited | Synthetic label | no | no | no 0.16 | no 0.03 | no 0.08 |
| r2_original | unedited | Banner decision | reject | reject | reject 1.00 | reject 1.00 | reject 1.00 |
| r2_original | unedited | Exactly 2 rejected | no | no | no 0.04 | no 0.01 | no 0.05 |
| r2_original | unedited | Legible | yes |  | yes 0.98 | yes 1.00 | yes 0.84 |
| r2_banner_accept | edited | Synthetic label | no | no | no 0.32 | no 0.03 | no 0.08 |
| r2_banner_accept | edited | Banner decision | accept | reject | accept 0.64 | accept 0.92 | reject 1.00 ✗ |
| r2_banner_accept | edited | Exactly 2 rejected | no | no | no 0.03 | no 0.01 | no 0.04 |
| r2_banner_accept | edited | Legible | yes |  | yes 0.95 | yes 1.00 | yes 0.84 |
| r2_banner_needs_review | edited | Synthetic label | no | no | no 0.39 | no 0.04 | no 0.07 |
| r2_banner_needs_review | edited | Banner decision | needs_review | reject | needs_review 0.96 | needs_review 0.99 | reject 1.00 ✗ |
| r2_banner_needs_review | edited | Exactly 2 rejected | no | no | no 0.04 | no 0.01 | no 0.05 |
| r2_banner_needs_review | edited | Legible | yes |  | yes 0.96 | yes 1.00 | yes 0.82 |
| r2_rejected_2 | edited | Synthetic label | no | no | no 0.34 | no 0.03 | no 0.08 |
| r2_rejected_2 | edited | Banner decision | reject | reject | reject 1.00 | reject 1.00 | reject 1.00 |
| r2_rejected_2 | edited | Exactly 2 rejected | yes | no | no 0.49 ✗ (close) | yes 0.73 | no 0.05 ✗ |
| r2_rejected_2 | edited | Legible | yes |  | yes 0.96 | yes 1.00 | yes 0.83 |
| r2_blur | controls | Synthetic label | none | no | no 0.25 | no 0.04 | no 0.08 |
| r2_blur | controls | Banner decision | none | reject | reject 0.97 | reject 0.94 | reject 1.00 |
| r2_blur | controls | Exactly 2 rejected | none | no | no 0.09 | no 0.04 | no 0.05 |
| r2_blur | controls | Legible | no |  | no 0.16 | no 0.01 | yes 0.84 ✗ |
| r3_original | unedited | Synthetic label | no | no | no 0.32 | no 0.24 | no 0.09 |
| r3_original | unedited | Banner decision | accept | accept | accept 1.00 | accept 1.00 | accept 1.00 |
| r3_original | unedited | Exactly 2 rejected | no | no | no 0.04 | no 0.01 | no 0.03 |
| r3_original | unedited | Legible | yes |  | yes 0.97 | yes 1.00 | yes 0.84 |
| r3_banner_needs_review | edited | Synthetic label | no | no | no 0.31 | no 0.08 | no 0.11 |
| r3_banner_needs_review | edited | Banner decision | needs_review | accept | needs_review 0.99 | needs_review 0.99 | accept 1.00 ✗ |
| r3_banner_needs_review | edited | Exactly 2 rejected | no | no | no 0.03 | no 0.01 | no 0.03 |
| r3_banner_needs_review | edited | Legible | yes |  | yes 0.96 | yes 1.00 | yes 0.84 |
| r3_banner_reject | edited | Synthetic label | no | no | yes 0.56 ✗ (close) | no 0.14 | no 0.11 |
| r3_banner_reject | edited | Banner decision | reject | accept | reject 1.00 | reject 0.99 | accept 1.00 ✗ |
| r3_banner_reject | edited | Exactly 2 rejected | no | no | no 0.03 | no 0.01 | no 0.03 |
| r3_banner_reject | edited | Legible | yes |  | yes 0.97 | yes 1.00 | yes 0.83 |
| r3_rejected_2 | edited | Synthetic label | no | no | no 0.50 (close) | no 0.18 | no 0.09 |
| r3_rejected_2 | edited | Banner decision | accept | accept | accept 1.00 | accept 1.00 | accept 1.00 |
| r3_rejected_2 | edited | Exactly 2 rejected | yes | no | no 0.36 ✗ | yes 0.86 | no 0.03 ✗ |
| r3_rejected_2 | edited | Legible | yes |  | yes 0.96 | yes 1.00 | yes 0.82 |
| r3_blur | controls | Synthetic label | none | no | no 0.28 | no 0.06 | no 0.11 |
| r3_blur | controls | Banner decision | none | accept | accept 0.98 | accept 0.95 | accept 1.00 |
| r3_blur | controls | Exactly 2 rejected | none | no | no 0.13 | no 0.06 | no 0.03 |
| r3_blur | controls | Legible | no |  | no 0.20 | no 0.02 | yes 0.84 ✗ |
| r1_unmarked | edited | Synthetic label | no | yes | no 0.36 | no 0.33 | yes 0.53 ✗ (close) |
| r1_unmarked | edited | Banner decision | reject | reject | reject 1.00 | reject 1.00 | reject 0.59 (close) |
| r1_unmarked | edited | Exactly 2 rejected | yes | yes | yes 0.94 | yes 0.99 | yes 0.93 |
| r1_unmarked | edited | Legible | yes |  | yes 0.94 | yes 0.99 | yes 0.85 |
| r1_unmarked_banner_accept | edited | Synthetic label | no | yes | no 0.36 | no 0.39 | yes 0.54 ✗ (close) |
| r1_unmarked_banner_accept | edited | Banner decision | accept | reject | accept 0.98 | accept 1.00 | reject 0.59 ✗ (close) |
| r1_unmarked_banner_accept | edited | Exactly 2 rejected | yes | yes | yes 0.93 | yes 0.99 | yes 0.93 |
| r1_unmarked_banner_accept | edited | Legible | yes |  | yes 0.94 | yes 0.99 | yes 0.86 |
| pdf_original | unedited | Synthetic label | yes | yes | yes 0.97 | not sent | yes 0.51 (close) |
| pdf_original | unedited | Banner decision | needs_review | needs_review | needs_review 1.00 | not sent | needs_review 1.00 |
| pdf_original | unedited | Exactly 2 rejected | no | no | no 0.05 | not sent | no 0.05 |
| pdf_original | unedited | Legible | yes |  | yes 0.99 | not sent | yes 0.80 |
| pdf_decision_accept | edited | Synthetic label | yes | yes | yes 0.96 | not sent | yes 0.52 (close) |
| pdf_decision_accept | edited | Banner decision | accept | needs_review | accept 0.99 | not sent | needs_review 1.00 ✗ |
| pdf_decision_accept | edited | Exactly 2 rejected | no | no | no 0.06 | not sent | no 0.04 |
| pdf_decision_accept | edited | Legible | yes |  | yes 0.99 | not sent | yes 0.82 |
| pdf_decision_reject | edited | Synthetic label | yes | yes | yes 0.95 | not sent | yes 0.52 (close) |
| pdf_decision_reject | edited | Banner decision | reject | needs_review | reject 0.99 | not sent | needs_review 1.00 ✗ |
| pdf_decision_reject | edited | Exactly 2 rejected | no | no | no 0.09 | not sent | no 0.05 |
| pdf_decision_reject | edited | Legible | yes |  | yes 0.98 | not sent | yes 0.82 |
| pdf_unmarked | edited | Synthetic label | no | yes | yes 0.81 ✗ | not sent | no 0.49 (close) |
| pdf_unmarked | edited | Banner decision | needs_review | needs_review | needs_review 1.00 | not sent | needs_review 1.00 |
| pdf_unmarked | edited | Exactly 2 rejected | no | no | no 0.05 | not sent | no 0.04 |
| pdf_unmarked | edited | Legible | yes |  | yes 0.99 | not sent | yes 0.84 |
| pdf_trigger_count | edited | Synthetic label | yes | yes | yes 0.97 | not sent | no 0.48 ✗ (close) |
| pdf_trigger_count | edited | Banner decision | needs_review | needs_review | needs_review 1.00 | not sent | needs_review 1.00 |
| pdf_trigger_count | edited | Exactly 2 rejected | no | no | no 0.10 | not sent | no 0.05 |
| pdf_trigger_count | edited | Legible | yes |  | yes 0.99 | not sent | yes 0.81 |
| pdf_4pt | recorded_only | Synthetic label | none | yes | yes 0.96 | not sent | no 0.49 (close) |
| pdf_4pt | recorded_only | Banner decision | none | needs_review | needs_review 1.00 | not sent | needs_review 1.00 |
| pdf_4pt | recorded_only | Exactly 2 rejected | none | no | no 0.06 | not sent | no 0.05 |
| pdf_4pt | recorded_only | Legible | none |  | no 0.43 (close) | not sent | yes 0.83 |
| pdf_page_image | controls | Synthetic label | none | yes | no 0.33 | not sent | yes 0.50 (close) |
| pdf_page_image | controls | Banner decision | none | needs_review | needs_review 0.90 | not sent | needs_review 1.00 |
| pdf_page_image | controls | Exactly 2 rejected | none | no | no 0.13 | not sent | no 0.05 |
| pdf_page_image | controls | Legible | no |  | no 0.14 | not sent | yes 0.82 ✗ |

## The 7 PDF cases side by side

| Case | Question | Label | Gemini, PDF (main run) | Gemini, PNG | Jev image preview, PNG | Jev text only (main run) |
| :- | :- | :- | :- | :- | :- | :- |
| pdf_original | Synthetic label | yes | yes 0.97 | yes 0.95 | yes 1.00 | yes 0.51 (close) |
| pdf_original | Banner decision | needs_review | needs_review 1.00 | needs_review 1.00 | needs_review 1.00 | needs_review 1.00 |
| pdf_original | Exactly 2 rejected | no | no 0.05 | no 0.05 | no 0.01 | no 0.05 |
| pdf_original | Legible | yes | yes 0.99 | yes 0.99 | yes 1.00 | yes 0.80 |
| pdf_decision_accept | Synthetic label | yes | yes 0.96 | yes 0.95 | yes 1.00 | yes 0.52 (close) |
| pdf_decision_accept | Banner decision | accept | accept 0.99 | accept 0.99 | accept 0.97 | needs_review 1.00 ✗ |
| pdf_decision_accept | Exactly 2 rejected | no | no 0.06 | no 0.06 | no 0.01 | no 0.04 |
| pdf_decision_accept | Legible | yes | yes 0.99 | yes 0.99 | yes 1.00 | yes 0.82 |
| pdf_decision_reject | Synthetic label | yes | yes 0.95 | yes 0.96 | yes 1.00 | yes 0.52 (close) |
| pdf_decision_reject | Banner decision | reject | reject 0.99 | reject 0.99 | reject 0.98 | needs_review 1.00 ✗ |
| pdf_decision_reject | Exactly 2 rejected | no | no 0.09 | no 0.10 | no 0.02 | no 0.05 |
| pdf_decision_reject | Legible | yes | yes 0.98 | yes 0.99 | yes 1.00 | yes 0.82 |
| pdf_unmarked | Synthetic label | no | yes 0.81 ✗ | yes 0.67 ✗ | no 0.15 | no 0.49 (close) |
| pdf_unmarked | Banner decision | needs_review | needs_review 1.00 | needs_review 1.00 | needs_review 1.00 | needs_review 1.00 |
| pdf_unmarked | Exactly 2 rejected | no | no 0.05 | no 0.05 | no 0.01 | no 0.04 |
| pdf_unmarked | Legible | yes | yes 0.99 | yes 0.99 | yes 1.00 | yes 0.84 |
| pdf_trigger_count | Synthetic label | yes | yes 0.97 | yes 0.96 | yes 1.00 | no 0.48 ✗ (close) |
| pdf_trigger_count | Banner decision | needs_review | needs_review 1.00 | needs_review 1.00 | needs_review 1.00 | needs_review 1.00 |
| pdf_trigger_count | Exactly 2 rejected | no | no 0.10 | no 0.07 | no 0.01 | no 0.05 |
| pdf_trigger_count | Legible | yes | yes 0.99 | yes 0.99 | yes 1.00 | yes 0.81 |
| pdf_4pt | Synthetic label | none | yes 0.96 | yes 0.96 | yes 0.99 | no 0.49 (close) |
| pdf_4pt | Banner decision | none | needs_review 1.00 | needs_review 1.00 | needs_review 1.00 | needs_review 1.00 |
| pdf_4pt | Exactly 2 rejected | none | no 0.06 | no 0.05 | no 0.02 | no 0.05 |
| pdf_4pt | Legible | none | no 0.43 (close) | no 0.53 (close) | yes 0.88 | yes 0.83 |
| pdf_page_image | Synthetic label | none | no 0.33 | no 0.41 (close) | no 0.08 | yes 0.50 (close) |
| pdf_page_image | Banner decision | none | needs_review 0.90 | needs_review 0.91 | needs_review 0.95 | needs_review 1.00 |
| pdf_page_image | Exactly 2 rejected | none | no 0.13 | no 0.20 | no 0.08 | no 0.05 |
| pdf_page_image | Legible | no | no 0.14 | no 0.19 | no 0.02 | yes 0.82 ✗ |

## Latency, tokens and cost

| Arm | Calls | Median ms | Max ms | Input tokens | USD at list price |
| :- | -: | -: | -: | -: | -: |
| Gemini Decision 1 (image or PDF + records) | 24 | 890.0 | 1394.5 | 111,403 | 0.01114 |
| Jev 1.14, image preview (image + records) | 17 | 185.0 | 344.2 | 162,795 | 0.00684 |
| Jev 1.14, text only (records only, cannot see the artifact) | 24 | 95.9 | 215.3 | 91,838 | 0.00386 |

Jev image tokens per 1200 x 1568 image, measured as the image arm minus the text only arm: 4834 on all 17 cases. Gemini ran on the free tier, so its calls cost nothing; the table uses the paid list price.

