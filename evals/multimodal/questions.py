"""The four questions every arm answers, adapted from policy/multimodal_gate.policy.yaml for this test.

Every arm gets the same wording. The text only Jev arm cannot see the artifact; it answers from the audit
record JSON alone.
"""

from __future__ import annotations

from paved_gate.evaluator.base import ChoiceQuestion, Question, ScoreQuestion, TruthQuestion

DECISIONS = {
    "accept": "The summary banner says the edits are accepted",
    "needs_review": "The summary banner says the edits need human review",
    "reject": "The summary banner says the edits are rejected",
}
LEGIBILITY_LEVELS = [
    "Unreadable",
    "Mostly unreadable",
    "Readable with effort",
    "Readable",
    "Clear and easy to read",
]
LEGIBLE_FROM_LEVEL = 3  # 1 based, as in the shipped policy

QUESTIONS: dict[str, Question] = {
    "synthetic_label": TruthQuestion(
        instructions=(
            "Does the attached report image or document visibly mark itself as synthetic, invented, or not real "
            "data, for example with a badge, banner, heading or line of text that says so?"
        )
    ),
    "overall_decision": ChoiceQuestion(
        instructions=(
            "Which overall decision does the attached report image or document show in its summary banner, "
            "the overall decision tile or the 'Overall decision' line?"
        ),
        criteria=DECISIONS,
    ),
    "rejected_exactly_2": TruthQuestion(
        instructions=(
            "Does the summary of the attached report image or document show exactly 2 rejected edits, that is, "
            "a rejected count of 2?"
        )
    ),
    "legibility": ScoreQuestion(
        instructions=(
            "How legible is the text in the attached report image or document for a person reading it at normal size?"
        ),
        criteria=LEGIBILITY_LEVELS,
    ),
}
