"""Multimodal gate: typed checks over an image or PDF plus text, in one evaluator call.

A published report screenshot or a policy PDF goes to the evaluator with the text it should agree with,
and Truth, Choice and Score checks decide allow, review or block, with a JSON audit record that stores
hashes and answers but never the file. Needs an evaluator that reads media: Gemini Decision 1.

    python -m paved_gate.integrations.multimodal_gate docs/harness2/1_synthetic_fixture.png \\
        --expect overall_decision=reject
"""

from paved_gate.integrations.multimodal_gate.config import MultimodalPolicy, load_multimodal_policy
from paved_gate.integrations.multimodal_gate.gate import MultimodalGate, MultimodalGateError, build_gate

__all__ = ["MultimodalGate", "MultimodalGateError", "MultimodalPolicy", "build_gate", "load_multimodal_policy"]
