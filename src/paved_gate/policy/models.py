from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from paved_gate.types import Route

DetectorName = Literal["email", "phone", "ssn", "credit_card", "mrn", "dob"]
ALL_DETECTORS: tuple[DetectorName, ...] = ("email", "phone", "ssn", "credit_card", "mrn", "dob")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


Provider = Literal["jev", "heuristic", "gemini_decision"]

# Defaults for providers other than Jev, applied only to fields a policy leaves out, so every existing
# Jev or heuristic policy validates and hashes exactly as before.
_PROVIDER_DEFAULTS: dict[str, dict[str, object]] = {
    "gemini_decision": {
        "model": "gemini-decision-1",
        "api_key_env": "GEMINI_API_KEY",
        "timeout_ms": 3000,
        "max_retries": 2,
    },
}
_PROVIDER_FIELDS = ("model", "base_url", "api_key_env", "timeout_ms", "max_retries")


class EvaluatorConfig(_Strict):
    provider: Provider = "jev"
    mode: Literal["mock", "live"] = "mock"
    model: str = "jev-latest"
    base_url: str = "https://api.typesafe.ai/v1"
    api_key_env: str = "TYPESAFE_API_KEY"
    timeout_ms: int = Field(default=400, gt=0)
    max_retries: int = Field(default=1, ge=0)
    fail_mode: Literal["closed", "open"] = "closed"

    @model_validator(mode="before")
    @classmethod
    def _provider_defaults(cls, data: object) -> object:
        if isinstance(data, dict) and isinstance(data.get("provider"), str):
            return {**_PROVIDER_DEFAULTS.get(data["provider"], {}), **data}
        return data

    def with_provider(self, provider: Provider) -> EvaluatorConfig:
        """Switch provider, dropping provider specific settings so the new provider's defaults apply."""
        if provider == self.provider:
            return self
        kept = self.model_dump(exclude=set(_PROVIDER_FIELDS))
        return EvaluatorConfig.model_validate({**kept, "provider": provider})


class IntentQuestionConfig(_Strict):
    instructions: str
    criteria: dict[str, str]
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _routes_match(self) -> Self:
        expected = {r.value for r in Route}
        if set(self.criteria) != expected:
            raise ValueError(f"intent.criteria keys must be exactly {sorted(expected)}")
        return self


class RiskQuestionConfig(_Strict):
    instructions: str
    criteria: list[str] = Field(min_length=2)
    scale_min: int = 1
    block_above: float

    @property
    def scale_max(self) -> int:
        return self.scale_min + len(self.criteria) - 1

    @model_validator(mode="after")
    def _threshold_in_range(self) -> Self:
        if not self.scale_min <= self.block_above <= self.scale_max:
            raise ValueError(f"risk.block_above must be within [{self.scale_min}, {self.scale_max}]")
        return self


class SensitiveQuestionConfig(_Strict):
    instructions: str
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class QuestionsConfig(_Strict):
    intent: IntentQuestionConfig
    risk: RiskQuestionConfig
    sensitive: SensitiveQuestionConfig


class PrivacyConfig(_Strict):
    mask_before_evaluate: bool = True
    detectors: list[DetectorName] = Field(default_factory=lambda: list(ALL_DETECTORS))
    local_detection_counts_as_sensitive: bool = True


class AuditConfig(_Strict):
    sink: Literal["jsonl", "stdout"] = "jsonl"
    path: str = "./logs/paved_gate.audit.jsonl"
    log_masked_payload: bool = False


class DownstreamConfig(_Strict):
    provider: Literal["anthropic", "openai", "none"] = "none"
    model: str | None = None
    max_tokens: int = Field(default=16000, gt=0)
    system_prompt: str | None = None


class Policy(_Strict):
    version: Literal[1]
    name: str
    evaluator: EvaluatorConfig = EvaluatorConfig()
    questions: QuestionsConfig
    privacy: PrivacyConfig = PrivacyConfig()
    audit: AuditConfig = AuditConfig()
    downstream: DownstreamConfig = DownstreamConfig()
