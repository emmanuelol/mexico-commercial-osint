"""
Abstract Outreach & Promo Protocols for Commercial Activation.
Defines clean-room contracts separating public OSINT scoring from downstream RPA execution.
"""
from typing import Optional, Protocol, runtime_checkable
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CouponSpec(BaseModel):
    """Specification for promotional discount coupons."""
    model_config = ConfigDict(extra="ignore")

    code: str = Field(description="Coupon alphanumeric code (4-12 characters)")
    coupon_type: str = Field(default="discount", description="Coupon type (discount, fixed)")
    description: str = Field(default="local_partners", description="Campaign identifier or category")
    discount_pct: float = Field(default=15.0, ge=0.0, le=50.0, description="Percentage discount")
    store_id: int = Field(default=525, description="Store branch ID")
    metadata: dict = Field(default_factory=dict, description="Custom campaign metadata")

    @field_validator("code")
    @classmethod
    def validate_code(cls, v: str) -> str:
        clean = v.strip().upper()
        if not (4 <= len(clean) <= 12) or not clean.isalnum():
            raise ValueError(f"Coupon code must be 4-12 alphanumeric characters, got: {clean}")
        return clean


class OutreachPayload(BaseModel):
    """Payload for scheduled commercial B2B outreach."""
    model_config = ConfigDict(extra="ignore")

    canonical_entity_id: str
    recipient_email: str
    email_subject: str
    email_body: str
    chronotype_key: str
    coupon_code: Optional[str] = None
    channel: str = Field(default="Direct Email")
    contact_risk: str = Field(default="low")
    email_confidence: str = Field(default="high")

    @field_validator("recipient_email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError(f"Invalid recipient email: {v}")
        return v.strip().lower()


@runtime_checkable
class PromoProviderProtocol(Protocol):
    """Protocol for downstream coupon provisioning services."""
    def provision_coupon(self, spec: CouponSpec) -> dict:
        ...

    def verify_coupon(self, code: str) -> bool:
        ...


@runtime_checkable
class OutreachDispatcherProtocol(Protocol):
    """Protocol for downstream email/messaging dispatchers."""
    def dispatch(self, payload: OutreachPayload) -> dict:
        ...


class DryRunPromoProvider:
    """Safe in-memory promo provider for testing and dry-run pipelines."""
    def __init__(self, max_discount_pct: float = 15.0):
        self.max_discount_pct = max_discount_pct
        self._registry: dict[str, CouponSpec] = {}

    def provision_coupon(self, spec: CouponSpec) -> dict:
        if spec.discount_pct > self.max_discount_pct:
            raise ValueError(f"Discount {spec.discount_pct}% exceeds max allowed {self.max_discount_pct}%")
        self._registry[spec.code] = spec
        return {
            "status": "PROVISIONED_DRY_RUN",
            "code": spec.code,
            "discount_pct": spec.discount_pct,
            "description": spec.description,
            "store_id": spec.store_id,
        }

    def verify_coupon(self, code: str) -> bool:
        return code.strip().upper() in self._registry


class DryRunOutreachDispatcher:
    """Safe in-memory outreach dispatcher enforcing clean-room delivery guardrails."""
    def __init__(self):
        self.dispatched: list[OutreachPayload] = []

    def dispatch(self, payload: OutreachPayload) -> dict:
        # Enforce automated sequence risk guardrail
        if payload.contact_risk != "low" or payload.email_confidence not in ("high", "medium-high"):
            raise ValueError(
                f"Outreach blocked by guardrail: risk={payload.contact_risk}, confidence={payload.email_confidence}"
            )
        self.dispatched.append(payload)
        return {
            "status": "DISPATCHED_DRY_RUN",
            "recipient": payload.recipient_email,
            "subject": payload.email_subject,
            "chronotype": payload.chronotype_key,
        }
