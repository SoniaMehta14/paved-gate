"""Intent labels and one-line descriptions for the Bitext customer-support dataset.

Every arm (Jev, GPT-4o, GPT-4o mini) is given exactly this list, so accuracy
differences come from the models, not from different instructions. Descriptions
are written to separate look-alike pairs (check vs get invoice, customer service
vs human agent, change vs set up shipping address, refund policy vs get vs track).
"""

from __future__ import annotations

INTENTS: dict[str, str] = {
    "cancel_order": "Wants to cancel an order that has already been placed",
    "change_order": "Wants to modify an existing order (items, quantities, options)",
    "change_shipping_address": "Wants to change a shipping address that is already on file or on an order",
    "check_cancellation_fee": "Asks whether there is a fee or penalty for cancelling",
    "check_invoice": "Wants to look at or check the details of an existing invoice",
    "check_payment_methods": "Asks which payment methods or options are accepted",
    "check_refund_policy": "Asks about the refund or money-back policy and its conditions",
    "complaint": "Wants to file a complaint or express dissatisfaction",
    "contact_customer_service": "Wants the customer service team's contact details or opening hours",
    "contact_human_agent": "Wants to speak to a live human agent instead of an automated assistant",
    "create_account": "Wants to open or sign up for a new account",
    "delete_account": "Wants to close or delete their account",
    "delivery_options": "Asks which delivery or shipping methods are available",
    "delivery_period": "Asks how long delivery takes or when an order will arrive",
    "edit_account": "Wants to update personal or profile details on their account",
    "get_invoice": "Wants to receive, download or be sent an invoice",
    "get_refund": "Wants to request or obtain a refund",
    "newsletter_subscription": "Wants to subscribe to or unsubscribe from the newsletter",
    "payment_issue": "Reports a problem with a payment that failed or went wrong",
    "place_order": "Wants to place or make a new order or purchase",
    "recover_password": "Forgot their password or wants to reset it",
    "registration_problems": "Reports a problem or error while signing up",
    "review": "Wants to leave a review, rating or feedback about a product or service",
    "set_up_shipping_address": "Wants to add or set up a new shipping address",
    "switch_account": "Wants to switch to a different account or change account type",
    "track_order": "Wants to know the status or location of an order",
    "track_refund": "Wants to know the status of a refund already requested",
}

INVALID = "(invalid)"


def labels() -> list[str]:
    return sorted(INTENTS)


def require_descriptions(dataset_intents: set[str]) -> None:
    """Fail loudly if the dataset contains an intent this list doesn't describe (or vice versa)."""
    missing = sorted(dataset_intents - INTENTS.keys())
    extra = sorted(INTENTS.keys() - dataset_intents)
    if missing or extra:
        raise ValueError(f"intent list mismatch: missing descriptions for {missing}; not in dataset: {extra}")
