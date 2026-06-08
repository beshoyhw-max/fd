"""
Round Number Bias Check — Fraud Detection Module #5.

Flags invoices with an unusually high proportion of round numbers.
Fabricated invoices tend to use round amounts (.00, .000, .500)
more frequently than real transactions.
"""

import logging
from typing import List, Optional

from src.models import ExtractedFields, FraudCheck, Severity

logger = logging.getLogger(__name__)

CHECK_NAME = "round_number_bias"

# What counts as a "round" number
ROUND_PATTERNS = {
    "exact_hundreds": lambda x: x >= 100 and x % 100 == 0,
    "exact_thousands": lambda x: x >= 1000 and x % 1000 == 0,
    "half_values": lambda x: x > 0 and (x * 10) % 10 == 5 and (x * 100) % 100 == 50,
    "zero_cents": lambda x: x > 0 and x == int(x),
}

# Expected round number ratio in normal business invoices (approx 20-35%)
EXPECTED_ROUND_RATIO = 0.30
SUSPICIOUS_THRESHOLD = 0.65  # More than 65% round = suspicious
HIGH_THRESHOLD = 0.85        # More than 85% round = very suspicious


def _collect_amounts(fields: ExtractedFields) -> List[float]:
    """Collect all monetary amounts from the invoice."""
    amounts = []
    for item in fields.line_items or []:
        if item.unit_price is not None and item.unit_price > 0:
            amounts.append(item.unit_price)
        if item.line_total is not None and item.line_total > 0:
            amounts.append(item.line_total)
    if fields.grand_total is not None and fields.grand_total > 0:
        amounts.append(fields.grand_total)
    if fields.tax_amount is not None and fields.tax_amount > 0:
        amounts.append(fields.tax_amount)
    return amounts


def _is_round(value: float) -> bool:
    """Check if a value matches any round number pattern."""
    return any(check(value) for check in ROUND_PATTERNS.values())


def run(fields: ExtractedFields, **kwargs) -> FraudCheck:
    """
    Detect round number bias in invoice amounts.

    Returns a FraudCheck scored by the proportion of round numbers.
    """
    amounts = _collect_amounts(fields)

    if len(amounts) < 2:
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail="Insufficient amounts to analyze round number bias.",
            evidence={"amount_count": len(amounts)},
        )

    round_count = sum(1 for a in amounts if _is_round(a))
    total_count = len(amounts)
    round_ratio = round_count / total_count

    # Categorize each amount
    amount_details = []
    for a in amounts:
        patterns_matched = [
            name for name, check in ROUND_PATTERNS.items() if check(a)
        ]
        amount_details.append({
            "amount": a,
            "is_round": bool(patterns_matched),
            "patterns": patterns_matched,
        })

    evidence = {
        "total_amounts": total_count,
        "round_count": round_count,
        "round_ratio": round(round_ratio, 3),
        "expected_ratio": EXPECTED_ROUND_RATIO,
        "amounts": amount_details,
    }

    if round_ratio >= HIGH_THRESHOLD:
        score = min(100, int(70 + (round_ratio - HIGH_THRESHOLD) * 200))
        return FraudCheck(
            name=CHECK_NAME,
            score=score,
            severity=Severity.MEDIUM,
            triggered=True,
            detail=(
                f"Very high round number bias: {round_count}/{total_count} amounts "
                f"({round_ratio*100:.0f}%) are round numbers "
                f"(expected ~{EXPECTED_ROUND_RATIO*100:.0f}%)."
            ),
            evidence=evidence,
        )
    elif round_ratio >= SUSPICIOUS_THRESHOLD:
        score = int(30 + (round_ratio - SUSPICIOUS_THRESHOLD) * 200)
        return FraudCheck(
            name=CHECK_NAME,
            score=score,
            severity=Severity.MEDIUM,
            triggered=True,
            detail=(
                f"Elevated round number ratio: {round_count}/{total_count} amounts "
                f"({round_ratio*100:.0f}%) are round numbers "
                f"(expected ~{EXPECTED_ROUND_RATIO*100:.0f}%)."
            ),
            evidence=evidence,
        )
    else:
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail=(
                f"Round number ratio is normal: {round_count}/{total_count} "
                f"({round_ratio*100:.0f}%) — within expected range."
            ),
            evidence=evidence,
        )
