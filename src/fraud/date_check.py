"""
Date Anomaly Check — Fraud Detection Module #3.

Detects suspicious date patterns:
- Future dates
- Invoice date after due date
- Weekend/holiday dates
- Invoice date far in the past
- Date sequence violations
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from dateutil import parser as dateutil_parser

from src.config import Config
from src.models import ExtractedFields, FraudCheck, Severity

logger = logging.getLogger(__name__)

CHECK_NAME = "date_anomalies"


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Safely parse a date string into a datetime object."""
    if not date_str or not date_str.strip():
        return None
    try:
        return dateutil_parser.parse(date_str, dayfirst=False)
    except (ValueError, TypeError):
        return None


def run(fields: ExtractedFields, config: Optional[Config] = None, **kwargs) -> FraudCheck:
    """
    Detect date anomalies in the invoice.

    Checks:
    1. Future invoice dates
    2. Invoice date after due date
    3. Weekend dates (configurable)
    4. Dates far in the past
    5. Due date in the past
    """
    config = config or Config.get()
    issues = []
    evidence = {"checks": []}

    now = datetime.utcnow()
    invoice_date = _parse_date(fields.invoice_date)
    due_date = _parse_date(fields.due_date)

    # ── 1. Future invoice date ──
    if invoice_date and invoice_date > now + timedelta(days=1):
        days_ahead = (invoice_date - now).days
        issues.append(f"Invoice date is {days_ahead} days in the future: {fields.invoice_date}")
        evidence["checks"].append({
            "check": "future_invoice_date",
            "days_ahead": days_ahead,
        })

    # ── 2. Invoice date after due date ──
    if invoice_date and due_date and invoice_date > due_date:
        issues.append(
            f"Invoice date ({fields.invoice_date}) is after due date ({fields.due_date})"
        )
        evidence["checks"].append({
            "check": "invoice_after_due",
            "invoice_date": fields.invoice_date,
            "due_date": fields.due_date,
        })

    # ── 3. Weekend dates ──
    if config.date_weekend_flag:
        if invoice_date and invoice_date.weekday() >= 5:
            day_name = invoice_date.strftime("%A")
            issues.append(f"Invoice dated on a {day_name}: {fields.invoice_date}")
            evidence["checks"].append({
                "check": "weekend_date",
                "day": day_name,
            })

    # ── 4. Invoice date far in the past ──
    max_past = config.date_max_past_days
    if invoice_date:
        days_ago = (now - invoice_date).days
        if days_ago > max_past:
            issues.append(
                f"Invoice date is {days_ago} days in the past (threshold: {max_past}): "
                f"{fields.invoice_date}"
            )
            evidence["checks"].append({
                "check": "old_invoice",
                "days_ago": days_ago,
                "threshold": max_past,
            })

    # ── 5. Due date already passed ──
    if due_date and due_date < now - timedelta(days=30):
        days_overdue = (now - due_date).days
        issues.append(
            f"Due date is {days_overdue} days in the past: {fields.due_date}"
        )
        evidence["checks"].append({
            "check": "overdue",
            "days_overdue": days_overdue,
        })

    # ── 6. Unreasonable payment terms ──
    if invoice_date and due_date:
        payment_days = (due_date - invoice_date).days
        if payment_days > 180:
            issues.append(f"Payment terms unusually long: {payment_days} days")
            evidence["checks"].append({
                "check": "long_payment_terms",
                "days": payment_days,
            })
        elif payment_days < 0:
            # Already caught by check #2, but add evidence
            pass

    # ── Scoring ──
    if not issues:
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail="All date checks passed — no anomalies detected.",
        )

    # Score based on number and severity of issues
    score = min(100, len(issues) * 20)

    # Future dates are most suspicious
    has_future = any(c.get("check") == "future_invoice_date" for c in evidence["checks"])
    has_sequence = any(c.get("check") == "invoice_after_due" for c in evidence["checks"])

    if has_future or has_sequence:
        severity = Severity.HIGH
        score = max(score, 60)
    else:
        severity = Severity.MEDIUM

    return FraudCheck(
        name=CHECK_NAME,
        score=score,
        severity=severity,
        triggered=True,
        detail="; ".join(issues),
        evidence=evidence,
    )
