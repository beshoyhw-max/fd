"""
Fraud Detection Engine — Orchestrator with Veto-based Risk Gate.

Runs all 13 fraud detection checks, applies the Veto-based Risk Gate
(一票否决门控机制), and computes the Suspicion Rank Index for
review queue prioritization.
"""

import logging
from typing import Any, Dict, List, Optional

from PIL import Image

from src.config import Config
from src.models import (
    DocumentType,
    ExtractedFields,
    FraudCheck,
    FraudResult,
    Invoice,
    RecommendedAction,
    RiskLevel,
    Severity,
)

# Import all 13 fraud check modules
from src.fraud import (
    math_check,
    missing_fields,
    date_check,
    benford_check,
    round_number,
    duplicate_check,
    threshold_split,
    vendor_pattern,
    pdf_metadata,
    ela_check,
    ai_generated,
    font_consistency,
    noise_analysis,
)

logger = logging.getLogger(__name__)

# Checks that require image data (skipped for text PDFs)
IMAGE_FORENSIC_CHECKS = {
    "ela_analysis",
    "ai_generated",
    "font_consistency",
    "noise_analysis",
}


class FraudEngine:
    """
    Orchestrates all 13 fraud detection checks and applies
    the Veto-based Risk Gate.

    Architecture:
    - Auto-approval ONLY if ALL checks pass (zero alerts)
    - ANY triggered check → auto-approval blocked
    - Suspicion Rank Index (0-100) used to prioritize review queue
    """

    def __init__(self, config: Optional[Config] = None):
        self._config = config or Config.get()

    def run_all_checks(
        self,
        invoice: Invoice,
        historical_invoices: Optional[List[Dict[str, Any]]] = None,
        page_image: Optional[Image.Image] = None,
        ocr_data: Optional[List[Dict]] = None,
    ) -> FraudResult:
        """
        Execute all applicable fraud checks on an invoice.

        Args:
            invoice: The invoice with extracted fields.
            historical_invoices: Historical invoice data for cross-checks.
            page_image: First page image (for forensic checks).
            ocr_data: Per-character OCR data (for font consistency).

        Returns:
            FraudResult with all check results, Veto gate decision, and
            Suspicion Rank Index.
        """
        fields = invoice.extracted_fields
        is_text_pdf = invoice.document_type == DocumentType.TEXT_PDF
        checks: List[FraudCheck] = []

        # ── Run all checks ──────────────────────────────────

        # 1. Math Verification
        checks.append(self._run_check(
            "math_verification",
            lambda: math_check.run(fields),
        ))

        # 2. Missing Fields
        checks.append(self._run_check(
            "missing_fields",
            lambda: missing_fields.run(fields),
        ))

        # 3. Date Anomalies
        checks.append(self._run_check(
            "date_anomalies",
            lambda: date_check.run(fields, config=self._config),
        ))

        # 4. Benford's Law
        hist_amounts = self._collect_historical_amounts(historical_invoices)
        checks.append(self._run_check(
            "benford_law",
            lambda: benford_check.run(
                fields,
                historical_amounts=hist_amounts,
                config=self._config,
            ),
        ))

        # 5. Round Number Bias
        checks.append(self._run_check(
            "round_number_bias",
            lambda: round_number.run(fields),
        ))

        # 6. Duplicate Detection
        checks.append(self._run_check(
            "duplicate_detection",
            lambda: duplicate_check.run(
                fields,
                historical_invoices=historical_invoices,
                config=self._config,
            ),
        ))

        # 7. Threshold Splitting
        checks.append(self._run_check(
            "threshold_splitting",
            lambda: threshold_split.run(
                fields,
                historical_invoices=historical_invoices,
                config=self._config,
            ),
        ))

        # 8. Vendor Pattern Analysis
        checks.append(self._run_check(
            "vendor_pattern",
            lambda: vendor_pattern.run(
                fields,
                historical_invoices=historical_invoices,
                config=self._config,
            ),
        ))

        # 9. PDF Metadata Forensics (applies to both text and scanned PDFs)
        pdf_meta = None
        if invoice.pdf_path:
            from src.ingestion.pdf_extractor import PDFExtractor
            try:
                pdf_meta = PDFExtractor().extract_metadata_only(invoice.pdf_path)
            except Exception:
                pass

        checks.append(self._run_check(
            "pdf_metadata",
            lambda: pdf_metadata.run(pdf_metadata=pdf_meta),
        ))

        # ── Image Forensics (skipped for text PDFs) ─────────

        if is_text_pdf:
            # Skip image forensics — analyzing self-rendered images would
            # produce false clean results
            for check_name in ["ela_analysis", "ai_generated", "font_consistency", "noise_analysis"]:
                checks.append(FraudCheck(
                    name=check_name,
                    score=0,
                    severity=Severity.LOW,
                    triggered=False,
                    detail="Skipped — text PDF (no original image to analyze).",
                ))
        else:
            # 10. Error Level Analysis
            checks.append(self._run_check(
                "ela_analysis",
                lambda: ela_check.run(
                    image=page_image,
                    invoice_id=invoice.invoice_id,
                    config=self._config,
                ),
            ))

            # 11. AI-Generated Detection
            checks.append(self._run_check(
                "ai_generated",
                lambda: ai_generated.run(image=page_image),
            ))

            # 12. Font Consistency
            checks.append(self._run_check(
                "font_consistency",
                lambda: font_consistency.run(
                    image=page_image,
                    ocr_data=ocr_data,
                ),
            ))

            # 13. Noise Pattern Analysis
            checks.append(self._run_check(
                "noise_analysis",
                lambda: noise_analysis.run(image=page_image),
            ))

        # ── Apply Veto-based Risk Gate ──────────────────────

        return self._apply_veto_gate(checks)

    def _run_check(self, name: str, check_fn) -> FraudCheck:
        """Run a single check with error handling."""
        try:
            result = check_fn()
            logger.debug(f"Check '{name}': score={result.score}, triggered={result.triggered}")
            return result
        except Exception as e:
            logger.error(f"Check '{name}' failed with error: {e}")
            return FraudCheck(
                name=name,
                score=0,
                severity=Severity.LOW,
                triggered=False,
                detail=f"Check failed: {e}",
            )

    def _apply_veto_gate(self, checks: List[FraudCheck]) -> FraudResult:
        """
        Apply the Veto-based Risk Gate (一票否决门控机制).

        Rules:
        1. Auto-approval ONLY if ALL checks pass (zero triggered).
        2. ANY triggered check → auto-approval blocked.
        3. Suspicion Rank Index computed as weighted score for queue priority.
        """
        triggered_checks = [c for c in checks if c.triggered]
        any_triggered = len(triggered_checks) > 0
        triggered_names = [c.name for c in triggered_checks]

        # Compute Suspicion Rank Index (weighted aggregate)
        aggregate_score = self._compute_suspicion_rank(checks)

        # Determine risk level
        risk_level = self._score_to_risk_level(aggregate_score)

        # Determine recommended action based on Veto gate
        if not any_triggered:
            # All clean → auto-approve
            recommended_action = RecommendedAction.APPROVE
        else:
            # Veto triggered → block or review
            has_critical = any(
                c.severity in (Severity.CRITICAL, Severity.HIGH) and c.triggered
                for c in checks
            )
            if has_critical or aggregate_score >= self._config.risk_threshold_medium:
                recommended_action = RecommendedAction.BLOCK
            else:
                recommended_action = RecommendedAction.REVIEW

        return FraudResult(
            checks=checks,
            aggregate_score=aggregate_score,
            risk_level=risk_level,
            recommended_action=recommended_action,
            any_triggered=any_triggered,
            triggered_checks=triggered_names,
            veto_applied=any_triggered,
        )

    def _compute_suspicion_rank(self, checks: List[FraudCheck]) -> int:
        """
        Compute the Suspicion Rank Index (0–100).

        Weighted average of all check scores, normalized to 0–100.
        This is used solely for review queue prioritization.
        """
        weights = self._config.fraud_weights
        total_weight = 0
        weighted_sum = 0

        for check in checks:
            weight = weights.get(check.name, 5)  # Default weight 5
            weighted_sum += check.score * weight
            total_weight += weight

        if total_weight == 0:
            return 0

        # Normalize: max possible is 100 * total_weight
        raw = weighted_sum / total_weight
        return min(100, int(raw))

    def _score_to_risk_level(self, score: int) -> RiskLevel:
        """Convert aggregate score to a risk level tier."""
        if score >= self._config.risk_threshold_medium:
            return RiskLevel.HIGH
        elif score >= self._config.risk_threshold_low:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    @staticmethod
    def _collect_historical_amounts(
        historical: Optional[List[Dict[str, Any]]],
    ) -> List[float]:
        """Extract monetary amounts from historical invoices for Benford's analysis."""
        if not historical:
            return []
        amounts = []
        for h in historical:
            total = h.get("grand_total")
            if total is not None and total > 0:
                amounts.append(float(total))
        return amounts
