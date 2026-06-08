"""
Pydantic Data Models for Invoice Fraud Detection System.

Defines the core data structures used throughout the pipeline:
Invoice, LineItem, FraudCheck, FraudResult, ProcessingResult, ReviewItem.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────


class DocumentType(str, Enum):
    """How the document was classified at ingestion."""
    TEXT_PDF = "text_pdf"
    SCANNED_PDF = "scanned_pdf"
    IMAGE = "image"


class RiskLevel(str, Enum):
    """Suspicion Rank Index tier."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RecommendedAction(str, Enum):
    """System's recommended disposition."""
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class ReviewStatus(str, Enum):
    """Human review decision state."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    AUTO_APPROVED = "AUTO_APPROVED"


class Severity(str, Enum):
    """Fraud check severity classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ── Invoice Data Models ────────────────────────────────────────


class LineItem(BaseModel):
    """A single line item on an invoice."""
    description: str = ""
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    line_total: Optional[float] = None


class BankDetails(BaseModel):
    """Bank payment details on an invoice."""
    iban: Optional[str] = None
    swift: Optional[str] = None
    account_number: Optional[str] = None
    bank_name: Optional[str] = None


class ExtractedFields(BaseModel):
    """All 10 extracted fields from an invoice."""
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    currency: Optional[str] = None
    line_items: List[LineItem] = Field(default_factory=list)
    tax_rate: Optional[float] = None
    tax_amount: Optional[float] = None
    grand_total: Optional[float] = None
    po_number: Optional[str] = None
    customer_name: Optional[str] = None
    bank_details: Optional[BankDetails] = None


class Invoice(BaseModel):
    """
    Complete invoice record including source metadata
    and extracted fields.
    """
    invoice_id: str = ""
    source_file: str = ""
    document_type: DocumentType = DocumentType.IMAGE
    page_count: int = 1
    extracted_fields: ExtractedFields = Field(default_factory=ExtractedFields)

    # Raw text extracted (for text-based PDFs)
    raw_text: Optional[str] = None

    # Image paths (for scanned PDFs / images — used by forensic checks)
    image_paths: List[str] = Field(default_factory=list)

    # PDF file path (for metadata forensics)
    pdf_path: Optional[str] = None


# ── Fraud Detection Models ─────────────────────────────────────


class FraudCheck(BaseModel):
    """Result of a single fraud detection check."""
    name: str
    score: int = 0                        # 0 = clean, 100 = maximum suspicion
    severity: Severity = Severity.LOW
    triggered: bool = False               # Whether this check tripped an alert
    detail: str = ""                      # Human-readable explanation
    evidence: Optional[Dict[str, Any]] = None  # Machine-readable evidence data


class FraudResult(BaseModel):
    """
    Aggregate fraud detection result.

    Uses a Veto-based Risk Gate:
    - Auto-approval ONLY if zero checks triggered.
    - Any triggered check → blocked for manual review.
    - aggregate_score is a Suspicion Rank Index (0–100) for queue prioritization.
    """
    checks: List[FraudCheck] = Field(default_factory=list)
    aggregate_score: int = 0
    risk_level: RiskLevel = RiskLevel.LOW
    recommended_action: RecommendedAction = RecommendedAction.APPROVE
    any_triggered: bool = False
    triggered_checks: List[str] = Field(default_factory=list)
    veto_applied: bool = False


# ── Processing Result ──────────────────────────────────────────


class ProcessingResult(BaseModel):
    """
    Complete result for a single invoice processing run.
    This is the JSON record persisted to data/results/.
    """
    invoice_id: str = ""
    source_file: str = ""
    processed_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    document_type: DocumentType = DocumentType.IMAGE
    extracted_fields: ExtractedFields = Field(default_factory=ExtractedFields)
    fraud_result: FraudResult = Field(default_factory=FraudResult)
    review_status: ReviewStatus = ReviewStatus.PENDING
    reviewer_notes: Optional[str] = None
    processing_time_seconds: float = 0.0
    errors: List[str] = Field(default_factory=list)


# ── Review Queue Item ──────────────────────────────────────────


class ReviewItem(BaseModel):
    """An item in the human review queue."""
    invoice_id: str
    source_file: str
    processed_at: str
    vendor_name: Optional[str] = None
    grand_total: Optional[float] = None
    currency: Optional[str] = None
    aggregate_score: int = 0
    risk_level: RiskLevel = RiskLevel.LOW
    triggered_checks: List[str] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.PENDING
    reviewer_notes: Optional[str] = None
    reviewed_at: Optional[str] = None
