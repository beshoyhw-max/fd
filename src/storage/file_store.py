"""
File-Based Storage Engine for Invoice Fraud Detection System.

Provides JSON per-invoice result storage, an in-memory index for
fast lookups, and CSV export functionality.
"""

import csv
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import Config
from src.models import (
    ProcessingResult,
    ReviewItem,
    ReviewStatus,
    RiskLevel,
)


class FileStore:
    """
    File-based storage engine.

    - Each invoice result → data/results/{invoice_id}.json
    - Index file → data/results/index.json (fast lookups)
    - CSV export → data/exports/
    """

    def __init__(self, config: Optional[Config] = None):
        self._config = config or Config.get()
        self._config.ensure_data_dirs()
        self._lock = threading.Lock()
        self._index: Dict[str, Dict[str, Any]] = {}
        self._load_index()

    # ── Index Management ──────────────────────────────────────

    @property
    def _index_path(self) -> Path:
        return self._config.results_dir / "index.json"

    def _load_index(self):
        """Load or rebuild the index from disk."""
        if self._index_path.exists():
            try:
                with open(self._index_path, "r", encoding="utf-8") as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, Exception):
                self._rebuild_index()
        else:
            self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild the index by scanning all result files."""
        self._index = {}
        results_dir = self._config.results_dir
        if not results_dir.exists():
            return
        for file in results_dir.glob("*.json"):
            if file.name == "index.json":
                continue
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                invoice_id = data.get("invoice_id", file.stem)
                self._index[invoice_id] = self._build_index_entry(data)
            except (json.JSONDecodeError, Exception):
                continue
        self._save_index()

    def _save_index(self):
        """Persist the index to disk."""
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, ensure_ascii=False)

    def _build_index_entry(self, data: Dict) -> Dict[str, Any]:
        """Extract index-relevant fields from a result record."""
        fields = data.get("extracted_fields", {})
        fraud = data.get("fraud_result", {})
        return {
            "source_file": data.get("source_file", ""),
            "processed_at": data.get("processed_at", ""),
            "document_type": data.get("document_type", ""),
            "vendor_name": fields.get("vendor_name"),
            "invoice_number": fields.get("invoice_number"),
            "grand_total": fields.get("grand_total"),
            "currency": fields.get("currency"),
            "aggregate_score": fraud.get("aggregate_score", 0),
            "risk_level": fraud.get("risk_level", "LOW"),
            "recommended_action": fraud.get("recommended_action", "APPROVE"),
            "any_triggered": fraud.get("any_triggered", False),
            "triggered_checks": fraud.get("triggered_checks", []),
            "review_status": data.get("review_status", "PENDING"),
            "reviewer_notes": data.get("reviewer_notes"),
        }

    # ── CRUD Operations ───────────────────────────────────────

    def save_result(self, result: ProcessingResult):
        """Save a processing result to disk and update the index."""
        with self._lock:
            # Save the full result JSON
            result_path = self._config.results_dir / f"{result.invoice_id}.json"
            data = result.model_dump(mode="json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Update the index
            self._index[result.invoice_id] = self._build_index_entry(data)
            self._save_index()

    def get_result(self, invoice_id: str) -> Optional[ProcessingResult]:
        """Load a full result by invoice ID."""
        result_path = self._config.results_dir / f"{invoice_id}.json"
        if not result_path.exists():
            return None
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ProcessingResult(**data)
        except Exception:
            return None

    def get_all_results(self) -> List[Dict[str, Any]]:
        """Return all index entries (lightweight summaries)."""
        return [
            {"invoice_id": k, **v}
            for k, v in self._index.items()
        ]

    def get_results_by_status(self, status: ReviewStatus) -> List[Dict[str, Any]]:
        """Filter index entries by review status."""
        return [
            {"invoice_id": k, **v}
            for k, v in self._index.items()
            if v.get("review_status") == status.value
        ]

    def get_flagged_invoices(self) -> List[Dict[str, Any]]:
        """Return all invoices where any fraud check was triggered."""
        return [
            {"invoice_id": k, **v}
            for k, v in self._index.items()
            if v.get("any_triggered", False)
        ]

    def get_review_queue(self) -> List[ReviewItem]:
        """
        Return the review queue — flagged invoices pending review,
        sorted by aggregate score descending (highest suspicion first).
        """
        flagged = [
            item for item in self.get_all_results()
            if item.get("any_triggered", False)
            and item.get("review_status") == ReviewStatus.PENDING.value
        ]
        flagged.sort(key=lambda x: x.get("aggregate_score", 0), reverse=True)

        return [
            ReviewItem(
                invoice_id=item["invoice_id"],
                source_file=item.get("source_file", ""),
                processed_at=item.get("processed_at", ""),
                vendor_name=item.get("vendor_name"),
                grand_total=item.get("grand_total"),
                currency=item.get("currency"),
                aggregate_score=item.get("aggregate_score", 0),
                risk_level=RiskLevel(item.get("risk_level", "LOW")),
                triggered_checks=item.get("triggered_checks", []),
                review_status=ReviewStatus.PENDING,
            )
            for item in flagged
        ]

    def update_review(
        self,
        invoice_id: str,
        status: ReviewStatus,
        notes: Optional[str] = None,
    ):
        """Update the review status of an invoice."""
        result = self.get_result(invoice_id)
        if result is None:
            return

        result.review_status = status
        result.reviewer_notes = notes
        self.save_result(result)

    # ── Statistics ─────────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics from the index."""
        total = len(self._index)
        if total == 0:
            return {
                "total": 0,
                "flagged": 0,
                "auto_approved": 0,
                "pending_review": 0,
                "approved": 0,
                "rejected": 0,
                "risk_distribution": {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0},
                "avg_score": 0,
            }

        flagged = sum(1 for v in self._index.values() if v.get("any_triggered"))
        auto_approved = sum(
            1 for v in self._index.values()
            if v.get("review_status") == ReviewStatus.AUTO_APPROVED.value
        )
        pending = sum(
            1 for v in self._index.values()
            if v.get("review_status") == ReviewStatus.PENDING.value
        )
        approved = sum(
            1 for v in self._index.values()
            if v.get("review_status") == ReviewStatus.APPROVED.value
        )
        rejected = sum(
            1 for v in self._index.values()
            if v.get("review_status") == ReviewStatus.REJECTED.value
        )

        risk_dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        total_score = 0
        for v in self._index.values():
            level = v.get("risk_level", "LOW")
            if level in risk_dist:
                risk_dist[level] += 1
            total_score += v.get("aggregate_score", 0)

        return {
            "total": total,
            "flagged": flagged,
            "auto_approved": auto_approved,
            "pending_review": pending,
            "approved": approved,
            "rejected": rejected,
            "risk_distribution": risk_dist,
            "avg_score": round(total_score / total, 1) if total > 0 else 0,
        }

    def get_vendor_stats(self) -> List[Dict[str, Any]]:
        """Aggregate risk statistics per vendor."""
        vendors: Dict[str, Dict] = {}
        for v in self._index.values():
            vendor = v.get("vendor_name") or "Unknown"
            if vendor not in vendors:
                vendors[vendor] = {
                    "vendor_name": vendor,
                    "count": 0,
                    "total_amount": 0,
                    "total_score": 0,
                    "flagged": 0,
                }
            vendors[vendor]["count"] += 1
            vendors[vendor]["total_amount"] += v.get("grand_total") or 0
            vendors[vendor]["total_score"] += v.get("aggregate_score", 0)
            if v.get("any_triggered"):
                vendors[vendor]["flagged"] += 1

        result = []
        for vendor_data in vendors.values():
            count = vendor_data["count"]
            vendor_data["avg_score"] = round(vendor_data["total_score"] / count, 1) if count > 0 else 0
            result.append(vendor_data)

        result.sort(key=lambda x: x["avg_score"], reverse=True)
        return result

    # ── CSV Export ─────────────────────────────────────────────

    def export_csv(self, filename: Optional[str] = None) -> Path:
        """Export all results to a CSV file."""
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"fraud_report_{timestamp}.csv"

        export_path = self._config.exports_dir / filename

        fieldnames = [
            "invoice_id", "source_file", "processed_at", "document_type",
            "vendor_name", "invoice_number", "grand_total", "currency",
            "aggregate_score", "risk_level", "recommended_action",
            "triggered_checks", "review_status", "reviewer_notes",
        ]

        with open(export_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for invoice_id, entry in self._index.items():
                row = {"invoice_id": invoice_id}
                for field in fieldnames:
                    if field == "invoice_id":
                        continue
                    val = entry.get(field, "")
                    if isinstance(val, list):
                        val = "; ".join(str(x) for x in val)
                    row[field] = val
                writer.writerow(row)

        return export_path

    # ── Duplicate Check Helpers ────────────────────────────────

    def get_historical_invoices(self, lookback_days: int = 365) -> List[Dict[str, Any]]:
        """
        Return lightweight invoice records for duplicate/pattern checks.
        Filters to invoices within the lookback period.
        """
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        cutoff_str = cutoff.isoformat() + "Z"

        results = []
        for invoice_id, entry in self._index.items():
            processed = entry.get("processed_at", "")
            if processed >= cutoff_str:
                results.append({"invoice_id": invoice_id, **entry})
        return results

    def invoice_exists(self, invoice_number: str) -> bool:
        """Check if an invoice number already exists in the index."""
        for v in self._index.values():
            if v.get("invoice_number") == invoice_number:
                return True
        return False
