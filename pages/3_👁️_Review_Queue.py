"""
👁️ Review Queue — Human-in-the-loop review workflow.

Features:
- Table of flagged invoices pending review (sorted by suspicion rank)
- Drill-down: extracted fields, fraud check details, ELA images
- Approve / Reject buttons with reviewer notes
"""

import streamlit as st
import pandas as pd
from pathlib import Path

st.markdown("# 👁️ Review Queue")
st.markdown("Flagged invoices requiring manual review. Sorted by **Suspicion Rank Index** (highest first).")
st.markdown("---")

from src.storage.file_store import FileStore
from src.models import ReviewStatus

store = FileStore()

# ── Filter controls ───────────────────────────────────────────

col_filter1, col_filter2, col_filter3 = st.columns([2, 2, 2])

with col_filter1:
    status_filter = st.selectbox(
        "Filter by status",
        ["Pending", "All Flagged", "Approved", "Rejected"],
        index=0,
    )

with col_filter2:
    sort_by = st.selectbox(
        "Sort by",
        ["Risk Score (High → Low)", "Risk Score (Low → High)", "Date (Newest)", "Date (Oldest)"],
        index=0,
    )

with col_filter3:
    search = st.text_input("Search vendor / invoice ID", placeholder="Type to filter...")

# ── Load data ─────────────────────────────────────────────────

if status_filter == "Pending":
    items = [
        r for r in store.get_all_results()
        if r.get("any_triggered") and r.get("review_status") == "PENDING"
    ]
elif status_filter == "All Flagged":
    items = store.get_flagged_invoices()
elif status_filter == "Approved":
    items = store.get_results_by_status(ReviewStatus.APPROVED)
elif status_filter == "Rejected":
    items = store.get_results_by_status(ReviewStatus.REJECTED)
else:
    items = store.get_all_results()

# Apply search filter
if search:
    search_lower = search.lower()
    items = [
        r for r in items
        if search_lower in (r.get("invoice_id", "").lower())
        or search_lower in (r.get("vendor_name", "") or "").lower()
        or search_lower in (r.get("source_file", "") or "").lower()
    ]

# Apply sort
if sort_by == "Risk Score (High → Low)":
    items.sort(key=lambda x: x.get("aggregate_score", 0), reverse=True)
elif sort_by == "Risk Score (Low → High)":
    items.sort(key=lambda x: x.get("aggregate_score", 0))
elif sort_by == "Date (Newest)":
    items.sort(key=lambda x: x.get("processed_at", ""), reverse=True)
elif sort_by == "Date (Oldest)":
    items.sort(key=lambda x: x.get("processed_at", ""))

# ── Queue Display ─────────────────────────────────────────────

st.markdown(f"**{len(items)} invoice(s)** matching filters")

if not items:
    st.info("🎉 No invoices in the review queue. All clear!")
else:
    for item in items:
        invoice_id = item.get("invoice_id", "unknown")
        vendor = item.get("vendor_name") or "Unknown Vendor"
        score = item.get("aggregate_score", 0)
        risk_level = item.get("risk_level", "LOW")
        total = item.get("grand_total")
        currency = item.get("currency", "")
        status = item.get("review_status", "PENDING")
        triggered = item.get("triggered_checks", [])
        processed_at = item.get("processed_at", "")

        # Risk color
        risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}.get(risk_level, "⚪")
        status_emoji = {"PENDING": "⏳", "APPROVED": "✅", "REJECTED": "❌", "AUTO_APPROVED": "✅"}.get(status, "⚪")

        # Card for each invoice
        with st.expander(
            f"{risk_emoji} **{vendor}** — Score: {score}/100 — "
            f"{currency} {total if total else 'N/A'} — {status_emoji} {status}",
            expanded=False,
        ):
            col1, col2, col3 = st.columns([2, 2, 2])

            with col1:
                st.markdown(f"**Invoice ID:** `{invoice_id}`")
                st.markdown(f"**File:** `{item.get('source_file', '')}`")
                st.markdown(f"**Processed:** {processed_at[:19] if processed_at else '—'}")

            with col2:
                st.markdown(f"**Risk Level:** {risk_emoji} {risk_level}")
                st.markdown(f"**Suspicion Score:** {score}/100")
                st.markdown(f"**Document Type:** {item.get('document_type', 'unknown')}")

            with col3:
                if total is not None:
                    st.markdown(f"**Amount:** {currency} {total:,.2f}")
                st.markdown(f"**Invoice #:** {item.get('invoice_number', '—')}")

            # Triggered checks
            if triggered:
                st.markdown("**⚠️ Triggered Checks:**")
                for check_name in triggered:
                    st.markdown(f"- 🚨 {check_name.replace('_', ' ').title()}")

            # Load full result for detailed view
            full_result = store.get_result(invoice_id)

            if full_result:
                # Fraud check details
                st.markdown("**📊 All Check Scores:**")
                check_data = []
                for check in full_result.fraud_result.checks:
                    check_data.append({
                        "Check": check.name.replace("_", " ").title(),
                        "Score": check.score,
                        "Severity": check.severity.value,
                        "Triggered": "⚠️ YES" if check.triggered else "✅ No",
                        "Detail": check.detail[:100] + "..." if len(check.detail) > 100 else check.detail,
                    })

                df_checks = pd.DataFrame(check_data)
                st.dataframe(df_checks, use_container_width=True, hide_index=True)

                # ELA image viewer
                if full_result.fraud_result.checks:
                    ela_check = next(
                        (c for c in full_result.fraud_result.checks if c.name == "ela_analysis"),
                        None,
                    )
                    if ela_check and ela_check.evidence:
                        ela_path = ela_check.evidence.get("ela_image_path")
                        if ela_path and Path(ela_path).exists():
                            st.markdown("**🔬 ELA Heatmap:**")
                            st.image(ela_path, caption="Error Level Analysis — bright regions indicate potential edits")

            # Review actions (only for pending items)
            if status == "PENDING":
                st.markdown("---")
                st.markdown("**Review Decision:**")

                notes = st.text_area(
                    "Reviewer notes",
                    key=f"notes_{invoice_id}",
                    placeholder="Add notes about your review decision...",
                )

                btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])

                with btn_col1:
                    if st.button("✅ Approve", key=f"approve_{invoice_id}", type="primary"):
                        store.update_review(
                            invoice_id,
                            ReviewStatus.APPROVED,
                            notes=notes or None,
                        )
                        st.success("Invoice approved!")
                        st.rerun()

                with btn_col2:
                    if st.button("❌ Reject", key=f"reject_{invoice_id}"):
                        store.update_review(
                            invoice_id,
                            ReviewStatus.REJECTED,
                            notes=notes or None,
                        )
                        st.error("Invoice rejected.")
                        st.rerun()
