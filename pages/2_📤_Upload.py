"""
📤 Upload — Upload and process invoices.

Features:
- Drag-and-drop file upload (PDF, images)
- Real-time processing progress
- Immediate results view with fraud check details
"""

import asyncio
import streamlit as st
from pathlib import Path
import tempfile
import os

st.markdown("# 📤 Upload & Process Invoices")
st.markdown("Upload invoice files for AI-powered fraud analysis. Supports PDF, JPEG, PNG, TIFF, BMP.")
st.markdown("---")

# ── File Upload ───────────────────────────────────────────────

uploaded_files = st.file_uploader(
    "Drop invoice files here",
    type=["pdf", "jpg", "jpeg", "png", "tiff", "tif", "bmp"],
    accept_multiple_files=True,
    help="Upload one or more invoice files for processing.",
    key="invoice_upload",
)

if uploaded_files:
    st.markdown(f"**{len(uploaded_files)} file(s) selected**")

    if st.button("🔍 Process Invoices", type="primary", use_container_width=True):
        from src.config import Config
        from src.pipeline import Pipeline

        config = Config.get()
        pipeline = Pipeline(config)

        results = []

        for i, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"---")
            st.markdown(f"### Processing: `{uploaded_file.name}`")

            progress_bar = st.progress(0, text="Initializing...")
            status_text = st.empty()

            # Save uploaded file to temp location
            temp_dir = Path(tempfile.mkdtemp())
            temp_path = temp_dir / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process with progress callback
            def progress_callback(stage: str, pct: float):
                progress_bar.progress(min(pct, 1.0), text=f"{stage}...")

            try:
                # Run the async pipeline
                loop = asyncio.new_event_loop()
                results_list = loop.run_until_complete(
                    pipeline.process_file(str(temp_path), progress_callback)
                )
                loop.close()

                # Handle single or multiple invoices
                if not isinstance(results_list, list):
                    results_list = [results_list]

                for result in results_list:
                    results.append(result)

                progress_bar.progress(1.0, text="✅ Complete!")

                # ── Display Results ───────────────────────────

                # If multiple invoices, show count
                if len(results_list) > 1:
                    st.success(f"📄 Detected {len(results_list)} invoices in this document")

                for idx, result in enumerate(results_list):
                    if len(results_list) > 1:
                        st.subheader(f"Invoice {idx + 1}: {result.invoice_id}")
                    else:
                        st.subheader(f"Invoice: {result.invoice_id}")

                    # Summary row
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Invoice ID", result.invoice_id)
                    with col2:
                        score = result.fraud_result.aggregate_score
                        st.metric("Risk Score", f"{score}/100")
                    with col3:
                        level = result.fraud_result.risk_level.value
                        emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}.get(level, "⚪")
                        st.metric("Risk Level", f"{emoji} {level}")
                    with col4:
                        status = result.review_status.value
                        st.metric("Status", status)

                    # Extracted fields
                    with st.expander("📋 Extracted Fields", expanded=False):
                        fields = result.extracted_fields
                        field_col1, field_col2 = st.columns(2)

                        with field_col1:
                            st.markdown(f"**Vendor:** {fields.vendor_name or '—'}")
                            st.markdown(f"**Invoice #:** {fields.invoice_number or '—'}")
                            st.markdown(f"**Date:** {fields.invoice_date or '—'}")
                            st.markdown(f"**Due Date:** {fields.due_date or '—'}")
                            st.markdown(f"**PO #:** {fields.po_number or '—'}")

                        with field_col2:
                            st.markdown(f"**Customer:** {fields.customer_name or '—'}")
                            st.markdown(f"**Currency:** {fields.currency or '—'}")
                            st.markdown(f"**Grand Total:** {fields.grand_total or '—'}")
                            st.markdown(f"**Tax Amount:** {fields.tax_amount or '—'}")
                        st.markdown(f"**Tax Rate:** {fields.tax_rate or '—'}")

                    if fields.line_items:
                        st.markdown("**Line Items:**")
                        import pandas as pd
                        items_data = [
                            {
                                "Description": li.description,
                                "Qty": li.quantity,
                                "Unit Price": li.unit_price,
                                "Line Total": li.line_total,
                            }
                            for li in fields.line_items
                        ]
                        st.dataframe(pd.DataFrame(items_data), hide_index=True, use_container_width=True)

                # Fraud check details
                with st.expander("🛡️ Fraud Check Results", expanded=True):
                    for check in result.fraud_result.checks:
                        severity_emoji = {
                            "LOW": "🟢", "MEDIUM": "🟡",
                            "HIGH": "🟠", "CRITICAL": "🔴",
                        }.get(check.severity.value, "⚪")

                        triggered_icon = "⚠️" if check.triggered else "✅"

                        with st.container():
                            c1, c2, c3 = st.columns([3, 1, 1])
                            with c1:
                                st.markdown(
                                    f"{triggered_icon} **{check.name.replace('_', ' ').title()}**"
                                )
                            with c2:
                                st.markdown(f"Score: **{check.score}**/100")
                            with c3:
                                st.markdown(f"{severity_emoji} {check.severity.value}")

                            if check.detail:
                                st.caption(check.detail)

                # Errors
                if result.errors:
                    with st.expander("⚠️ Processing Errors", expanded=False):
                        for err in result.errors:
                            st.warning(err)

                st.caption(f"Processing time: {result.processing_time_seconds:.1f}s")

            except Exception as e:
                progress_bar.progress(1.0, text="❌ Failed")
                st.error(f"Processing failed: {e}")

            finally:
                # Cleanup temp file
                try:
                    os.unlink(str(temp_path))
                    os.rmdir(str(temp_dir))
                except Exception:
                    pass

        # Summary
        if results:
            st.markdown("---")
            st.success(f"✅ Processed {len(results)} invoice(s) successfully.")

            flagged = sum(1 for r in results if r.fraud_result.any_triggered)
            if flagged > 0:
                st.warning(
                    f"🚨 {flagged} invoice(s) flagged for review. "
                    f"Go to the **Review Queue** to approve or reject."
                )

else:
    # Empty state
    st.markdown("""
    <div style="
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, rgba(26,26,46,0.3), rgba(15,52,96,0.2));
        border-radius: 12px;
        border: 2px dashed rgba(0,210,255,0.3);
        margin: 2rem 0;
    ">
        <h2 style="color: #00d2ff; margin-bottom: 0.5rem;">📄 Drop Files Here</h2>
        <p style="color: #8888aa;">
            Supported formats: PDF, JPEG, PNG, TIFF, BMP<br/>
            Maximum file size: 50 MB per file
        </p>
    </div>
    """, unsafe_allow_html=True)
