"""
📊 Dashboard — Real-time analytics and fraud monitoring.

Displays:
- Summary statistics
- Risk distribution chart
- Recent activity timeline
- Top risky vendors
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

st.markdown("# 📊 Fraud Detection Dashboard")
st.markdown("Real-time analytics and risk monitoring across all processed invoices.")
st.markdown("---")

# ── Load data ─────────────────────────────────────────────────

from src.storage.file_store import FileStore
store = FileStore()
stats = store.get_statistics()
all_results = store.get_all_results()
vendor_stats = store.get_vendor_stats()

# ── Summary Cards ─────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Processed",
        stats["total"],
        help="Total invoices processed by the system",
    )

with col2:
    st.metric(
        "🚨 Flagged",
        stats["flagged"],
        help="Invoices with at least one triggered fraud check",
    )

with col3:
    st.metric(
        "⏳ Pending Review",
        stats["pending_review"],
        help="Flagged invoices awaiting human review",
    )

with col4:
    st.metric(
        "✅ Approved",
        stats["approved"] + stats["auto_approved"],
        help="Manually approved + auto-approved invoices",
    )

with col5:
    st.metric(
        "❌ Rejected",
        stats["rejected"],
        help="Invoices rejected after review",
    )

st.markdown("---")

# ── Charts ────────────────────────────────────────────────────

if stats["total"] == 0:
    st.info("📭 No invoices processed yet. Upload invoices from the **Upload** page to see analytics here.")
else:
    chart_col1, chart_col2 = st.columns(2)

    # Risk Distribution Pie Chart
    with chart_col1:
        st.markdown("### Risk Distribution")
        risk_data = stats["risk_distribution"]
        
        colors = {"LOW": "#00ff88", "MEDIUM": "#00d2ff", "HIGH": "#ffaa00", "CRITICAL": "#ff4444"}
        
        fig_risk = go.Figure(data=[go.Pie(
            labels=list(risk_data.keys()),
            values=list(risk_data.values()),
            hole=0.45,
            marker_colors=[colors.get(k, "#888") for k in risk_data.keys()],
            textinfo="label+value",
            textfont_size=13,
        )])
        fig_risk.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
            margin=dict(t=30, b=30, l=30, r=30),
            height=350,
            showlegend=True,
            legend=dict(font=dict(color="#aaa")),
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    # Review Status Bar Chart
    with chart_col2:
        st.markdown("### Review Status")
        status_data = {
            "Auto-Approved": stats["auto_approved"],
            "Pending": stats["pending_review"],
            "Approved": stats["approved"],
            "Rejected": stats["rejected"],
        }
        status_colors = ["#00ff88", "#ffaa00", "#00d2ff", "#ff4444"]

        fig_status = go.Figure(data=[go.Bar(
            x=list(status_data.keys()),
            y=list(status_data.values()),
            marker_color=status_colors,
            text=list(status_data.values()),
            textposition="auto",
        )])
        fig_status.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
            margin=dict(t=30, b=30, l=30, r=30),
            height=350,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        )
        st.plotly_chart(fig_status, use_container_width=True)

    st.markdown("---")

    # ── Score Distribution Histogram ──────────────────────────

    st.markdown("### Suspicion Score Distribution")
    if all_results:
        scores = [r.get("aggregate_score", 0) for r in all_results]
        fig_hist = go.Figure(data=[go.Histogram(
            x=scores,
            nbinsx=20,
            marker_color="#00d2ff",
            marker_line_color="#0f3460",
            marker_line_width=1,
        )])
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
            margin=dict(t=30, b=50, l=50, r=30),
            height=300,
            xaxis_title="Suspicion Rank Index",
            yaxis_title="Count",
            xaxis=dict(showgrid=False, range=[0, 100]),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # ── Top Risky Vendors ─────────────────────────────────────

    st.markdown("### Top Risky Vendors")
    if vendor_stats:
        top_vendors = vendor_stats[:10]
        df_vendors = pd.DataFrame(top_vendors)
        
        if not df_vendors.empty and "vendor_name" in df_vendors.columns:
            display_cols = {
                "vendor_name": "Vendor",
                "count": "Invoices",
                "flagged": "Flagged",
                "avg_score": "Avg Score",
                "total_amount": "Total Amount",
            }
            available_cols = {k: v for k, v in display_cols.items() if k in df_vendors.columns}
            df_display = df_vendors[list(available_cols.keys())].rename(columns=available_cols)
            
            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")

    # ── Recent Activity ───────────────────────────────────────

    st.markdown("### Recent Activity")
    recent = sorted(all_results, key=lambda x: x.get("processed_at", ""), reverse=True)[:10]

    if recent:
        df_recent = pd.DataFrame(recent)
        display_cols = {
            "invoice_id": "Invoice ID",
            "source_file": "File",
            "vendor_name": "Vendor",
            "grand_total": "Amount",
            "aggregate_score": "Risk Score",
            "risk_level": "Risk Level",
            "review_status": "Status",
            "processed_at": "Processed",
        }
        available_cols = {k: v for k, v in display_cols.items() if k in df_recent.columns}
        df_display = df_recent[list(available_cols.keys())].rename(columns=available_cols)

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
        )
