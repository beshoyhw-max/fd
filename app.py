"""
Invoice Fraud Detection System — Streamlit Application Entry Point.

Main app with sidebar navigation, branding, and session state initialization.
"""

import streamlit as st
from pathlib import Path

# Page configuration — must be the first Streamlit command
st.set_page_config(
    page_title="Invoice Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize session state ──────────────────────────────────

if "initialized" not in st.session_state:
    from src.config import Config
    config = Config.get()
    config.ensure_data_dirs()
    st.session_state.initialized = True
    st.session_state.processing = False
    st.session_state.watcher_running = False


# ── Custom CSS ────────────────────────────────────────────────

st.markdown("""
<style>
    /* Modern dark-themed styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0, 210, 255, 0.2);
    }
    .main-header h1 {
        color: #ffffff;
        margin: 0;
        font-size: 1.8rem;
    }
    .main-header p {
        color: #a0a0b0;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid rgba(0, 210, 255, 0.15);
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 210, 255, 0.4);
    }
    .stat-card .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00d2ff;
    }
    .stat-card .stat-label {
        font-size: 0.85rem;
        color: #8888aa;
        margin-top: 0.3rem;
    }
    .risk-high { color: #ff4444 !important; }
    .risk-medium { color: #ffaa00 !important; }
    .risk-low { color: #00ff88 !important; }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1 {
        color: #00d2ff;
        font-size: 1.3rem;
    }
    
    /* Status badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-critical { background: rgba(255, 68, 68, 0.2); color: #ff4444; }
    .badge-high { background: rgba(255, 170, 0, 0.2); color: #ffaa00; }
    .badge-medium { background: rgba(0, 210, 255, 0.2); color: #00d2ff; }
    .badge-low { background: rgba(0, 255, 136, 0.2); color: #00ff88; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🛡️ Fraud Detection")
    st.markdown("---")
    st.markdown("""
    **AI-Powered Invoice Analysis**
    
    13-layer fraud detection with image forensics, 
    statistical analysis, and behavioral monitoring.
    """)
    st.markdown("---")
    
    # System status
    st.markdown("### System Status")
    
    from src.storage.file_store import FileStore
    store = FileStore()
    stats = store.get_statistics()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Processed", stats["total"])
    with col2:
        st.metric("Flagged", stats["flagged"])
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Pending", stats["pending_review"])
    with col4:
        st.metric("Approved", stats["approved"] + stats["auto_approved"])
    
    st.markdown("---")
    st.caption("v1.0 • Enterprise Edition")


# ── Main Page Content ─────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🛡️ Invoice Fraud Detection System</h1>
    <p>AI-powered 13-layer fraud detection • Image forensics • Human-in-the-loop review</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### Welcome

This system processes invoices through **13 independent fraud detection engines** to identify 
financial irregularities, document tampering, AI-generated forgeries, and behavioral anomalies.

**Navigate using the sidebar** to access the system pages:

| Page | Description |
|---|---|
| 📊 **Dashboard** | Real-time analytics, risk distribution, vendor insights |
| 📤 **Upload** | Upload and process invoices (drag-and-drop) |
| 👁️ **Review Queue** | Review flagged invoices, approve or reject |
| 📁 **Folder Watch** | Auto-process invoices from a watched folder |
| ⚙️ **Settings** | Configure LLM, thresholds, and system parameters |
""")

# Quick stats
st.markdown("### Quick Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{stats['total']}</div>
        <div class="stat-label">Total Processed</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value risk-high">{stats['flagged']}</div>
        <div class="stat-label">Flagged for Review</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value risk-low">{stats['auto_approved']}</div>
        <div class="stat-label">Auto-Approved</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{stats['avg_score']}</div>
        <div class="stat-label">Avg. Risk Score</div>
    </div>
    """, unsafe_allow_html=True)
