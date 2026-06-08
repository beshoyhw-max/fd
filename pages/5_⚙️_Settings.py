"""
⚙️ Settings — System configuration management.

Features:
- LLM API URL and model name
- Risk thresholds (low/medium/high boundaries)
- Approval limit for threshold-splitting
- Concurrency limit
- Watched folder path
- Save settings (updates config.yaml)
- Export data to CSV
"""

import streamlit as st

st.markdown("# ⚙️ System Settings")
st.markdown("Configure the fraud detection system parameters. Changes are saved to `config.yaml`.")
st.markdown("---")

from src.config import Config
from src.storage.file_store import FileStore

config = Config.get()

# ── LLM Configuration ────────────────────────────────────────

st.markdown("### 🤖 LLM Configuration")

col1, col2 = st.columns(2)

with col1:
    llm_base_url = st.text_input(
        "LLM API Base URL",
        value=config.llm_base_url,
        help="OpenAI-compatible API endpoint (e.g., http://localhost:11434/v1 for Ollama)",
    )
    llm_model = st.text_input(
        "Model Name",
        value=config.llm_model,
        help="Model identifier (e.g., llava, gpt-4-vision-preview)",
    )

with col2:
    llm_temperature = st.slider(
        "Temperature",
        min_value=0.0, max_value=1.0,
        value=config.llm_temperature,
        step=0.05,
        help="Lower = more deterministic. Recommended: 0.05-0.15 for extraction.",
    )
    llm_max_tokens = st.number_input(
        "Max Tokens",
        min_value=256, max_value=16384,
        value=config.llm_max_tokens,
        step=256,
    )
    llm_timeout = st.number_input(
        "Timeout (seconds)",
        min_value=10, max_value=600,
        value=config.llm_timeout,
        step=10,
    )

# Test LLM connection
if st.button("🔌 Test LLM Connection"):
    import asyncio
    from src.extraction.llm_client import LLMClient

    with st.spinner("Testing connection..."):
        try:
            # Temporarily create a client with the new settings
            temp_config = Config.get()
            client = LLMClient(temp_config)
            loop = asyncio.new_event_loop()
            is_healthy = loop.run_until_complete(client.health_check())
            loop.close()

            if is_healthy:
                st.success("✅ LLM server is reachable and responding!")
            else:
                st.error("❌ LLM server is not responding. Check the URL and ensure the server is running.")
        except Exception as e:
            st.error(f"❌ Connection failed: {e}")

st.markdown("---")

# ── OCR Configuration ─────────────────────────────────────────

st.markdown("### 📝 OCR Configuration")

col1, col2 = st.columns(2)

with col1:
    ocr_enabled = st.checkbox(
        "Enable OCR Fallback",
        value=config.ocr_enabled,
        help="Use Tesseract OCR when the Vision LLM fails.",
    )

with col2:
    ocr_languages = st.text_input(
        "OCR Languages",
        value=", ".join(config.ocr_languages),
        help="Comma-separated Tesseract language codes (e.g., eng, ara, chi_sim)",
    )

st.markdown("---")

# ── Risk Thresholds ───────────────────────────────────────────

st.markdown("### 📊 Risk Thresholds")
st.caption("Define the Suspicion Rank Index boundaries for risk levels.")

col1, col2, col3 = st.columns(3)

with col1:
    threshold_low = st.number_input(
        "LOW → MEDIUM threshold",
        min_value=1, max_value=99,
        value=config.risk_threshold_low,
        help="Scores 0 to this value = LOW risk",
    )

with col2:
    threshold_medium = st.number_input(
        "MEDIUM → HIGH threshold",
        min_value=2, max_value=100,
        value=config.risk_threshold_medium,
        help="Scores above this value = HIGH risk",
    )

with col3:
    approval_limit = st.number_input(
        "Approval Limit ($)",
        min_value=100.0, max_value=1000000.0,
        value=config.approval_limit,
        step=500.0,
        help="Threshold splitting detection limit",
    )

st.markdown("---")

# ── Pipeline Configuration ────────────────────────────────────

st.markdown("### ⚡ Pipeline Configuration")

col1, col2 = st.columns(2)

with col1:
    concurrency = st.number_input(
        "Concurrency (parallel workers)",
        min_value=1, max_value=20,
        value=config.concurrency,
        help="Number of invoices processed simultaneously",
    )

with col2:
    max_file_size = st.number_input(
        "Max File Size (MB)",
        min_value=1, max_value=200,
        value=config.max_file_size_mb,
    )

st.markdown("---")

# ── Fraud Check Weights ───────────────────────────────────────

st.markdown("### ⚖️ Fraud Check Weights")
st.caption("Adjust the relative importance of each fraud check in the Suspicion Rank Index calculation.")

weights = config.fraud_weights.copy()

# Display in 3 columns
check_names = list(weights.keys())
cols = st.columns(3)

for i, check_name in enumerate(check_names):
    with cols[i % 3]:
        display_name = check_name.replace("_", " ").title()
        weights[check_name] = st.slider(
            display_name,
            min_value=0, max_value=20,
            value=weights[check_name],
            key=f"weight_{check_name}",
        )

st.markdown("---")

# ── Save Settings ─────────────────────────────────────────────

col_save, col_reset, col_spacer = st.columns([1, 1, 3])

with col_save:
    if st.button("💾 Save Settings", type="primary", use_container_width=True):
        updates = {
            "llm": {
                "base_url": llm_base_url,
                "model": llm_model,
                "temperature": llm_temperature,
                "max_tokens": llm_max_tokens,
                "timeout_seconds": llm_timeout,
            },
            "ocr": {
                "enabled": ocr_enabled,
                "languages": [l.strip() for l in ocr_languages.split(",") if l.strip()],
            },
            "risk_thresholds": {
                "low": threshold_low,
                "medium": threshold_medium,
            },
            "approval_limit": approval_limit,
            "pipeline": {
                "concurrency": concurrency,
                "max_file_size_mb": max_file_size,
            },
            "fraud_weights": weights,
        }

        config.update(updates)
        Config.reset()  # Reset singleton to reload
        st.success("✅ Settings saved to config.yaml!")

with col_reset:
    if st.button("🔄 Reset to Defaults", use_container_width=True):
        import shutil
        from src.config import CONFIG_PATH
        # The easiest way to reset is to delete and let defaults load
        if CONFIG_PATH.exists():
            CONFIG_PATH.unlink()
        Config.reset()
        st.success("Settings reset to defaults. Refresh the page.")
        st.rerun()

st.markdown("---")

# ── Data Management ───────────────────────────────────────────

st.markdown("### 📦 Data Management")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Export Results to CSV**")
    if st.button("📥 Export CSV Report"):
        store = FileStore()
        export_path = store.export_csv()
        st.success(f"✅ Exported to: `{export_path}`")
        
        # Offer download
        with open(export_path, "r", encoding="utf-8") as f:
            csv_data = f.read()
        st.download_button(
            "⬇️ Download CSV",
            data=csv_data,
            file_name=export_path.name,
            mime="text/csv",
        )

with col2:
    st.markdown("**System Statistics**")
    store = FileStore()
    stats = store.get_statistics()
    st.json(stats)
