"""
📁 Folder Watch — Auto-process invoices from a watched directory.

Features:
- Start / Stop folder watcher
- Configure watched folder path
- Processing log display
"""

import asyncio
import streamlit as st
from pathlib import Path

st.markdown("# 📁 Folder Watch")
st.markdown("Automatically process invoices dropped into a watched folder.")
st.markdown("---")

from src.config import Config

config = Config.get()

# ── Initialize session state ──────────────────────────────────

if "watcher_log" not in st.session_state:
    st.session_state.watcher_log = []
if "watcher_running" not in st.session_state:
    st.session_state.watcher_running = False

# ── Configuration ─────────────────────────────────────────────

st.markdown("### Configuration")

col1, col2 = st.columns([3, 1])

with col1:
    watch_folder = st.text_input(
        "Watched Folder Path",
        value=str(Path(config.watcher_folder).resolve()),
        help="Invoice files dropped into this folder will be automatically processed.",
    )

with col2:
    poll_interval = st.number_input(
        "Poll Interval (s)",
        min_value=1,
        max_value=60,
        value=config.watcher_poll_interval,
        help="How often to check for new files.",
    )

# Ensure the folder exists
folder_path = Path(watch_folder)
if not folder_path.exists():
    try:
        folder_path.mkdir(parents=True, exist_ok=True)
        st.success(f"Created folder: {watch_folder}")
    except Exception as e:
        st.error(f"Cannot create folder: {e}")

st.markdown("---")

# ── Watcher Controls ──────────────────────────────────────────

st.markdown("### Watcher Status")

col_status, col_start, col_stop = st.columns([3, 1, 1])

with col_status:
    if st.session_state.watcher_running:
        st.markdown("🟢 **Watcher is RUNNING** — monitoring for new files")
    else:
        st.markdown("🔴 **Watcher is STOPPED** — not monitoring")

with col_start:
    if st.button("▶️ Start", disabled=st.session_state.watcher_running, use_container_width=True):
        st.session_state.watcher_running = True
        st.session_state.watcher_log.append(
            f"✅ Watcher started — monitoring: {watch_folder}"
        )
        st.rerun()

with col_stop:
    if st.button("⏹️ Stop", disabled=not st.session_state.watcher_running, use_container_width=True):
        st.session_state.watcher_running = False
        st.session_state.watcher_log.append("🛑 Watcher stopped")
        st.rerun()

st.markdown("---")

# ── Instructions ──────────────────────────────────────────────

st.markdown("### How It Works")

st.markdown(f"""
1. **Configure** the watched folder path above
2. **Start** the watcher using the Start button  
3. **Drop files** into the folder: `{watch_folder}`
4. Files are **automatically detected** and processed through the full fraud detection pipeline
5. Results appear in the **Dashboard** and flagged invoices go to the **Review Queue**

**Supported file types:** PDF, JPEG, PNG, TIFF, BMP
""")

st.info(
    "💡 **Tip:** For production use, configure the watched folder to point to your "
    "network share or ERP export directory for fully automated processing."
)

# ── Processing Log ────────────────────────────────────────────

st.markdown("---")
st.markdown("### Processing Log")

if st.session_state.watcher_log:
    for entry in reversed(st.session_state.watcher_log[-50:]):
        st.text(entry)
else:
    st.caption("No activity yet. Start the watcher and drop files to see processing logs.")

# Clear log button
if st.session_state.watcher_log:
    if st.button("🗑️ Clear Log"):
        st.session_state.watcher_log = []
        st.rerun()

# ── Folder Contents ───────────────────────────────────────────

st.markdown("---")
st.markdown("### Current Folder Contents")

if folder_path.exists():
    files = list(folder_path.iterdir())
    invoice_files = [
        f for f in files
        if f.suffix.lower() in [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"]
    ]

    if invoice_files:
        st.markdown(f"**{len(invoice_files)} invoice file(s)** in the watched folder:")
        for f in invoice_files:
            size_kb = f.stat().st_size / 1024
            st.text(f"  📄 {f.name} ({size_kb:.1f} KB)")

        if st.button("🔄 Process All Now", type="primary"):
            from src.pipeline import Pipeline

            pipeline = Pipeline(config)
            progress = st.progress(0, text="Processing...")

            results = []
            for i, f in enumerate(invoice_files):
                progress.progress(
                    (i + 1) / len(invoice_files),
                    text=f"Processing {f.name}..."
                )
                try:
                    loop = asyncio.new_event_loop()
                    results_list = loop.run_until_complete(
                        pipeline.process_file(str(f))
                    )
                    loop.close()

                    # Handle single or multiple invoices
                    if not isinstance(results_list, list):
                        results_list = [results_list]

                    for result in results_list:
                        results.append(result)
                        st.session_state.watcher_log.append(
                            f"✅ Processed: {result.invoice_id} — Score: {result.fraud_result.aggregate_score}"
                        )
                except Exception as e:
                    st.session_state.watcher_log.append(f"❌ Failed: {f.name} — {e}")

            progress.progress(1.0, text="Complete!")
            flagged = sum(1 for r in results if r.fraud_result.any_triggered)
            st.success(f"Processed {len(results)} invoices from {len(invoice_files)} files. {flagged} flagged for review.")
    else:
        st.caption("No invoice files in the watched folder.")
else:
    st.warning(f"Folder does not exist: {watch_folder}")
