import streamlit as st
from datetime import datetime
from typing import List, Dict, Any

from utils.s3_utils import s3_list_json, s3_read_json, s3_generate_presigned_url


def _parse_iso(ts: str) -> datetime:
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return datetime.min


def create_history_tab():
    st.markdown("### 🕓 Project History")
    st.caption("View past generations stored in S3")

    # Styles and scroll container
    st.markdown(
        """
        <style>
        .history-scroll { max-height: 540px; overflow-y: auto; padding-right: 8px; }
        .history-card { border: 1px solid #eee; border-radius: 8px; padding: 12px; margin-bottom: 10px; background: #fff; }
        .history-title { font-weight: 600; font-size: 0.95rem; }
        .history-meta { color: #666; font-size: 0.85rem; }
        .chip-ok { display:inline-block; padding:2px 8px; border-radius:12px; background:#e6ffed; color:#1a7f37; font-size:0.8rem; margin-left:8px; }
        .chip-bad { display:inline-block; padding:2px 8px; border-radius:12px; background:#ffecec; color:#b30000; font-size:0.8rem; margin-left:8px; }
        .btn { display:inline-block; padding:8px 14px; background:linear-gradient(45deg,#667eea,#764ba2); color:#fff; border-radius:6px; text-decoration:none; font-weight:600; }
        .btn.disabled { background:#ccc; pointer-events:none; color:#666; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        keys = s3_list_json(prefix="projects/")
    except Exception as e:
        st.error(f"Cannot list S3 projects: {e}")
        return

    if not keys:
        st.info("No history yet. Generate audio to see entries here.")
        return

    # Build entries
    entries: List[Dict[str, Any]] = []
    for key in keys:
        data = s3_read_json(key) or {}
        project_id = data.get("project_id") or key.split(
            "/")[-1].replace(".json", "")
        status = data.get("status", "INCOMPLETE")
        last_updated = data.get("last_updated") or "1970-01-01T00:00:00Z"
        dt = _parse_iso(last_updated)
        audio_key = f"projects/{project_id}/consolidated.mp3"
        url = s3_generate_presigned_url(
            audio_key, expires_seconds=3600) if status == "COMPLETED" else None
        entries.append({
            "project_id": project_id,
            "status": status,
            "last_updated": last_updated,
            "dt": dt,
            "url": url,
        })

    # Sort by most recent
    entries.sort(key=lambda e: e["dt"], reverse=True)

    # Render
    st.markdown('<div class="history-scroll">', unsafe_allow_html=True)
    for e in entries:
        status_ok = e["status"] == "COMPLETED"
        chip = '<span class="chip-ok">✅ Success</span>' if status_ok else '<span class="chip-bad">❌ Failed</span>'
        download_html = (
            f'<a class="btn" href="{e["url"]}" target="_blank">Download</a>' if status_ok and e.get(
                "url") else '<span class="btn disabled">No file</span>'
        )
        card_html = f"""
        <div class="history-card">
            <div class="history-title">{e["project_id"]} {chip}</div>
            <div class="history-meta">Updated: {e["last_updated"]}</div>
            <div style=\"margin-top:8px;\">{download_html}</div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
