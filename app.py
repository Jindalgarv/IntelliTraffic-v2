"""IntelliTraffic AI v2 — Clean Streamlit Interface.

Run with:
    streamlit run app.py

A clean, single-page traffic violation detection system with:
  - Image & video upload
  - Explainable violation evidence cards
  - Gemini "Second Opinion" validation
  - Analytics dashboard
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from pipeline import IntelliTrafficPipeline, SEVERITY
from video_processor import VideoProcessor, get_video_info
from utils.database import (
    delete_all_violations,
    get_all_violations,
    get_location_stats,
    get_repeat_offenders,
    get_violation_stats,
    init_db,
    insert_violation,
    update_review_status,
)

# ── Config ───────────────────────────────────────────────────────────────────

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
ANNOTATED_DIR = DATA_DIR / "annotated"
CROPS_DIR = DATA_DIR / "crops"
DB_PATH = DATA_DIR / "traffic_violations.db"

for d in (DATA_DIR, UPLOADS_DIR, ANNOTATED_DIR, CROPS_DIR):
    d.mkdir(parents=True, exist_ok=True)

LOCATIONS = {
    "Vastrapur Crossroad, Ahmedabad": "23.0389° N, 72.5298° E",
    "Lal Darwaza, Ahmedabad": "23.0225° N, 72.5714° E",
    "Hazratganj Junction, Lucknow": "26.8467° N, 80.9462° E",
    "MG Road, Bangalore": "12.9716° N, 77.6094° E",
    "Connaught Place, Delhi": "28.6315° N, 77.2167° E",
    "Custom Location": "",
}

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IntelliTraffic AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)
init_db(DB_PATH)

# ── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1rem; max-width: 1200px; }

    /* Header */
    .app-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .app-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 1rem;
    }

    /* KPI Cards */
    .kpi-row { display: flex; gap: 12px; margin-bottom: 1rem; }
    .kpi-card {
        flex: 1;
        background: linear-gradient(145deg, #1e1b4b 0%, #312e81 100%);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .kpi-label { font-size: 0.75rem; color: #a5b4fc; }
    .kpi-value { font-size: 1.8rem; font-weight: 800; color: #e0e7ff; }

    /* Evidence Card */
    .evidence-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        background: #fff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .evidence-card.violation {
        border-left: 4px solid #ef4444;
    }
    .violation-type {
        font-size: 1.1rem;
        font-weight: 700;
        color: #dc2626;
    }
    .reasoning-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        color: #166534;
    }
    .gemini-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        color: #1e3a8a;
    }

    /* Badges */
    .badge {
        display: inline-block;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.7rem;
        font-weight: 700;
    }
    .badge-red { color: #b91c1c; background: #fef2f2; border: 1px solid #fecaca; }
    .badge-green { color: #166534; background: #f0fdf4; border: 1px solid #bbf7d0; }
    .badge-yellow { color: #92400e; background: #fffbeb; border: 1px solid #fde68a; }
    .badge-blue { color: #1e40af; background: #eff6ff; border: 1px solid #bfdbfe; }

    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1f2937;
        margin: 1.5rem 0 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def badge(text: str, color: str = "blue") -> str:
    return f'<span class="badge badge-{color}">{text}</span>'


def generate_vid() -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"TVAI-{ts}-{uuid4().hex[:8].upper()}"


def compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_upload(uploaded_file, data: bytes) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = UPLOADS_DIR / f"{ts}_{uploaded_file.name.replace(' ', '_')}"
    path.write_bytes(data)
    return path


def crop_violation(image: Image.Image, bbox: List[float]) -> Image.Image:
    w, h = image.size
    x1 = max(0, int(bbox[0]) - 10)
    y1 = max(0, int(bbox[1]) - 10)
    x2 = min(w, int(bbox[2]) + 10)
    y2 = min(h, int(bbox[3]) + 10)
    return image.crop((x1, y1, x2, y2))


@st.cache_resource
def get_pipeline(conf: float) -> IntelliTrafficPipeline:
    return IntelliTrafficPipeline(confidence=conf)


# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <div class="app-title">IntelliTraffic AI</div>
    <div class="app-subtitle">
        Computer Vision Pipeline · Explainable Evidence · Automated Enforcement
    </div>
</div>
""", unsafe_allow_html=True)

# ── Top KPI Bar ──────────────────────────────────────────────────────────────

db_df = get_all_violations(DB_PATH)
total = len(db_df)
pending = int((db_df["review_status"] == "Pending Review").sum()) if total else 0
approved = int((db_df["review_status"] == "Approved").sum()) if total else 0
plates = int(db_df["ocr_plate_number"].nunique()) if total else 0

st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-card"><div class="kpi-label">Total Violations</div><div class="kpi-value">{total}</div></div>
    <div class="kpi-card"><div class="kpi-label">Pending Review</div><div class="kpi-value">{pending}</div></div>
    <div class="kpi-card"><div class="kpi-label">Approved</div><div class="kpi-value">{approved}</div></div>
    <div class="kpi-card"><div class="kpi-label">Unique Vehicles</div><div class="kpi-value">{plates}</div></div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_detect, tab_review, tab_analytics = st.tabs([
    "🔍 Detect Violations", "📋 Review Dashboard", "📊 Analytics"
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: DETECTION
# ═══════════════════════════════════════════════════════════════════════════

with tab_detect:
    # ── Settings row ──
    s1, s2, s3, s4, s5 = st.columns([1, 1, 1, 1, 1])
    with s1:
        traffic_light = st.selectbox("Traffic Signal", ["RED", "GREEN", "YELLOW"])
    with s2:
        confidence = st.slider("Confidence", 0.15, 0.80, 0.25, 0.05)
    with s3:
        location = st.selectbox("Location", list(LOCATIONS.keys()))
        gps = LOCATIONS[location]
    with s4:
        stop_line = st.slider("Stop Line", 0.30, 0.90, 0.60, 0.05)
    with s5:
        gemini_key = st.text_input("Gemini Key (optional)", value=GEMINI_API_KEY, type="password")

    st.divider()

    # ── Upload ──
    upload_col, mode_col = st.columns([3, 1])
    with upload_col:
        uploaded = st.file_uploader(
            "Upload traffic image or video",
            type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov"],
            help="Supports images and short video clips (up to 60 seconds).",
        )
    with mode_col:
        if uploaded:
            is_video = uploaded.name.lower().endswith((".mp4", ".avi", ".mov"))
            st.info(f"**{'Video' if is_video else 'Image'}** mode")
            if is_video:
                frame_interval = st.number_input("Process every N frames", 3, 30, 5)

    if uploaded is None:
        st.info("Upload a traffic image or video to begin detection.")
    else:
        file_bytes = uploaded.getvalue()
        is_video = uploaded.name.lower().endswith((".mp4", ".avi", ".mov"))

        if is_video:
            # ── VIDEO PROCESSING ──
            st.markdown('<div class="section-header">🎥 Video Processing</div>', unsafe_allow_html=True)

            # Save video to temp
            temp_path = UPLOADS_DIR / f"temp_{uploaded.name}"
            temp_path.write_bytes(file_bytes)
            info = get_video_info(str(temp_path))

            st.caption(
                f"Duration: {info.get('duration_sec', 0):.1f}s · "
                f"FPS: {info.get('fps', 0):.0f} · "
                f"Resolution: {info.get('resolution', '?')}"
            )

            if st.button("Process Video", type="primary", use_container_width=True):
                pipe = get_pipeline(confidence)
                vp = VideoProcessor(frame_interval=frame_interval, max_frames=100)

                progress = st.progress(0, text="Processing frames...")

                def update_progress(current, total):
                    progress.progress(current / total, text=f"Frame {current}/{total}")

                def detect_fn(img, **kw):
                    return pipe.analyze(
                        img,
                        traffic_light=traffic_light,
                        stop_line_ratio=stop_line,
                        location=location,
                    )

                result = vp.process_video(
                    str(temp_path), detect_fn, progress_fn=update_progress
                )
                progress.empty()
                st.session_state["video_result"] = result

                if result.get("error"):
                    st.error(result["error"])
                else:
                    st.success(
                        f"Processed {result['processed_frames']} frames · "
                        f"Found {len(result['violations'])} unique violations"
                    )

            # Show video results
            if "video_result" in st.session_state:
                result = st.session_state["video_result"]
                violations = result.get("violations", [])

                if violations:
                    st.markdown('<div class="section-header">Detected Violations</div>', unsafe_allow_html=True)
                    for i, v in enumerate(violations):
                        with st.container(border=True):
                            c1, c2, c3 = st.columns([2, 1, 1])
                            with c1:
                                st.markdown(f"**{v['violation_type']}** — {v.get('vehicle_type', '?')}")
                                st.caption(v.get("details", ""))
                            with c2:
                                st.metric("Confidence", f"{v.get('confidence', 0):.2f}")
                            with c3:
                                st.metric("Seen in", f"{v.get('times_seen', 1)} frames")

                # Timeline
                timeline = result.get("timeline", [])
                if timeline:
                    st.markdown('<div class="section-header">Timeline</div>', unsafe_allow_html=True)
                    tl_df = pd.DataFrame(timeline)
                    st.line_chart(tl_df.set_index("timestamp_sec")["violations_count"])

                # Show annotated frames
                annotated_frames = result.get("annotated_frames", [])
                if annotated_frames:
                    st.markdown('<div class="section-header">Key Frames</div>', unsafe_allow_html=True)
                    cols = st.columns(min(4, len(annotated_frames)))
                    for idx, af in enumerate(annotated_frames[:8]):
                        with cols[idx % len(cols)]:
                            st.image(af["image"], caption=f"Frame {af['frame_number']}", use_container_width=True)

        else:
            # ── IMAGE PROCESSING ──
            try:
                image = Image.open(BytesIO(file_bytes)).convert("RGB")
            except Exception:
                st.error("Could not open the uploaded file as an image.")
                st.stop()

            # Preview
            p1, p2 = st.columns(2)
            with p1:
                st.image(image, caption=f"Original · {image.width}×{image.height}", use_container_width=True)

            if st.button("Run Detection", type="primary", use_container_width=True):
                with st.spinner("Running YOLO detection pipeline..."):
                    pipe = get_pipeline(confidence)
                    result = pipe.analyze(
                        image,
                        traffic_light=traffic_light,
                        stop_line_ratio=stop_line,
                        location=location,
                    )
                    # Save state
                    img_hash = compute_hash(file_bytes)
                    img_path = save_upload(uploaded, file_bytes)
                    st.session_state["result"] = result
                    st.session_state["img_hash"] = img_hash
                    st.session_state["img_path"] = str(img_path)
                    st.session_state["location"] = location
                    st.session_state["gps"] = gps
                    st.session_state["saved_violations"] = set()
                    st.rerun()

            # ── Show results ──
            if "result" in st.session_state:
                result = st.session_state["result"]
                annotated = result["annotated"]
                violations = result["violations"]
                summary = result["summary"]

                with p2:
                    st.image(annotated, caption="Annotated", use_container_width=True)

                # Summary metrics
                st.markdown('<div class="section-header">Detection Summary</div>', unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Vehicles", summary["total_vehicles"])
                m2.metric("Persons", summary["total_persons"])
                m3.metric("Plates Found", summary["total_plates"])
                m4.metric("Violations", summary["total_violations"])

                # Evidence hash
                img_hash = st.session_state.get("img_hash", "")
                if img_hash:
                    st.caption(f"Evidence Integrity (SHA-256): `{img_hash[:48]}...`")

                # ── Violation Evidence Cards ──
                if not violations:
                    st.success("No violations detected under current settings.")
                else:
                    st.markdown('<div class="section-header">Violation Evidence</div>', unsafe_allow_html=True)

                    for i, v in enumerate(violations):
                        with st.container(border=True):
                            # Header row
                            hc1, hc2 = st.columns([3, 1])
                            with hc1:
                                sev_badge = badge(f"Severity: {v['severity']}/10", "red")
                                vtype_badge = badge(v.get("vehicle_type", "?"), "blue")
                                st.markdown(
                                    f'<span class="violation-type">🚨 {v["violation_type"]}</span> '
                                    f'{sev_badge} {vtype_badge}',
                                    unsafe_allow_html=True,
                                )
                            with hc2:
                                st.metric("Confidence", f"{v.get('confidence', 0):.3f}")

                            # Evidence crops
                            try:
                                vehicle_crop = crop_violation(result["enhanced_image"], v["bbox"])
                                ec1, ec2, ec3 = st.columns([1, 1, 2])
                                with ec1:
                                    st.image(vehicle_crop, caption="Vehicle Crop", use_container_width=True)
                                with ec2:
                                    if v.get("plate_crop"):
                                        st.image(v["plate_crop"], caption="Plate", use_container_width=True)
                                    elif v.get("enhanced_plate_crop"):
                                        st.image(v["enhanced_plate_crop"], caption="Plate (enhanced)", use_container_width=True)
                                    else:
                                        st.caption("No plate detected")
                                with ec3:
                                    st.markdown(f"**License Plate:** `{v.get('license_plate', 'N/A')}`")
                                    st.markdown(f"**OCR Engine:** {v.get('ocr_engine', 'none')}")
                                    st.markdown(f"**OCR Confidence:** {v.get('ocr_confidence', 0):.2f}")
                                    if v.get("rider_count"):
                                        st.markdown(f"**Riders Detected:** {v['rider_count']}")
                            except Exception:
                                st.caption("Could not generate crop.")

                            # Explainable reasoning
                            reasoning = v.get("reasoning", {})
                            if reasoning:
                                st.markdown(
                                    f'<div class="reasoning-box">'
                                    f'<strong>How this was detected:</strong> '
                                    f'{v.get("details", "")}<br>'
                                    f'<em>Method: {reasoning.get("method", "N/A")}</em>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                            # Gemini Second Opinion
                            if gemini_key:
                                gemini_btn_key = f"gemini_{i}"
                                if st.button(f"🤖 Get Gemini Second Opinion", key=gemini_btn_key):
                                    with st.spinner("Asking Gemini..."):
                                        from gemini_validator import validate_violation
                                        try:
                                            vehicle_crop = crop_violation(result["enhanced_image"], v["bbox"])
                                        except Exception:
                                            vehicle_crop = result["enhanced_image"]
                                        gv = validate_violation(
                                            vehicle_crop, v["violation_type"], gemini_key
                                        )
                                        if gv.get("error"):
                                            st.warning(f"Gemini error: {gv['error']}")
                                        else:
                                            status = "✅ Confirmed" if gv["confirmed"] else "❌ Not Confirmed"
                                            st.markdown(
                                                f'<div class="gemini-box">'
                                                f'<strong>Gemini Second Opinion: {status}</strong><br>'
                                                f'{gv.get("reasoning", "")}<br>'
                                                f'<em>Confidence: {gv.get("confidence", "?")}'
                                                f'{" · Plate: " + gv["plate_text"] if gv.get("plate_text") else ""}'
                                                f'</em></div>',
                                                unsafe_allow_html=True,
                                            )

                            # Individual Save button
                            st.divider()
                            if i not in st.session_state.get("saved_violations", set()):
                                if st.button(f"💾 Save this Violation", key=f"save_viol_{i}"):
                                    vid = generate_vid()
                                    ann_path = ANNOTATED_DIR / f"{vid}.jpg"
                                    annotated.save(str(ann_path))
                                    try:
                                        vc = crop_violation(result["enhanced_image"], v["bbox"])
                                        crop_path = CROPS_DIR / f"{vid}_crop.jpg"
                                        vc.save(str(crop_path))
                                    except Exception:
                                        crop_path = ""

                                    insert_violation(
                                        violation_id=vid,
                                        image_filename=Path(st.session_state.get("img_path", "")).name,
                                        vehicle_type=v.get("vehicle_type", "unknown"),
                                        violation_type=v["violation_type"],
                                        violations_json=[v["violation_type"]],
                                        confidence=v.get("confidence", 0),
                                        ocr_plate_number=v.get("license_plate", "N/A"),
                                        ocr_confidence=v.get("ocr_confidence"),
                                        ocr_engine=v.get("ocr_engine", ""),
                                        speed_mph=None,
                                        severity=v.get("severity", 5),
                                        details=v.get("details", ""),
                                        original_image_path=st.session_state.get("img_path", ""),
                                        annotated_image_path=str(ann_path),
                                        evidence_path=str(crop_path),
                                        image_hash=st.session_state.get("img_hash", ""),
                                        location_name=st.session_state.get("location", ""),
                                        gps_coordinates=st.session_state.get("gps", ""),
                                        review_status="Pending Review",
                                        db_path=DB_PATH,
                                    )
                                    if "saved_violations" not in st.session_state:
                                        st.session_state["saved_violations"] = set()
                                    st.session_state["saved_violations"].add(i)
                                    st.rerun()
                            else:
                                st.success("✅ Saved to database")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: REVIEW DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

with tab_review:
    st.markdown('<div class="section-header">Human Review Dashboard</div>', unsafe_allow_html=True)
    review_df = get_all_violations(DB_PATH)

    if review_df.empty:
        st.info("No violations saved yet. Run detection and save violations first.")
    else:
        # Filters
        f1, f2, f3 = st.columns(3)
        with f1:
            status_filter = st.selectbox(
                "Status", ["All", "Pending Review", "Approved", "Rejected"]
            )
        with f2:
            vtype_filter = st.selectbox(
                "Violation Type",
                ["All"] + sorted(review_df["violation_type"].dropna().unique().tolist()),
            )
        with f3:
            st.metric("Records", len(review_df))

        filtered = review_df.copy()
        if status_filter != "All":
            filtered = filtered[filtered["review_status"] == status_filter]
        if vtype_filter != "All":
            filtered = filtered[filtered["violation_type"] == vtype_filter]

        # Table
        display_cols = ["id", "violation_id", "timestamp", "violation_type",
                        "confidence", "ocr_plate_number", "review_status"]
        avail = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[avail], use_container_width=True, hide_index=True)

        # Review individual record
        if not filtered.empty:
            st.divider()
            id_map = {
                rid: str(vid) if vid else f"#{rid}"
                for rid, vid in zip(filtered["id"], filtered["violation_id"])
            }
            sel = st.selectbox(
                "Select violation to review",
                filtered["id"].tolist(),
                format_func=lambda x: id_map.get(x, f"#{x}"),
            )
            row = filtered[filtered["id"] == sel].iloc[0]

            rc1, rc2 = st.columns([2, 1])
            with rc1:
                ann_p = row.get("annotated_image_path", "")
                if ann_p and Path(str(ann_p)).exists():
                    st.image(str(ann_p), caption=f"Evidence: {row.get('violation_id')}", use_container_width=True)
                else:
                    st.warning("Evidence image not found.")
            with rc2:
                status = str(row.get("review_status", "Pending Review"))
                color = {"Approved": "green", "Rejected": "red", "Pending Review": "yellow"}.get(status, "blue")
                st.markdown(badge(status, color), unsafe_allow_html=True)
                st.markdown(f"**Violation:** {row.get('violation_type')}")
                st.markdown(f"**Plate:** `{row.get('ocr_plate_number', 'N/A')}`")
                st.markdown(f"**Confidence:** {row.get('confidence', 0):.3f}")
                st.markdown(f"**Severity:** {row.get('severity', '?')}/10")
                st.markdown(f"**Location:** {row.get('location_name', 'N/A')}")
                st.markdown(f"**Time:** {row.get('timestamp', '')}")

                # Review buttons
                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("✅ Approve", key=f"app_{sel}"):
                        update_review_status(int(sel), "Approved", DB_PATH)
                        st.rerun()
                with b2:
                    if st.button("❌ Reject", key=f"rej_{sel}"):
                        update_review_status(int(sel), "Rejected", DB_PATH)
                        st.rerun()
                with b3:
                    if st.button("🔍 Manual", key=f"man_{sel}"):
                        update_review_status(int(sel), "Needs Manual Check", DB_PATH)
                        st.rerun()

        # Danger zone
        with st.expander("⚠️ Danger Zone"):
            if st.button("Clear All Records", type="secondary"):
                delete_all_violations(DB_PATH)
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════

with tab_analytics:
    st.markdown('<div class="section-header">Analytics & Risk Intelligence</div>', unsafe_allow_html=True)
    analytics_df = get_all_violations(DB_PATH)

    if analytics_df.empty:
        st.info("No data yet. Detect and save violations to see analytics.")
    else:
        ac1, ac2 = st.columns(2)

        with ac1:
            st.markdown("#### Violation Distribution")
            vtype_counts = analytics_df["violation_type"].value_counts()
            st.bar_chart(vtype_counts)

        with ac2:
            st.markdown("#### Review Status")
            status_counts = analytics_df["review_status"].value_counts()
            st.bar_chart(status_counts)

        # Daily trend
        stats_df = get_violation_stats(DB_PATH)
        if not stats_df.empty:
            st.markdown("#### Daily Trend")
            pivot = stats_df.pivot_table(
                index="date", columns="violation_type",
                values="count", aggfunc="sum", fill_value=0
            )
            st.line_chart(pivot)

        # Location hotspots
        loc_df = get_location_stats(DB_PATH)
        if not loc_df.empty:
            st.markdown("#### Location Hotspots")
            st.dataframe(loc_df, use_container_width=True, hide_index=True)

        # Repeat offenders
        st.divider()
        st.markdown("#### Repeat Offenders")
        offenders = get_repeat_offenders(DB_PATH, min_violations=1)
        if offenders.empty:
            st.caption("No repeat offenders detected yet.")
        else:
            for _, row in offenders.iterrows():
                with st.container(border=True):
                    oc1, oc2, oc3 = st.columns([2, 1, 1])
                    with oc1:
                        st.markdown(f"**🚗 `{row.get('license_plate', 'N/A')}`**")
                        st.caption(str(row.get("violation_types", "")))
                    with oc2:
                        st.metric("Violations", int(row.get("total_violations", 0)))
                    with oc3:
                        score = float(row.get("risk_score", 0))
                        color = "red" if score >= 70 else "yellow" if score >= 40 else "green"
                        st.markdown(f"Risk: {badge(f'{score:.0f}', color)}", unsafe_allow_html=True)
