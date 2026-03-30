"""Streamlit frontend for the Neurons Creative Editor API."""

import io
import json
import threading
import time
import zipfile

import requests
import streamlit as st
from PIL import Image

API_BASE_URL = "http://localhost:8000"
APPLY_RECOMMENDATIONS_URL = f"{API_BASE_URL}/creative-editor/apply-recommendations"

RECOMMENDATION_TYPES = [
    "colour_mood",
    "copy_messaging",
    "contrast_salience",
    "composition",
]

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neurons Creative Editor",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 Neurons Creative Editor")
st.caption("Apply AI-powered recommendations to your creative assets.")

# ── Session state initialisation ─────────────────────────────────────────────
if "protected_regions" not in st.session_state:
    st.session_state.protected_regions = [""]

if "recommendations" not in st.session_state:
    st.session_state.recommendations = [
        {"id": "rec_1", "title": "", "description": "", "type": "contrast_salience"}
    ]

if "is_running" not in st.session_state:
    st.session_state.is_running = False

if "trigger_run" not in st.session_state:
    st.session_state.trigger_run = False

# Stores parsed results so they survive download-triggered reruns
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None

# ── Sidebar – API settings ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input(
        "API base URL",
        value=API_BASE_URL,
        help="URL where the FastAPI service is running.",
    )
    APPLY_RECOMMENDATIONS_URL = f"{api_url}/creative-editor/apply-recommendations"
    st.divider()
    st.caption("Make sure `uvicorn main:app` is running before submitting.")

# ── 1. Image upload ───────────────────────────────────────────────────────────
st.header("1. Upload Creative Image")
uploaded_file = st.file_uploader(
    "Select the image to edit",
    type=["png", "jpg", "jpeg", "webp"],
    help="Supported formats: PNG, JPG, JPEG, WEBP",
)
if uploaded_file:
    st.image(uploaded_file, caption="Original image", use_container_width=True)

# ── 2. Brand guidelines ───────────────────────────────────────────────────────
st.header("2. Brand Guidelines")

col_left, col_right = st.columns(2)

with col_left:
    typography = st.text_input(
        "Typography",
        value="Maintain existing font style and hierarchy for all text elements",
        help="Typography rules to maintain across all edits.",
    )
    aspect_ratio = st.text_input(
        "Aspect ratio",
        value="1572x1720",
        help="Aspect ratio constraint, e.g. '1572x1720'.",
    )

with col_right:
    brand_elements = st.text_area(
        "Brand elements",
        value="Ensure logo remains visible and legible at all times",
        help="Brand elements that must remain visible.",
        height=105,
    )

# Protected regions – dynamic list
st.subheader("Protected Regions")
st.caption("Regions that must not be modified. Add as many as needed.")

regions_to_delete = []
for i, region in enumerate(st.session_state.protected_regions):
    col_input, col_del = st.columns([9, 1])
    with col_input:
        st.session_state.protected_regions[i] = st.text_input(
            f"Region {i + 1}",
            value=region,
            key=f"region_{i}",
            label_visibility="collapsed",
            placeholder=f"e.g. Do not modify or remove the brand logo",
        )
    with col_del:
        if st.button("✕", key=f"del_region_{i}", help="Remove this region"):
            regions_to_delete.append(i)

for idx in sorted(regions_to_delete, reverse=True):
    st.session_state.protected_regions.pop(idx)
    st.rerun()

if st.button("＋ Add protected region", key="add_region"):
    st.session_state.protected_regions.append("")
    st.rerun()

# ── 3. Recommendations ────────────────────────────────────────────────────────
st.header("3. Recommendations")
st.caption("Define one or more recommendations to apply to the image.")

recs_to_delete = []
for i, rec in enumerate(st.session_state.recommendations):
    with st.expander(
        f"Recommendation {i + 1}: {rec['title'] or '(untitled)'}",
        expanded=True,
    ):
        c1, c2 = st.columns([3, 1])
        with c1:
            rec["id"] = st.text_input(
                "ID",
                value=rec["id"],
                key=f"rec_id_{i}",
                placeholder="e.g. rec_1",
            )
        with c2:
            rec["type"] = st.selectbox(
                "Type",
                options=RECOMMENDATION_TYPES,
                index=RECOMMENDATION_TYPES.index(rec["type"]),
                key=f"rec_type_{i}",
            )
        rec["title"] = st.text_input(
            "Title",
            value=rec["title"],
            key=f"rec_title_{i}",
            placeholder="e.g. Strengthen Headline Impact",
        )
        rec["description"] = st.text_area(
            "Description",
            value=rec["description"],
            key=f"rec_desc_{i}",
            placeholder="Detailed description of what to change and why…",
            height=100,
        )
        if st.button(f"✕ Remove recommendation {i + 1}", key=f"del_rec_{i}"):
            recs_to_delete.append(i)

for idx in sorted(recs_to_delete, reverse=True):
    st.session_state.recommendations.pop(idx)
    st.rerun()

if st.button("＋ Add recommendation", key="add_rec"):
    next_id = f"rec_{len(st.session_state.recommendations) + 1}"
    st.session_state.recommendations.append(
        {"id": next_id, "title": "", "description": "", "type": "contrast_salience"}
    )
    st.rerun()

# ── 4. Submit ─────────────────────────────────────────────────────────────────
st.divider()

def _validate() -> list[str]:
    errors: list[str] = []
    if not uploaded_file:
        errors.append("Please upload an image.")
    if not typography.strip():
        errors.append("Typography field is required.")
    if not aspect_ratio.strip():
        errors.append("Aspect ratio field is required.")
    if not brand_elements.strip():
        errors.append("Brand elements field is required.")
    if not any(r.strip() for r in st.session_state.protected_regions):
        errors.append("Add at least one protected region.")
    if not st.session_state.recommendations:
        errors.append("Add at least one recommendation.")
    for i, rec in enumerate(st.session_state.recommendations, start=1):
        if not rec["id"].strip():
            errors.append(f"Recommendation {i}: ID is required.")
        if not rec["title"].strip():
            errors.append(f"Recommendation {i}: Title is required.")
        if not rec["description"].strip():
            errors.append(f"Recommendation {i}: Description is required.")
    return errors


# Render the button disabled while the pipeline is running.
# On click: validate, store form snapshot in session state, set flags, rerun
# so the disabled state is visible before the blocking API call starts.
clicked = st.button(
    "🚀 Apply Recommendations",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.is_running,
)

if clicked and not st.session_state.is_running:
    errors = _validate()
    if errors:
        for err in errors:
            st.error(err)
    else:
        # Snapshot mutable widget values before the rerun loses local scope
        st.session_state._form_snapshot = {
            "image_name": uploaded_file.name,
            "image_bytes": uploaded_file.getvalue(),
            "image_type": uploaded_file.type,
            "typography": typography,
            "aspect_ratio": aspect_ratio,
            "brand_elements": brand_elements,
            "protected_regions": [r for r in st.session_state.protected_regions if r.strip()],
            "recommendations": json.dumps(st.session_state.recommendations),
        }
        st.session_state.is_running = True
        st.session_state.trigger_run = True
        st.rerun()  # re-render with disabled button before blocking call

# ── API call (runs in the rerun where trigger_run=True) ───────────────────────
if st.session_state.trigger_run:
    st.session_state.trigger_run = False
    snap = st.session_state._form_snapshot

    # Run the blocking HTTP call in a background thread so the main thread
    # can drive the progress bar.
    _result: dict = {}

    def _call_api() -> None:
        try:
            fields: list[tuple] = []
            for region in snap["protected_regions"]:
                fields.append(("protected_regions", region))
            fields.append(("typography", snap["typography"]))
            fields.append(("aspect_ratio", snap["aspect_ratio"]))
            fields.append(("brand_elements", snap["brand_elements"]))
            fields.append(("recommendations", snap["recommendations"]))

            files = [
                ("image", (snap["image_name"], snap["image_bytes"], snap["image_type"]))
            ]

            _result["response"] = requests.post(
                APPLY_RECOMMENDATIONS_URL,
                data=fields,
                files=files,
                timeout=300,
            )
        except Exception as exc:
            _result["error"] = exc

    thread = threading.Thread(target=_call_api, daemon=True)
    thread.start()

    # Fake progress bar: advances to 95 % over FAKE_DURATION seconds, then
    # waits for the thread to finish before jumping to 100 %.
    FAKE_DURATION = 100          # seconds for the bar to reach ~95 %
    TICK = 0.25                  # update interval in seconds
    TICKS = int(FAKE_DURATION / TICK)

    progress_bar = st.progress(0, text="Calling API… ⏳  0 %")

    for tick in range(TICKS):
        if not thread.is_alive():
            break
        time.sleep(TICK)
        pct = min(int((tick + 1) / TICKS * 95), 95)
        progress_bar.progress(pct, text=f"Calling API… ⏳  {pct} %")

    thread.join()  # ensure thread is done before reading _result
    progress_bar.progress(100, text="Done! ✅")
    time.sleep(0.4)             # brief pause so the user sees 100 %
    progress_bar.empty()

    if "error" in _result:
        exc = _result["error"]
        st.session_state.is_running = False
        if isinstance(exc, requests.exceptions.ConnectionError):
            st.error(
                f"Could not connect to the API at **{api_url}**. "
                "Make sure `uvicorn main:app` is running."
            )
        elif isinstance(exc, requests.exceptions.Timeout):
            st.error("Request timed out (300 s). The pipeline may still be running.")
        else:
            st.error(f"Unexpected error: {exc}")
    else:
        response = _result["response"]
        if response.status_code == 200:
            zip_bytes = io.BytesIO(response.content)
            with zipfile.ZipFile(zip_bytes) as zf:
                names = zf.namelist()
                image_names = [n for n in names if not n.endswith(".json")]
                audit_names = [n for n in names if n.endswith(".json")]
                images = {name: zf.read(name) for name in image_names}
                audit_json = zf.read(audit_names[0]).decode("utf-8") if audit_names else None

            zip_bytes.seek(0)
            st.session_state.pipeline_results = {
                "image_names": image_names,
                "images": images,
                "zip_bytes": zip_bytes.getvalue(),
                "audit_json": audit_json,
            }
            st.session_state.is_running = False
            st.rerun()
        else:
            st.session_state.is_running = False
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            st.error(f"API error {response.status_code}: {detail}")

# ── 5. Results (persisted in session state, survives download reruns) ─────────
if st.session_state.pipeline_results is not None:
    results = st.session_state.pipeline_results

    st.header("5. Results")
    st.success("✅ Pipeline completed successfully!")

    if results["image_names"]:
        st.subheader("Edited Images")
        cols = st.columns(min(len(results["image_names"]), 3))
        for idx, img_name in enumerate(results["image_names"]):
            img_data = results["images"][img_name]
            col = cols[idx % len(cols)]
            with col:
                try:
                    image = Image.open(io.BytesIO(img_data))
                    st.image(image, caption=img_name, use_container_width=True)
                except Exception:
                    st.warning(f"Could not preview {img_name}")
                st.download_button(
                    label=f"⬇️ Download {img_name}",
                    data=img_data,
                    file_name=img_name,
                    mime="image/png",
                    key=f"dl_{img_name}",
                    use_container_width=True,
                )

    st.subheader("Download All")
    st.download_button(
        label="⬇️ Download ZIP (all images + audit trail)",
        data=results["zip_bytes"],
        file_name="edited_creatives.zip",
        mime="application/zip",
        use_container_width=True,
    )

    if results["audit_json"]:
        with st.expander("📋 Audit Trail", expanded=False):
            st.json(json.loads(results["audit_json"]))

    st.divider()
    if st.button("🗑️ Clear results", use_container_width=False):
        st.session_state.pipeline_results = None
        st.rerun()
