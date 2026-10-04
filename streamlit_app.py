import os
import time
from pathlib import Path

import inngest
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Science RAG", page_icon="📄", layout="centered")


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def save_upload(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / Path(file.name).name
    file_path.write_bytes(file.getbuffer())
    return file_path


def send_rag_ingest_event(pdf_path: Path) -> None:
    # This one event triggers BOTH text ingestion and image ingestion
    get_inngest_client().send_sync(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


def send_image_ingest_event(image_path: Path) -> None:
    get_inngest_client().send_sync(
        inngest.Event(
            name="rag/ingest_image",
            data={
                "image_path": str(image_path.resolve()),
                "source_id": image_path.name,
            },
        )
    )


def send_rag_query_event(question: str, top_k: int) -> str:
    event_ids = get_inngest_client().send_sync(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={"question": question, "top_k": top_k},
        )
    )
    return event_ids[0]


def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json().get("data", [])


def wait_for_run_output(
    event_id: str, timeout_s: float = 240.0, poll_interval_s: float = 0.5
) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}: {run.get('output')}")
        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for run output (last status: {last_status})"
            )
        time.sleep(poll_interval_s)


# ---------------- PDF upload ----------------
st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    file_key = f"{uploaded.name}:{uploaded.size}"
    # Only trigger ingestion once per uploaded file, not on every Streamlit rerun
    if st.session_state.get("last_ingested") != file_key:
        with st.spinner("Uploading and triggering ingestion..."):
            path = save_upload(uploaded)
            send_rag_ingest_event(path)
            time.sleep(0.3)
        st.session_state["last_ingested"] = file_key
        st.session_state["last_ingested_name"] = path.name
    st.success(f"Triggered text + image ingestion for: {st.session_state['last_ingested_name']}")
    st.caption("Check the Inngest dashboard for progress. Images are described one by one.")
else:
    st.session_state.pop("last_ingested", None)

# ---------------- standalone image upload ----------------
st.divider()
st.title("Upload a picture, graph or drawing")
uploaded_img = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=False,
    key="image_uploader",
)

if uploaded_img is not None:
    img_key = f"{uploaded_img.name}:{uploaded_img.size}"
    if st.session_state.get("last_image") != img_key:
        with st.spinner("Uploading and triggering image ingestion..."):
            img_path = save_upload(uploaded_img)
            send_image_ingest_event(img_path)
            time.sleep(0.3)
        st.session_state["last_image"] = img_key
        st.session_state["last_image_name"] = img_path.name
    st.success(f"Triggered ingestion for image: {st.session_state['last_image_name']}")
else:
    st.session_state.pop("last_image", None)

# ---------------- ask ----------------
st.divider()
st.title("Ask a question about your notes")

with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input(
        "How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1
    )
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    try:
        with st.spinner("Sending event and generating answer..."):
            event_id = send_rag_query_event(question.strip(), int(top_k))
            output = wait_for_run_output(event_id)
            answer = output.get("answer", "")
            sources = output.get("sources", [])
            images = output.get("images", [])

        st.subheader("Answer")
        st.write(answer or "(No answer)")

        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")

        if images:
            st.subheader("Related figures")
            for img in images:
                if not os.path.isfile(img["path"]):
                    continue
                where = f"{img['source']}" + (f", page {img['page']}" if img.get("page") else "")
                st.image(img["path"], caption=f"{where} (match {img['score']:.2f})")
                with st.expander("What this figure shows"):
                    st.write(img["caption"])
    except Exception as e:
        st.error(f"Query failed: {e}")