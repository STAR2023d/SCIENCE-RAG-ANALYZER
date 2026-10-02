import os
import time
from pathlib import Path

import inngest
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()



st.set_page_config(page_title="BASIC INTERGRATED SCIENCE SYNTHESIZER", page_icon="📄", layout="centered")

st.title("BASIC INTERGRATED SCIENCE SYNTHESIZER")
@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / Path(file.name).name
    file_path.write_bytes(file.getbuffer())
    return file_path


def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    client.send_sync(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )



st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    file_key = f"{uploaded.name}:{uploaded.size}"
    # Only trigger ingestion once per uploaded file, not on every Streamlit rerun
    if st.session_state.get("last_ingested") != file_key:
        with st.spinner("Uploading and triggering ingestion..."):
            path = save_uploaded_pdf(uploaded)
            send_rag_ingest_event(path)
            time.sleep(0.3)
        st.session_state["last_ingested"] = file_key
        st.session_state["last_ingested_name"] = path.name
    st.success(f"Triggered ingestion for: {st.session_state['last_ingested_name']}")
    st.caption("You can upload another PDF if you like.")
else:
    st.session_state.pop("last_ingested", None)

st.divider()
st.title("Are you in grade 4-9? Ask me questions about science")


def send_rag_query_event(question: str, top_k: int) -> str:
    client = get_inngest_client()
    event_ids = client.send_sync(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )
    return event_ids[0]


def _inngest_api_base() -> str:
    # Local dev server default; configurable via env
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def wait_for_run_output(
    event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5
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
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for run output (last status: {last_status})"
            )
        time.sleep(poll_interval_s)


with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input(
        "How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1
    )
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    with st.spinner("wait wait waaaait...here comes the answer!"):
        event_id = send_rag_query_event(question.strip(), int(top_k))
        output = wait_for_run_output(event_id)
        answer = output.get("answer", "")
        sources = output.get("sources", [])

    st.subheader("Answer")
    st.write(answer or "(No answer)")
    if sources:
        st.caption("Sources")
        for s in sources:
            st.write(f"- {s}")