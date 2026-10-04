"""Streamlit front end. It only sends Inngest events and polls for results, so it has no
direct dependency on the vector database or any AI provider."""

from __future__ import annotations

import os
import time
from pathlib import Path

import inngest
import requests
import streamlit as st

from science_rag import events
from science_rag.config import get_settings

st.set_page_config(page_title="Science RAG Analyzer", page_icon="🔬", layout="centered")

INNGEST_API_BASE = os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")
DONE_STATUSES = {"Completed", "Succeeded", "Success", "Finished"}
FAILED_STATUSES = {"Failed", "Cancelled"}


@st.cache_resource
def get_client() -> inngest.Inngest:
    return inngest.Inngest(app_id=events.APP_ID, is_production=False)


def save_upload(file) -> Path:
    uploads_dir = get_settings().uploads_dir
    uploads_dir.mkdir(parents=True, exist_ok=True)
    path = uploads_dir / Path(file.name).name  # .name strips any directory components
    path.write_bytes(file.getbuffer())
    return path.resolve()


def send_event(name: str, data: dict) -> str:
    return get_client().send_sync(inngest.Event(name=name, data=data))[0]


def wait_for_output(event_id: str, timeout_s: float = 240.0, poll_s: float = 0.5) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        response = requests.get(f"{INNGEST_API_BASE}/events/{event_id}/runs", timeout=10)
        response.raise_for_status()
        runs = response.json().get("data", [])
        if runs:
            status = runs[0].get("status")
            if status in DONE_STATUSES:
                return runs[0].get("output") or {}
            if status in FAILED_STATUSES:
                raise RuntimeError(f"Run {status}: {runs[0].get('output')}")
        time.sleep(poll_s)
    raise TimeoutError("Timed out waiting for the answer")


def upload_section(title: str, types: list[str], key: str, event_name: str, path_field: str):
    """File uploader that triggers its event once per file (Streamlit reruns the script)."""
    st.title(title)
    uploaded = st.file_uploader("Choose a file", type=types, key=key)
    state_key = f"sent_{key}"
    if uploaded is None:
        st.session_state.pop(state_key, None)
        return
    file_key = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get(state_key) != file_key:
        path = save_upload(uploaded)
        send_event(event_name, {path_field: str(path), "source_id": path.name})
        st.session_state[state_key] = file_key
    st.success(
        f"Ingestion triggered for {uploaded.name}. Progress is visible in the Inngest dashboard."
    )


upload_section("Upload a PDF", ["pdf"], "pdf", events.INGEST_PDF, "pdf_path")
st.divider()
upload_section(
    "Upload a picture, graph or drawing",
    ["png", "jpg", "jpeg", "webp"],
    "image",
    events.INGEST_IMAGE,
    "image_path",
)
st.divider()

st.title("Ask a question about your notes")
with st.form("ask"):
    question = st.text_input("Your question")
    top_k = st.number_input("Chunks to retrieve", min_value=1, max_value=20, value=5)
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    try:
        with st.spinner("Searching your notes..."):
            event_id = send_event(
                events.ASK_QUESTION, {"question": question.strip(), "top_k": int(top_k)}
            )
            output = wait_for_output(event_id)

        st.subheader("Answer")
        st.write(output.get("answer") or "(No answer)")

        if output.get("sources"):
            st.caption("Sources")
            for source in output["sources"]:
                st.write(f"- {source}")

        figures = [f for f in output.get("images", []) if os.path.isfile(f["path"])]
        if figures:
            st.subheader("Related figures")
            for fig in figures:
                where = fig["source"] + (f", page {fig['page']}" if fig.get("page") else "")
                st.image(fig["path"], caption=f"{where} (match {fig['score']:.2f})")
                with st.expander("What this figure shows"):
                    st.write(fig["caption"])
    except Exception as exc:  # noqa: BLE001 - show any failure in the page
        st.error(f"Query failed: {exc}")
