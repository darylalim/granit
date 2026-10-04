"""Ingest: add recordings and documents, watch the queue (PLAN.md §2.1, §3.6).

Uploads are queued, not processed at once: the models for ingest and for Q&A don't fit in memory together. Queued jobs run
as one batch after a minute without questions, or right away with "Process now". "Accurate tables (slower)" re-reads every
table in the uploaded documents with Granite Vision.
"""

from datetime import datetime

import streamlit as st

from granit.ui.layout import reading_column, safe_md, table_height
from granit.ui.session import backend, open_store, plural, process_now_button
from granit.ui.views import UPLOAD_TYPES, add_upload, job_rows

ACCURATE_HELP = (
    "Re-reads every table in these documents with Granite Vision (about 5 s per table). "
    "Worth it for scans and complex tables; Docling's own tables are usually right. Recordings ignore it."
)

st.title("Ingest")
st.caption("Files, models and answers stay on this Mac.")


@st.fragment(run_every=2)
def queue() -> None:
    status = backend().status()
    store = open_store()
    rows = job_rows(store.job_rows(limit=30))
    st.subheader("Queue")
    with st.container(horizontal=True, vertical_alignment="center"):
        process_now_button(status, key="ingest_process")
        if status.queued:
            st.caption(
                f"{plural(status.queued, 'job')} waiting. Q&A pauses while they run, then comes back on its own."
            )
    if not rows:
        st.caption("Nothing queued yet.")
        return
    st.dataframe(
        rows,
        hide_index=True,
        column_order=["File", "Task", "Status", "Attempts", "Queued", "Error"],
        column_config={
            "File": st.column_config.TextColumn(width="medium"),
            "Task": st.column_config.TextColumn(width="medium"),
            "Attempts": st.column_config.NumberColumn(width="small"),
        },
        height=table_height(len(rows)),
        key="ingest_jobs",
    )
    failed = [r for r in rows if r["Status"] == "failed"]
    for row in failed[:5]:
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge("failed", color="red", icon=":material/error:")
            st.markdown(f"**{safe_md(row['File'])}**: {safe_md(row['Error'][:200])}")
            st.space("stretch")
            if st.button("Retry", key=f"retry_{row['id']}", icon=":material/replay:"):
                store.retry(row["id"])
                st.rerun()
    if backend().events:
        with st.expander("Activity"):
            for t, line in reversed(backend().events):
                st.text(f"{datetime.fromtimestamp(t):%H:%M:%S}  {line}")


with reading_column("ingest"):
    with st.form("ingest_upload", clear_on_submit=True, border=True):
        files = st.file_uploader(
            "Recordings and documents",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            help="Audio (WAV, FLAC, MP3, M4A, …) and documents (PDF, PNG, JPEG, TIFF, …). English only.",
        )
        accurate = st.toggle("Accurate tables (slower)", help=ACCURATE_HELP)
        submitted = st.form_submit_button("Add to queue", type="primary", icon=":material/add:")
    if submitted and files:
        store = open_store()
        added, existing = [], []
        for f in files:
            try:
                source, created = add_upload(store, f.name, f.getbuffer(), accurate)
            except ValueError as exc:
                st.error(safe_md(str(exc)), icon=":material/error:")
                continue
            (added if created else existing).append(source.name)
        if added:
            st.success(f"Queued {plural(len(added), 'file')}: {safe_md(', '.join(added))}")
        if existing:
            st.info(
                f"Already in your library: {safe_md(', '.join(existing))}. To read a document's tables again"
                " with Granite Vision, use **Re-ingest with accurate tables** in the Library.",
                icon=":material/info:",
            )
    queue()
