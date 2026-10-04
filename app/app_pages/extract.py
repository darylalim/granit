"""Extract: a document + a JSON Schema → validated JSON (PLAN.md §1, §3.6).

Layout: the document page (520 px) and the fields (520 px) side by side from ~1512 px, stacked below that. Extraction runs
in the ingest phase (Granite Vision), so it's queued like an upload and runs with the next batch or "Process now".
"""

import json
from pathlib import Path

import streamlit as st

from granit.ui.layout import EXTRACT_PANEL_WIDTH, safe_md, two_panels
from granit.ui.session import backend, open_store, process_now_button
from granit.ui.views import (
    DOCUMENT_TYPES,
    EXAMPLE_SCHEMA,
    add_upload,
    check_schema,
    form_result,
    model_label,
    page_count,
    page_png,
    source_param,
    stamp,
)

SCHEMA_HELP = (
    "A JSON Schema for an object: one property per field, with a `description` saying what to look for."
    ' Dates with `"format": "date"` come back as YYYY-MM-DD.'
)


@st.cache_data(max_entries=32, show_spinner=False)
def page_image(path: str, page: int) -> bytes:
    return page_png(Path(path), page)


st.title("Extract")
store = open_store()
docs = [s for s in reversed(store.sources()) if s.kind == "document" and s.status != "failed"]
document_panel, fields_panel = two_panels("extract")

with fields_panel:
    with st.expander("Add a document", icon=":material/upload_file:", expanded=not docs):
        upload = st.file_uploader("Form or invoice", type=DOCUMENT_TYPES, key="extract_upload")
        if upload is not None and st.button("Add to library", icon=":material/add:"):
            source, _ = add_upload(store, upload.name, upload.getbuffer())
            st.session_state.extract_source = source.id
            st.rerun()
    if not docs:
        st.stop()
    ids = [s.id for s in docs]
    names = {s.id: s.name for s in docs}
    linked = source_param(st.query_params.get("source"))  # a deep link: /extract?source=3
    if linked in ids and not st.session_state.get("extract_linked"):
        st.session_state.extract_source, st.session_state.extract_linked = linked, True
    if st.session_state.get("extract_source") not in ids:
        st.session_state.extract_source = ids[0]
    source_id = st.selectbox("Document", ids, format_func=names.__getitem__, key="extract_source")
    source = store.source(source_id)
    forms = [e for e in store.extractions(source.id) if e["kind"] == "form"]
    pending = [j for j in store.jobs("queued") + store.jobs("running") if j.source_id == source.id]

    with st.expander("Schema", icon=":material/edit_note:", expanded=not forms):
        text = st.text_area(
            "JSON Schema",
            value=json.dumps(EXAMPLE_SCHEMA, indent=2),
            height=320,
            key="extract_schema",
            help=SCHEMA_HELP,
        )
        schema, problem = check_schema(text)
        if problem:
            st.error(safe_md(problem), icon=":material/error:")
        if st.button(
            "Extract fields",
            type="primary",
            icon=":material/data_object:",
            disabled=schema is None,
        ):
            store.enqueue_extract(source, schema or {})
            st.rerun()

    if pending:
        status = backend().status()
        with st.container(horizontal=True, vertical_alignment="center"):
            st.info(
                "Queued: runs with the next ingest batch (after a minute without questions).",
                icon=":material/schedule:",
            )
            process_now_button(status, key="extract_process")

    if forms:
        result = form_result(forms[-1])
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader("Fields")
            if result.valid:
                st.badge("schema valid", color="green", icon=":material/check:")
            else:
                st.badge("invalid", color="red", icon=":material/close:")
        st.dataframe(
            result.fields,
            hide_index=True,
            column_order=["Field", "Value", "Found"],
            column_config={"Found": st.column_config.CheckboxColumn(width="small")},
        )
        if result.missing:
            st.caption(f"Not found in the document: {safe_md(', '.join(result.missing))}")
        for error in result.errors:
            st.error(safe_md(error), icon=":material/error:")
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(
                f"{safe_md(stamp(result.created_at))} UTC · {safe_md(model_label(result.model))}"
            )
            st.space("stretch")
            st.download_button(
                "Download JSON",
                data=forms[-1]["content"],
                file_name=f"{Path(source.name).stem}.json",
                mime="application/json",
                icon=":material/download:",
            )
    elif not pending:
        st.caption("Edit the schema if needed, then **Extract fields**.")

with document_panel:
    path = store.file_path(source)
    pages = page_count(path)
    page = 1
    if pages > 1:
        page = st.number_input("Page", min_value=1, max_value=pages, value=1, key="extract_page")
    st.image(
        page_image(str(path), int(page)),
        width=EXTRACT_PANEL_WIDTH,
        alt=f"{source.name}, page {page}",
    )
