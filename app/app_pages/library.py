"""Library: everything ingested, and what's inside each file (PLAN.md §2.3, §4.8).

The table stretches to the window (more columns on big monitors); text (transcripts, document Markdown) stays at reading
width. Recordings get a meeting summary (decisions + action items, Phase B) and Granite Guardian's checks of it against the
library's summary criteria (Phase C, PLAN.md §2.4), and a Speakers panel to name who spoke (PLAN.md §3.7: only named
speakers are labelled in summaries); documents show their Markdown, the tables and charts Granite Vision read, and their form
extractions.
"""

import io
import json
from typing import Literal

import streamlit as st

from granit.config import LOCAL_MODELS
from granit.models.phases import PhaseBusy
from granit.store.db import Source
from granit.ui.layout import READING_WIDTH, safe_md, table_height
from granit.ui.session import backend, open_store
from granit.ui.views import (
    audio_format,
    details,
    document_md,
    form_result,
    latest,
    library_rows,
    model_label,
    names_changed,
    needs_speakers,
    other_vocabulary,
    read_json,
    save_summary_criteria,
    source_param,
    speaker_rows,
    stamp,
    summary_checks,
    summary_criteria_text,
    transcript_md,
)

BadgeColor = Literal["green", "blue", "orange", "red", "gray"]
STATUS_COLOR: dict[str, BadgeColor] = {
    "ready": "green",
    "queued": "blue",
    "processing": "blue",
    "failed": "red",
}

st.title("Library")
store = open_store()
sources = store.sources()
if not sources:
    st.info(
        "Nothing here yet: add recordings and documents on the Ingest page.", icon=":material/info:"
    )
    st.page_link("app_pages/ingest.py", label="Go to Ingest", icon=":material/upload_file:")
    st.stop()

show = st.segmented_control(
    "Show",
    ["All", "Documents", "Recordings"],
    default="All",
    required=True,
    key="library_kind",
    label_visibility="collapsed",
)
kinds = {"All": ("audio", "document"), "Documents": ("document",), "Recordings": ("audio",)}[
    show or "All"
]
listed = [s for s in sources if s.kind in kinds]
rows = library_rows(listed)
event = st.dataframe(
    rows,
    hide_index=True,
    column_order=["Name", "Type", "Status", "Contents", "Chunks", "Size", "Added"],
    height=table_height(len(rows), max_rows=12),
    on_select="rerun",
    selection_mode="single-row",
    key="library_table",
)
if event.selection.rows:
    st.session_state.library_selected = rows[event.selection.rows[0]]["id"]
elif "library_selected" not in st.session_state:  # a deep link: /library?source=3
    st.session_state.library_selected = source_param(st.query_params.get("source"))
selected = st.session_state.get("library_selected")
if selected not in {s.id for s in listed}:
    st.caption("Select a row to see what's inside.")
    st.stop()
source = store.source(selected)


@st.dialog("Delete from the library?")
def confirm_delete(source: Source) -> None:
    st.markdown(
        f"**{safe_md(source.name)}**, its text, search index entries and extractions will be deleted."
        " Past answers that cited it keep their text but lose the link."
    )
    with st.container(horizontal=True):
        if st.button("Delete", type="primary", icon=":material/delete:"):
            open_store().delete_source(source)
            st.session_state.library_selected = None
            st.rerun()
        if st.button("Cancel"):
            st.rerun()


def show_summary(source: Source) -> None:
    row = latest(store.extractions(source.id), "summary")
    if row is None:
        st.caption("No summary yet: **Summarize meeting** writes one (decisions and action items).")
        return
    data = json.loads(row["content"])
    st.markdown(safe_md(data["summary"]))
    if data["decisions"]:
        st.markdown("**Decisions**")
        st.markdown("\n".join(f"- {safe_md(d)}" for d in data["decisions"]))
    if data["action_items"]:
        st.markdown("**Action items**")
        st.dataframe(
            [
                {"Owner": i["owner"] or "", "Task": i["task"], "Due": i["due"] or ""}
                for i in data["action_items"]
            ],
            hide_index=True,
            width=READING_WIDTH,
        )
    st.caption(
        f"Summarized {safe_md(stamp(row['created_at']))} UTC by {safe_md(model_label(row['model']))}"
    )
    if names_changed(row["content"], store.speaker_names(source.id)):
        st.caption(
            "Speaker names changed since this summary: **Summarize meeting** again to use them for owners."
        )
    show_checks(row["id"])


CHECK_COLOR: dict[str, BadgeColor] = {"met": "green", "not met": "orange"}


def show_checks(extraction_id: int) -> None:
    """Guardian's verdicts on this summary, one per criterion; the criteria are the library's and editable here."""
    st.markdown("**Checks**")
    checks = summary_checks(store, extraction_id)
    for c in checks:
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.badge(
                safe_md(c["Result"]), color=CHECK_COLOR.get(c["Result"], "gray"), width="content"
            )
            st.markdown(safe_md(c["Check"]))
    if not checks:
        st.caption("No checks set: add some below.")
    with st.container(horizontal=True, vertical_alignment="center"):
        if st.button(
            "Check with Guardian",
            icon=":material/fact_check:",
            disabled=all(c["Result"] != "not checked yet" for c in checks),
            help="Queues a Guardian check of every summary and answer not checked yet. It runs with the next batch:"
            " after a minute without questions, or now with **Process now** on the Ingest page.",
        ):
            store.enqueue_verify()
            st.toast(
                "Guardian will check this summary with the next batch.",
                icon=":material/fact_check:",
            )
    with st.expander("Edit the checks"):
        st.caption(
            "One requirement per line, written so that *yes* means it's met. They apply to every meeting summary."
            " Guardian's custom checks need testing: if a verdict looks wrong, reword the check."
        )
        text = st.text_area(
            "Summary checks",
            summary_criteria_text(store),
            label_visibility="collapsed",
            key=f"summary_criteria_{extraction_id}",
        )
        if st.button("Save checks", icon=":material/save:"):
            save_summary_criteria(store, text)
            st.rerun()


def show_speakers(source: Source, transcript: dict) -> None:
    """Name who spoke (PLAN.md §3.7). Only named speakers are labelled in summaries, so a wrong name is the user's to see."""
    names = store.speaker_names(source.id)
    rows = speaker_rows(transcript, names)
    if not rows:
        return
    named = sum(bool(r.name) for r in rows)
    with st.expander(
        f"Speakers · {named} of {len(rows)} named",
        icon=":material/record_voice_over:",
        expanded=named < len(rows),
    ):
        st.caption(
            "Name the people you recognize: summaries use the names to give action items an owner."
            " Leave a speaker blank if its samples mix two people (its lines are then summarized without a name),"
            " and give two speakers the same name if they're one person."
        )
        with st.form(f"speakers_{source.id}", border=False):
            path = store.file_path(source)
            entered: dict[int, str] = {}
            for row in rows:
                with st.container(border=True):
                    st.markdown(f"**Speaker {row.speaker:d}** · {safe_md(row.stats)}")
                    st.caption(
                        "\n\n".join(f"`{safe_md(t)}` {safe_md(text)}" for t, text in row.samples)
                    )
                    with st.container(horizontal=True, vertical_alignment="bottom"):
                        st.audio(
                            str(path),
                            format=audio_format(path),
                            start_time=row.clip[0],
                            end_time=row.clip[1],
                            alt=f"Speaker {row.speaker}'s longest turn",
                        )
                        entered[row.speaker] = st.text_input(
                            f"Name for speaker {row.speaker}",
                            value=row.name,
                            placeholder="Name or role",
                            key=f"speaker_name_{source.id}_{row.speaker}",
                        )
            if st.form_submit_button("Save names", icon=":material/save:"):
                store.set_speaker_names(source, entered)
                st.toast("Speaker names saved.", icon=":material/record_voice_over:")
                st.rerun()


@st.dialog("Re-transcribe this recording?")
def confirm_retranscribe(source: Source) -> None:
    st.markdown(
        f"**{safe_md(source.name)}** will be transcribed again on the next batch. Its speakers are numbered afresh,"
        " so the names given to them are cleared."
    )
    with st.container(horizontal=True):
        if st.button("Re-transcribe", type="primary", icon=":material/spellcheck:"):
            open_store().queue_ingest(source)
            st.rerun()
        if st.button("Cancel"):
            st.rerun()


def show_vision(source: Source) -> None:
    rows = [e for e in store.extractions(source.id) if e["kind"] in ("chart", "table")]
    if not rows:
        st.caption(
            "No charts were found. Tables are read by Docling; **Re-ingest with accurate tables** reads them"
            " again with Granite Vision."
        )
        return
    for e in rows:
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(f"**{safe_md(e['kind'].capitalize())}** · p. {e['page']:d}")
                if e["valid"]:
                    st.badge("valid", color="green", icon=":material/check:")
                else:
                    st.badge("invalid", color="red", icon=":material/close:")
            if e["crop"]:
                crop = store.derived_dir(source) / e["crop"]
                if crop.is_file():
                    st.image(str(crop), width=480, alt=f"{e['kind']} on page {e['page']}")
            if e["format"] == "csv":
                import pandas as pd

                try:
                    st.dataframe(pd.read_csv(io.StringIO(e["content"])), hide_index=True)
                except (ValueError, pd.errors.ParserError):
                    st.code(e["content"], language=None)
            else:
                st.code(e["content"], language=None)


with st.container(horizontal=True, vertical_alignment="center"):
    st.subheader(safe_md(source.name), width="content")
    st.badge(safe_md(source.status), color=STATUS_COLOR.get(source.status, "gray"))
st.caption(details(source))
if source.status == "failed":
    job = next((j for j in reversed(store.jobs("failed")) if j.source_id == source.id), None)
    if job is not None:
        st.error(f"Ingest failed: {safe_md(str(job.error))}", icon=":material/error:")

with st.container(horizontal=True):
    ready = source.status == "ready"
    if source.kind == "audio":
        summarize = st.button(
            "Summarize meeting", type="primary", icon=":material/summarize:", disabled=not ready
        )
        if st.button(
            "Re-transcribe",
            icon=":material/spellcheck:",
            disabled=not ready
            or not (
                other_vocabulary(source, store.vocabulary())
                or needs_speakers(source, LOCAL_MODELS["diarization"].is_built())
            ),
            help="Transcribes the recording again on the next batch, with the library's current names and terms (set"
            " on the Ingest page) and who spoke when. Enabled when the list has changed since this recording was"
            " transcribed, or when it was transcribed before speakers were available.",
        ):
            if store.speaker_names(source.id):
                confirm_retranscribe(source)
            else:
                store.queue_ingest(source)
                st.rerun()
    else:
        summarize = False
        if st.button(
            "Re-ingest with accurate tables",
            icon=":material/table_chart:",
            disabled=not ready or bool(source.info.get("accurate_tables")),
            help="Reads every table again with Granite Vision on the next batch. The current text stays searchable until then.",
        ):
            store.queue_ingest(source, {"vision_tables": True})
            st.rerun()
    if st.button("Delete", icon=":material/delete:", disabled=source.status == "processing"):
        confirm_delete(source)

if summarize:
    try:
        with st.spinner("Summarizing the meeting…"):
            backend().summarize(source)
    except PhaseBusy as busy:
        st.info(
            f"{safe_md(str(busy))}. Try again when Q&A is back.", icon=":material/hourglass_top:"
        )
    except Exception as exc:
        st.error(
            f"The summary failed: {safe_md(f'{type(exc).__name__}: {exc}')}",
            icon=":material/error:",
        )

if source.kind == "audio":
    transcript = read_json(store.derived_dir(source) / "transcript.json")
    if transcript is not None and source.status == "ready":
        with st.container(width=READING_WIDTH):
            show_speakers(source, transcript)
    summary_tab, transcript_tab = st.tabs(["Summary", "Transcript"])
    with summary_tab, st.container(width=READING_WIDTH):
        show_summary(source)
    with transcript_tab:
        if transcript is None:
            st.caption("The transcript appears here once the recording is ingested.")
        else:
            with st.container(width=READING_WIDTH, height=520):
                st.markdown(transcript_md(transcript, store.speaker_names(source.id)))
else:
    content_tab, vision_tab, forms_tab = st.tabs(["Content", "Tables and charts", "Fields"])
    with content_tab:
        md = store.derived_dir(source) / "document.md"
        if md.is_file():
            with st.container(width=READING_WIDTH, height=620):
                st.markdown(document_md(md.read_text()))
        else:
            st.caption("The document's text appears here once it's ingested.")
    with vision_tab:
        show_vision(source)
    with forms_tab:
        forms = [e for e in store.extractions(source.id) if e["kind"] == "form"]
        if not forms:
            st.caption("No fields extracted yet.")
        for e in reversed(forms):
            result = form_result(e)
            with st.container(border=True, width=READING_WIDTH):
                st.badge(
                    "schema valid" if result.valid else "invalid",
                    color="green" if result.valid else "red",
                )
                st.dataframe(result.fields, hide_index=True, column_order=["Field", "Value"])
                st.caption(
                    f"{safe_md(stamp(result.created_at))} UTC · {safe_md(model_label(result.model))}"
                )
        st.page_link("app_pages/extract.py", label="Extract fields", icon=":material/data_object:")
