"""Ask: questions about everything ingested, answered with citations (PLAN.md §1, §2.1.1, §4.8).

Layout: a 720 px conversation with the chat input under it, and a 380 px sources panel beside it on wide windows (below on
narrow ones). Answers stream in; thinking ("low" by default) is folded into a compact "Thought for N s" toggle. A question
asked while Q&A is paused is kept and asked as soon as Q&A is back. "Check answers" queues a Guardian verify job (Phase C);
its verdicts appear as badges under each answer (PLAN.md §2.4).
"""

import time
from typing import Any

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from granit.models.phases import PhaseBusy
from granit.ui.layout import READING_WIDTH, answer_md, reading_with_side_panel, safe_md
from granit.ui.session import backend, open_store
from granit.ui.views import (
    ChatTurn,
    SourceRef,
    attach_checks,
    turn_from_answer,
    turns_from_history,
    unchecked_answers,
)

USER = ":material/person:"  # neutral avatars: Streamlit's default red / orange read as error / warning (§4.7)
ASSISTANT = ":material/neurology:"
CHECK_HELP = (
    "Granite Guardian checks whether each answer is supported by the passages it was based on (about 8 s per answer)."
    " It runs with the next batch: after a minute without questions, or now with **Process now** on the Ingest page."
)
THINKING_HELP = (
    "How much the model reasons before answering. **Low** (default) is quick and keeps answers short; "
    "**off** tends to ramble; **on** can use its whole budget thinking."
)

store = open_store()
if "ask_turns" not in st.session_state:
    st.session_state.ask_turns = turns_from_history(store, limit=10)
turns: list[ChatTurn] = st.session_state.ask_turns
attach_checks(store, turns)
selected = st.session_state.get("ask_selected")
if selected is None or selected >= len(turns):
    selected = len(turns) - 1


def select(i: int) -> None:
    st.session_state.ask_selected = i


def new_conversation() -> None:
    st.session_state.ask_turns = []
    st.session_state.ask_selected = None


def citation_badges(sources: list[SourceRef]) -> None:
    with st.container(horizontal=True, gap="xsmall"):
        for s in sources:
            st.badge(f"[{s.number:d}] {safe_md(s.citation)}", color="gray")


def show_turn(i: int, turn: ChatTurn) -> None:
    with st.chat_message("user", avatar=USER):
        st.markdown(safe_md(turn.question))
    with st.chat_message("assistant", avatar=ASSISTANT):
        if turn.reasoning:
            with st.expander("Thinking", type="compact"):
                st.markdown(safe_md(turn.reasoning))
        st.markdown(answer_md(turn.answer))
        if turn.cited:
            citation_badges(turn.cited)
        if turn.checks:
            with st.container(horizontal=True, gap="xsmall"):
                for c in turn.checks:
                    st.badge(safe_md(c.label), color=c.color, icon=c.icon, help=safe_md(c.help))
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            if turn.seconds is not None:
                st.caption(f"{turn.seconds:.1f} s")
            if turn.sources and i != selected:
                st.button(
                    "Show sources",
                    key=f"ask_sources_{i}",
                    type="tertiary",
                    icon=":material/description:",
                    on_click=select,
                    args=(i,),
                )


class Stream:
    """Shows the answer as it arrives (``on_delta``), at most every 50 ms so the page keeps up."""

    def __init__(self, status: Any, reasoning: DeltaGenerator, answer: DeltaGenerator) -> None:
        self.status, self.reasoning_slot, self.answer_slot = status, reasoning, answer
        self.reasoning = self.content = ""
        self.started = time.monotonic()
        self.thought_s: float | None = None
        self.shown = 0.0

    def __call__(self, delta: tuple[str, str]) -> None:
        kind, text = delta
        if kind == "reasoning":
            if not self.reasoning:
                self.status.update(label=":shimmer[Thinking]")
            self.reasoning += text
        else:
            if self.thought_s is None:
                self.thought_s = time.monotonic() - self.started
                label = (
                    f"Thought for {self.thought_s:.0f} s"
                    if self.reasoning
                    else "Searched your library"
                )
                self.status.update(label=label, state="complete")
            self.content += text
        if time.monotonic() - self.shown > 0.05:
            self.show(cursor=True)

    def show(self, cursor: bool = False) -> None:
        self.shown = time.monotonic()
        if self.reasoning:
            self.reasoning_slot.markdown(safe_md(self.reasoning))
        if self.content:
            self.answer_slot.markdown(answer_md(self.content) + (" ▌" if cursor else ""))


st.title("Ask")
conversation, sources_panel = reading_with_side_panel("ask")

with conversation:
    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.segmented_control(
            "Thinking",
            ["off", "low", "on"],
            default="low",
            required=True,
            key="ask_thinking",
            help=THINKING_HELP,
        )
        st.space("stretch")
        unchecked = unchecked_answers(store)
        if st.button(
            "Check answers",
            type="tertiary",
            icon=":material/fact_check:",
            disabled=not unchecked,
            help=CHECK_HELP,
        ):
            store.enqueue_verify()
            st.toast(
                f"Guardian will check {unchecked} answer{'s' * (unchecked != 1)} with the next batch.",
                icon=":material/fact_check:",
            )
        st.button(
            "New conversation",
            type="tertiary",
            icon=":material/restart_alt:",
            on_click=new_conversation,
            disabled=not turns,
        )
    if not turns and not store.sources():
        st.info(
            "Your library is empty: add recordings and documents first.",
            icon=":material/library_books:",
        )
        st.page_link("app_pages/ingest.py", label="Go to Ingest", icon=":material/upload_file:")
    for i, turn in enumerate(turns):
        show_turn(i, turn)

with sources_panel:
    st.subheader("Sources")
    if not turns:
        st.caption(
            "The passages each answer is based on appear here, with their file and page or time."
        )
    else:
        turn = turns[selected]
        st.caption(f"For: {safe_md(turn.question)}")
        if not turn.sources:
            st.caption("Nothing in your library matched this question.")
        for s in turn.cited:
            with st.container(gap="xsmall"):
                st.badge(f"[{s.number:d}] {safe_md(s.citation)}", color="gray")
                if s.context:
                    st.caption(safe_md(s.context))
                st.markdown(safe_md(s.text))
        others = [s for s in turn.sources if not s.cited]
        if others:
            label = "Also given to the model" if turn.cited else "Given to the model (none cited)"
            with st.expander(f"{label}: {len(others)}"):
                for s in others:
                    st.badge(f"[{s.number:d}] {safe_md(s.citation)}", color="gray")
                    st.markdown(safe_md(s.text))

question = st.chat_input(
    "Ask about your documents and recordings",
    key="ask_input",
    width=READING_WIDTH,
    submit_mode="disable",
)
pending = st.session_state.get("ask_pending")
if question is None and pending and backend().status().ready:
    question, st.session_state.ask_pending = pending, None
elif question and pending:
    st.session_state.ask_pending = None  # a new question replaces the waiting one

if question:
    with conversation:
        with st.chat_message("user", avatar=USER):
            st.markdown(safe_md(question))
        with st.chat_message("assistant", avatar=ASSISTANT):
            status = st.status(":shimmer[Searching your library]", type="compact")
            reasoning_slot = status.empty()
            answer_slot = st.empty()
            stream = Stream(status, reasoning_slot, answer_slot)
            try:
                answer = backend().ask(
                    question, thinking=st.session_state.ask_thinking, on_delta=stream
                )
            except PhaseBusy:
                status.update(label="Waiting for Q&A", state="complete")
                st.session_state.ask_pending = question
            else:
                stream.show()
                turn = turn_from_answer(answer)
                turn.thinking = st.session_state.ask_thinking
                turns.append(turn)
                st.session_state.ask_selected = len(turns) - 1
                st.rerun()

if waiting := st.session_state.get("ask_pending"):
    with conversation:
        if not question:
            with st.chat_message("user", avatar=USER):
                st.markdown(safe_md(str(waiting)))
        st.info(
            f"{safe_md(backend().status().message)}. Your question will be asked as soon as Q&A is back.",
            icon=":material/hourglass_top:",
        )
