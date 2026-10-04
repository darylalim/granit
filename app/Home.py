"""granit: local document and meeting intelligence (PLAN.md §1, §2). Start it with ``uv run granit ui``.

The entrypoint: page config, navigation, the phase banner on every page. The first visit starts the backend (search models,
then Q&A) in the background; the banner shows its progress.
"""

import streamlit as st

from granit.ui.session import backend, phase_banner

st.set_page_config(
    page_title="granit",
    page_icon=":material/landscape:",
    layout="wide",
)

page = st.navigation(
    [
        st.Page("app_pages/ingest.py", title="Ingest", icon=":material/upload_file:"),
        st.Page("app_pages/library.py", title="Library", icon=":material/library_books:"),
        st.Page("app_pages/ask.py", title="Ask", icon=":material/forum:", default=True),
        st.Page("app_pages/extract.py", title="Extract", icon=":material/data_object:"),
    ],
    position="top",  # no sidebar: its ~280 px would push the side panels below on 13–14" screens (§4.8)
)

backend()

phase_banner()
page.run()
