# granit

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Private, fully local document and meeting intelligence on Apple Silicon. IBM Granite models run on MLX to transcribe
recordings, convert documents, extract forms and answer questions with citations, all from a CLI or a Streamlit app.

Everything runs on your Mac. The only network traffic is the one-time model download from Hugging Face.

## Features

- **Meetings:** audio → transcript → summary, decisions and action items.
- **Speakers:** who spoke when. Name the speakers you recognize, and summaries, search and Ask use those names.
  Unnamed speakers stay unlabelled.
- **Documents:** PDFs and scans → Markdown with accurate tables and chart data.
- **Forms:** document image + JSON Schema → validated JSON.
- **Search and Ask:** hybrid search and cited answers across everything you've ingested.
- **Answer checks:** Granite Guardian checks that answers are supported by their sources, and checks meeting summaries
  against requirements you write.

## Requirements

- Apple Silicon Mac with 32 GB of memory (developed on an M2 Max), macOS 26
- [uv](https://docs.astral.sh/uv/)
- About 30 GB of free disk space for model weights (about 47 GB during setup, until the local builds finish)

## Quick start

```bash
uv sync                                                     # install
uv run granit models download                               # pinned runtime models
uv run granit models download guardian-source diarization-source
uv run granit models convert --delete-source                # build Guardian (8-bit) and diarization (fp32) locally
uv run granit models smoke                                  # check each model loads
uv run granit ui                                            # open the app
```

## Usage

| Command | What it does |
|---|---|
| `granit ui` | The app: Ingest, Library, Ask and Extract pages |
| `granit ingest FILE…` | Add files to the library |
| `granit sources` | List the library |
| `granit search "question"` | Cited search hits (`--mode bm25\|vectors\|hybrid\|hybrid+rerank`) |
| `granit ask "question"` | Cited answer (`--thinking off\|low\|on`) |
| `granit meeting SOURCE` | Meeting summary as JSON |
| `granit speakers SOURCE [1=Priya …]` | List or name a recording's speakers |
| `granit verify` | Run Guardian checks on answers and summaries that haven't been checked yet |
| `granit transcribe FILE…` | Audio → timestamped segments |
| `granit convert FILE…` | PDF or image → Markdown and chart data |
| `granit extract FILE --schema S.json` | Form fields → validated JSON |

Prefix each command with `uv run`. Add `--help` to any command for its options.

## Models and licenses

Model weights are **not** part of this repository. They are downloaded from Hugging Face at pinned revisions.

| Model | Role | License | Source |
|---|---|---|---|
| Granite Speech 5.0 470M TurboCTC | Audio → transcript | Apache-2.0 | [ibm-granite/granite-speech-5.0-470m-turboctc](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) |
| Silero VAD v6 | Splits audio at pauses | MIT | [mlx-community/silero-vad-v6](https://huggingface.co/mlx-community/silero-vad-v6) |
| Granite-Docling 258M (MLX) | Page image → Markdown | Apache-2.0 | [ibm-granite/granite-docling-258M-mlx](https://huggingface.co/ibm-granite/granite-docling-258M-mlx) |
| Granite Vision 4.1 4B | Tables, charts, forms → HTML, CSV, JSON | Apache-2.0 | [ibm-granite/granite-vision-4.1-4b](https://huggingface.co/ibm-granite/granite-vision-4.1-4b) |
| Granite 4.2 8B (q8 MLX) | Reasoning, RAG, summaries | Apache-2.0 | [ibm-granite/granite-4.2-8b-q8-mlx](https://huggingface.co/ibm-granite/granite-4.2-8b-q8-mlx) |
| Granite Embedding English R2 | Search vectors | Apache-2.0 | [ibm-granite/granite-embedding-english-r2](https://huggingface.co/ibm-granite/granite-embedding-english-r2) |
| Granite Embedding Reranker English R2 | Reranks search results | Apache-2.0 | [ibm-granite/granite-embedding-reranker-english-r2](https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2) |
| Granite Guardian 4.1 8B ¹ | Checks answers are grounded and relevant | Apache-2.0 | [ibm-granite/granite-guardian-4.1-8b](https://huggingface.co/ibm-granite/granite-guardian-4.1-8b) |
| Nemotron 3 Diarization ¹ | Who spoke when | [OpenMDW-1.1](https://openmdw.ai/license/1-1/) | [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) |

¹ Converted locally by `granit models convert`: Guardian to 8-bit MLX, diarization to fp32 MLX.

## Status

Version 1.3. All v1 milestones (M0–M7) plus M8 (answer checks), M9 (speakers) and M10 (speaker names in search and
Ask) are built. Both evaluation sets meet the 1.0 quality criteria ([PLAN.md](PLAN.md) §4.9).

## Documentation

- [PLAN.md](PLAN.md): design, milestones and decisions
- [CONTRIBUTING.md](CONTRIBUTING.md): development setup, tests and licensing rules

## License

granit is licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE).
