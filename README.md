# granit

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Private, fully local document & meeting intelligence on Apple Silicon. IBM Granite models (speech, Docling,
vision, 8B LLM, embeddings, reranker, Guardian) on MLX, with hybrid search, cited answers and a Streamlit UI.

> **Status:** 1.0. All v1 milestones (M0–M7) are built: transcription, document conversion, form extraction,
> hybrid search, cited answers, meeting summaries, the Streamlit app and the evaluation harness. Both evaluation sets
> pass the 1.0 quality criteria ([PLAN.md](PLAN.md) §4.9, with the private set's goals for 1.x). See PLAN.md for the full design.

## What it does (v1)

1. **Meeting / call transcripts**: audio file → transcript → summary, decisions, action items.
2. **Document Q&A**: PDFs and scans → Markdown with accurate tables and chart data → answers with citations.
3. **Form / invoice extraction**: document image + JSON Schema → validated JSON.
4. **Cross-source questions** over everything ingested.

Everything runs locally. The only network traffic is the one-time model download from Hugging Face.

## Requirements

- Apple Silicon Mac with **32 GB** of memory (developed on an M2 Max), macOS 26
- [uv](https://docs.astral.sh/uv/)
- About 30 GB of free disk space for model weights

## Models & licenses

Model weights are **not** part of this repository. They are downloaded from Hugging Face at pinned revisions.

| Model | Role | License | Source |
|---|---|---|---|
| Granite Speech 5.0 470M TurboCTC | Audio → transcript | Apache-2.0 | [ibm-granite/granite-speech-5.0-470m-turboctc](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) |
| Silero VAD v6 | Splits audio at pauses | MIT | [mlx-community/silero-vad-v6](https://huggingface.co/mlx-community/silero-vad-v6) |
| Granite-Docling 258M (MLX) | Page image → Markdown | Apache-2.0 | [ibm-granite/granite-docling-258M-mlx](https://huggingface.co/ibm-granite/granite-docling-258M-mlx) |
| Granite Vision 4.1 4B | Tables, charts, forms → HTML / CSV / JSON | Apache-2.0 | [ibm-granite/granite-vision-4.1-4b](https://huggingface.co/ibm-granite/granite-vision-4.1-4b) |
| Granite 4.2 8B (q8 MLX) | Reasoning, RAG, summaries | Apache-2.0 | [ibm-granite/granite-4.2-8b-q8-mlx](https://huggingface.co/ibm-granite/granite-4.2-8b-q8-mlx) |
| Granite Embedding English R2 | Semantic search vectors | Apache-2.0 | [ibm-granite/granite-embedding-english-r2](https://huggingface.co/ibm-granite/granite-embedding-english-r2) |
| Granite Embedding Reranker English R2 | Reranks search results | Apache-2.0 | [ibm-granite/granite-embedding-reranker-english-r2](https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2) |
| Granite Guardian 4.1 8B | Judges answer groundedness / relevance (converted to 8-bit MLX locally) | Apache-2.0 | [ibm-granite/granite-guardian-4.1-8b](https://huggingface.co/ibm-granite/granite-guardian-4.1-8b) |

## License

granit is licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE).
