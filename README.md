# AUgen AI — FAU Search Assistant

An AI-powered Retrieval-Augmented Generation (RAG) assistant for [Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)](https://www.fau.eu/). AUgen AI lets you ask natural-language questions about FAU websites, programs, research, and campus life, and get accurate, source-cited answers powered by semantic search and large language  models.
Currently as a proof of concept, this project only supports queries for the course of Electromolbility-ACES but is built in a scalable manner to eventually cover the entire FAU.

---

## Features

- **Semantic Search** — Uses FAISS with HNSW indexing and multilingual sentence embeddings to retrieve the most relevant FAU web pages for any query.
- **RAG Pipeline** — Fetches and chunks live page content, then re-ranks chunks by cosine similarity before passing to an LLM.
- **Multi-LLM Support** — Works with Google Gemini (default), OpenAI GPT-4, and local Ollama models.
- **Gradio Web UI** — A clean chat interface (`app.py`) for interactive Q&A.
- **CLI Interface** — A terminal-based interactive query loop (`main.py`).
- **Data Pipeline Tools** — Utilities to scrape FAU sitemaps, split URLs, enrich metadata with titles/descriptions, and build the vector index.

---

## Project Structure

```
FAUgen-AI/
├── app.py                          # Gradio web UI (chat interface)
├── main.py                         # CLI interactive RAG query loop
├── pyproject.toml                  # Project dependencies (uv/pip)
├── data/
│   └── embeddings.index            # FAISS vector index
├── src/
│   ├── llm_client.py               # LLM factory (Gemini / OpenAI / Ollama)
│   ├── agent/
│   │   ├── rag_agent.py            # Core RAG agent (retrieval + scraping)
│   │   ├── embeddings.py           # FAISS index builder
│   │   ├── agent.py                # Agent utilities
│   │   └── ex_main.py              # Example usage
│   ├── excel_processing/
│   │   ├── sitemap2excel.py        # Extract URLs from XML sitemaps → Excel
│   │   ├── split_urls.py           # Split/filter URL lists
│   │   ├── title_descriptions.py   # Scrape page titles & meta descriptions
│   │   └── website2excel.py        # Full site scrape → Excel
│   └── tools/
│       └── scraper.py              # General-purpose web scraper
```

---

##  Getting Started

### Prerequisites

- Python ≥ 3.13
- [uv](https://github.com/astral-sh/uv) (recommended) **or** pip
- API key for at least one LLM provider (Gemini recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/FAUgen-AI.git
cd FAUgen-AI
```

### 2. Install Dependencies

```bash
# With uv (recommended)
uv sync

# Or with pip
pip install -e 
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here   # optional
```


---

##  Running the Application

### Web UI (Gradio)

```bash
python app.py
```

Opens a chat interface in your browser at localhost

### CLI Mode

```bash
python main.py
```

Follow the prompts to select an LLM provider and start querying.

---

##  LLM Providers

Currently the following LLMs are supported by AUgen-AI. More models can also be added.

| Provider | Default Model | Environment Variable |
|---|---|---|
| **Gemini** (default) | `gemini-2.5-flash` | `GOOGLE_API_KEY` |
| **OpenAI** | `gpt-4` | `OPENAI_API_KEY` |
| **Ollama** | `llama2` | *(local, no key needed)* |

 
---

##  How It Works

1. **Indexing** — FAU page URLs, titles, and descriptions are embedded using `paraphrase-multilingual-mpnet-base-v2` and stored in a FAISS HNSW index.
2. **Retrieval** — A user query is embedded and the top-K most similar documents are retrieved from the index.
3. **Augmentation** — The retrieved URLs are scraped live; content is chunked and re-ranked by cosine similarity to the query.
4. **Generation** — The top chunks are injected as context into an LLM prompt, which produces a grounded, source-cited answer.

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.

