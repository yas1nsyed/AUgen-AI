# 🎓 FAUgen AI — FAU Search Assistant

An AI-powered Retrieval-Augmented Generation (RAG) assistant for [Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)](https://www.fau.eu/). FAUgen AI lets you ask natural-language questions about FAU websites, programs, research, and campus life, and get accurate, source-cited answers powered by semantic search and large language models.

---

## ✨ Features

- **Semantic Search** — Uses FAISS with HNSW indexing and multilingual sentence embeddings to retrieve the most relevant FAU web pages for any query.
- **RAG Pipeline** — Fetches and chunks live page content, then re-ranks chunks by cosine similarity before passing to an LLM.
- **Multi-LLM Support** — Works with Google Gemini (default), OpenAI GPT-4, and local Ollama models.
- **Gradio Web UI** — A clean chat interface (`app.py`) for interactive Q&A.
- **CLI Interface** — A terminal-based interactive query loop (`main.py`).
- **Data Pipeline Tools** — Utilities to scrape FAU sitemaps, split URLs, enrich metadata with titles/descriptions, and build the vector index.

---

## 🗂️ Project Structure

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

## 🚀 Getting Started

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
pip install -e .
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here   # optional
```

---

## 🗄️ Building the Vector Index

Before running the assistant, you need to build the FAISS embedding index from FAU page metadata.

**Step 1 — Collect URLs from a sitemap:**
```bash
python src/excel_processing/sitemap2excel.py
```

**Step 2 — Enrich with titles & descriptions:**
```bash
python src/excel_processing/title_descriptions.py
```

**Step 3 — Build the FAISS index:**

In `src/agent/embeddings.py`, uncomment the last two lines and run:
```bash
python src/agent/embeddings.py
```

This creates `data/embeddings.index` and `data/metadata.parquet`.

---

## 💬 Running the Application

### Web UI (Gradio)

```bash
python app.py
```

Opens a chat interface in your browser at `http://localhost:7860`.

### CLI Mode

```bash
python main.py
```

Follow the prompts to select an LLM provider and start querying.

---

## 🔧 LLM Providers

| Provider | Default Model | Environment Variable |
|---|---|---|
| **Gemini** (default) | `gemini-2.5-flash` | `GOOGLE_API_KEY` |
| **OpenAI** | `gpt-4` | `OPENAI_API_KEY` |
| **Ollama** | `llama2` | *(local, no key needed)* |

---

## 🧠 How It Works

1. **Indexing** — FAU page URLs, titles, and descriptions are embedded using `paraphrase-multilingual-mpnet-base-v2` and stored in a FAISS HNSW index.
2. **Retrieval** — A user query is embedded and the top-K most similar documents are retrieved from the index.
3. **Augmentation** — The retrieved URLs are scraped live; content is chunked and re-ranked by cosine similarity to the query.
4. **Generation** — The top chunks are injected as context into an LLM prompt, which produces a grounded, source-cited answer.

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `sentence-transformers` | Multilingual embeddings |
| `faiss-cpu` | Fast approximate nearest-neighbor search |
| `gradio` | Web chat UI |
| `langchain` / `langchain-openai` | LLM integrations |
| `google-generativeai` | Gemini API |
| `beautifulsoup4` / `requests` | Web scraping |
| `pandas` / `openpyxl` | Data processing |

---

## 📄 License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
An AI-Based solution for all FAU queries
