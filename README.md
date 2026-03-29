# Multi-Source Research Aggregator

A Gradio-based web app that scrapes content from multiple URLs, deduplicates it, builds a semantic search index, and lets you ask natural language questions across all sources at once. Powered by LLaMA 3.3 70B via Groq.

---

## What It Does

1. You provide one or more URLs (comma-separated) and a question.
2. The app fetches and extracts clean text from each URL using Trafilatura (with a BeautifulSoup fallback for sites Trafilatura can't handle).
3. All extracted text is chunked and embedded using a local HuggingFace sentence transformer.
4. Near-duplicate chunks are removed via cosine similarity before indexing.
5. The cleaned, deduplicated chunks are stored in a FAISS vector index.
6. Your question is answered by retrieving the top 4 most relevant chunks and passing them to LLaMA 3.3 70B.
7. The answer and up to 2 source excerpts are displayed in the UI.

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| UI Framework | Gradio |
| Primary Web Scraper | Trafilatura |
| Fallback Web Scraper | Requests + BeautifulSoup4 |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Deduplication | Cosine similarity (NumPy) |
| Vector Store | FAISS |
| LLM | Groq `llama-3.3-70b-versatile` |
| RAG Chain | LangChain `RetrievalQA` |

---

## Project Structure

```
.
├── app.py          # Main application (all logic lives here)
└── README.md
```

---

## Prerequisites

- Python 3.9+
- A valid [Groq API key](https://console.groq.com/)

---

## Installation

```bash
# Clone the repo
git clone https://github.com/your-username/multi-source-research-aggregator.git
cd multi-source-research-aggregator

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
gradio
requests
beautifulsoup4
trafilatura
numpy
langchain
langchain-community
langchain-groq
faiss-cpu
sentence-transformers
```

---

## Configuration

The Groq API key is hardcoded in the script. Before sharing or deploying, replace it with an environment variable:

```python
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

Then set it before running:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

---

## Running the App

```bash
python app.py
```

Gradio will start a local server at `http://localhost:7860` and print a public share link (because `share=True` is set).

---

## How to Use

1. Open the app in your browser.
2. In the "URLs" field, enter one or more URLs separated by commas:
   ```
   https://en.wikipedia.org/wiki/Artificial_intelligence, https://example.com/article
   ```
3. In the "Question" field, type your question (default is "Summarize all sources").
4. Click "Run".
5. The answer appears in the "Answer" box, and relevant source excerpts appear in the "Sources" box.

---

## Key Implementation Details

### Two-Layer Web Scraping

The `fetch_text()` function first tries Trafilatura, which is purpose-built for extracting clean article text and handles most news/blog sites well. If Trafilatura returns less than 200 characters, it falls back to a raw Requests + BeautifulSoup scrape that strips `<script>`, `<style>`, and `<noscript>` tags before extracting visible text.

### Chunk Deduplication

After splitting text into 500-character chunks (100-character overlap), the `deduplicate()` function embeds all chunks and computes pairwise cosine similarity. Any chunk with a similarity score above 0.85 against an already-kept chunk is dropped. This prevents the LLM from receiving redundant context, which wastes the context window and can skew answers.

### Text Splitting

`RecursiveCharacterTextSplitter` splits on paragraph breaks, then sentences, then words — preserving as much semantic coherence as possible within each chunk.

### Retrieval

At query time, the top 4 most semantically similar chunks are retrieved from FAISS and injected into the LLM prompt as context.

### LLM Settings

`temperature=0.2` keeps answers factual and consistent. The model is re-instantiated on every `process()` call (stateless by design — no session memory).

---

## Validation & Error Handling

The `process()` function validates at each stage and raises descriptive errors if:
- No URLs are provided
- All URLs fail to fetch
- Extracted content is under 200 characters
- Fewer than 2 chunks remain after splitting or deduplication

Errors are surfaced directly in the "Answer" output box in the UI.

---

## Limitations

- The FAISS index is rebuilt from scratch on every run — there is no caching between queries.
- JavaScript-heavy single-page apps (React, Angular, etc.) may return little or no content since both scrapers work on the raw HTML response.
- No conversation memory — each click of "Run" is a fresh, independent query.
- The hardcoded API key must be replaced with an environment variable before any deployment.
- `share=True` in `app.launch()` creates a public Gradio tunnel. Remove it or set `share=False` for private/local use.

---

## License

MIT
