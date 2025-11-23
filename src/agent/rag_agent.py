import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from urllib.parse import urljoin, urlparse
from collections import deque
import fitz  # PyMuPDF


class FAURAGAgent:
    """Base FAU RAG Agent: Excel search + bi-encoder."""

    def __init__(self, excel_file):
        self.df = pd.read_excel(excel_file).fillna('')

        print("Loading bi-encoder model…")
        self.bi_encoder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )

        # Combine title + description for semantic signal
        self.corpus = (
            self.df.iloc[:, 1].astype(str) + " " + self.df.iloc[:, 2].astype(str)
        ).tolist()

        print("Encoding corpus (bi-encoder)…")
        self.corpus_embs = self.bi_encoder.encode(
            self.corpus, convert_to_tensor=True, batch_size=64
        )
        print("Done.")

    def top_descriptions(self, query, k=5):
        """Retrieve top-k Excel links using bi-encoder similarity."""
        q_emb = self.bi_encoder.encode(query, convert_to_tensor=True)
        sims = torch.nn.functional.cosine_similarity(q_emb, self.corpus_embs)
        top_k = min(k, len(sims))
        top_ids = torch.topk(sims, k=top_k).indices.tolist()
        return self.df.iloc[top_ids]


class FAURAGAgentEnhanced(FAURAGAgent):
    """Enhanced FAU RAG Agent with query-specific recursive scraping + PDFs."""

    def __init__(self, excel_file):
        super().__init__(excel_file)

    def fetch_page(self, url):
        """Fetch HTML or PDF and return text + embedded links."""
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla"})
            content_type = r.headers.get("Content-Type", "")
            if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
                doc = fitz.open(stream=r.content, filetype="pdf")
                text = " ".join([page.get_text() for page in doc])
                return text, []
            else:
                soup = BeautifulSoup(r.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
                text = re.sub(r"\s+", " ", text)
                links = [urljoin(url, a['href']) for a in soup.find_all("a", href=True)]
                return text, links
        except Exception as e:
            print(f"❌ Failed to fetch {url}: {e}")
            return "", []

    def is_valid_url(self, url):
        parsed = urlparse(url)
        return parsed.scheme in ["http", "https"]

    def chunk(self, text, size=350):
        """Split text into word chunks."""
        words = text.split()
        return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]

    def crawl_links_depth3(self, urls, max_depth=1):
        """Crawl top URLs recursively up to depth 3 and gather text chunks."""
        visited = set()
        queue = deque([(url, 0) for url in urls])
        all_texts = []
        all_sources = []

        while queue:
            url, depth = queue.popleft()
            if url in visited or depth > max_depth:
                continue
            visited.add(url)

            print(f"Scraping depth {depth}: {url}")
            text, links = self.fetch_page(url)
            if text:
                chunks = self.chunk(text)
                all_texts.extend(chunks)
                all_sources.extend([url] * len(chunks))

            # Enqueue only links from top 5 results (avoid unrelated sites)
            if depth < max_depth:
                for link in links:
                    if link not in visited and self.is_valid_url(link):
                        queue.append((link, depth + 1))

        return all_texts, all_sources

    def build_rag_db(self, query, top_k_links=5):
        """Build query-specific RAG database from top 5 Excel links and depth-3 crawling."""
        top_df = self.top_descriptions(query, k=top_k_links)
        urls = top_df.iloc[:, 0].tolist()
        texts, sources = self.crawl_links_depth3(urls, max_depth=1)
        if not texts:
            return []

        embs = self.bi_encoder.encode(texts, convert_to_numpy=True, batch_size=32)
        rag_db = [{"text": t, "emb": e, "source": s} for t, e, s in zip(texts, embs, sources)]
        return rag_db

    def query_rag(self, query, top_k_chunks=5):
        """Query temporary RAG database built only from top 5 results + depth-3 crawling."""
        rag_db = self.build_rag_db(query, top_k_links=5)
        if not rag_db:
            return "❌ Could not extract enough content from the top links."

        q_emb = self.bi_encoder.encode([query], convert_to_numpy=True)
        embs = np.array([entry["emb"] for entry in rag_db])
        sims = np.dot(embs, q_emb.T).squeeze()
        top_idx = np.argsort(sims)[-top_k_chunks:][::-1]

        results = [rag_db[i] for i in top_idx]
        answer_text = "\n".join([res["text"] for res in results])
        sources = list({res["source"] for res in results})

        return f"{answer_text}\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)


if __name__ == "__main__":
    excel_file = "/home/ubuntu/projects/FAUgen-AI/src/excel_processing/aces_metadata.xlsx"
    agent = FAURAGAgentEnhanced(excel_file)

    while True:
        q = input("\nAsk a question about FAU (or 'quit'):\n> ").strip()
        if q.lower() in ["quit", "exit"]:
            break
        print("\n🔄 Processing your query…\n")
        answer = agent.query_rag(q)
        print("\n✅ Answer:\n")
        print(answer)
