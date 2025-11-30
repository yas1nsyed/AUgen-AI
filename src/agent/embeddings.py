# embeddings.py
import faiss
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

EMBEDDING_DIM = 768
INDEX_PATH = "data/embeddings.index"
METADATA_PATH = "data/metadata.parquet"

class DocumentStore:
    def save(self, df: pd.DataFrame):
        Path(METADATA_PATH).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(METADATA_PATH, index=False)
        print(f"[DocumentStore] Saved metadata → {METADATA_PATH}")

class EmbeddingStore:
    
    def build_and_save(self, embeddings: torch.Tensor):
        Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)

        vecs = embeddings.cpu().numpy().astype("float32")
        faiss.normalize_L2(vecs)

        dim = vecs.shape[1]
        index = faiss.IndexHNSWFlat(dim, 64)
        index.hnsw.efConstruction = 1000

        index.add(vecs)
        faiss.write_index(index, INDEX_PATH)

        print(f"[EmbeddingStore] Saved FAISS index → {INDEX_PATH}")


    @staticmethod
    def chunk_text(text, chunk_size=2000, overlap=250):
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks


    @staticmethod
    def build_embeddings_from_excel(excel_file: str):
        print(f"[embeddings.py] Loading Excel → {excel_file}")
        df = pd.read_excel(excel_file).fillna("")

        model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )

        all_chunks = []
        metadata = []

        for idx, row in df.iterrows():
            doc_id = str(row[0])      # Link or ID in column 1
            full_text = str(row[5])   # Main text in column 6

            chunks = EmbeddingStore.chunk_text(full_text)

            for chunk_id, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata.append({
                    "doc_id": doc_id,
                    "row": idx,
                    "chunk_id": chunk_id,
                    "chunk_text": chunk
                })

        print(f"Total chunks created: {len(all_chunks)}")

        embeddings = model.encode(
            all_chunks,
            convert_to_tensor=True,
            batch_size=256,
            show_progress_bar=True
        )

        # SAVE METADATA (convert list → DataFrame)
        metadata_df = pd.DataFrame(metadata)
        doc_store = DocumentStore()
        doc_store.save(metadata_df)

        # SAVE EMBEDDINGS
        emb_store = EmbeddingStore()
        emb_store.build_and_save(embeddings)

        print("Embedding database built successfully!")


# --- Runner ---
if __name__ == "__main__":
    EmbeddingStore.build_embeddings_from_excel(
        "src/excel_processing/aces_metadata_output.xlsx"
    )
