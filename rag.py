import os

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()

DOCUMENTS_FOLDER = "documents"
DATA_FILE = os.path.join("data", "college_info.txt")
CHUNK_SIZE = 1000
EMBED_MODEL = "gemini-embedding-001"


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")
    return genai.Client(api_key=api_key)


def _chunk_text(text: str, source: str, chunks: list[dict]) -> None:
    for i in range(0, len(text), CHUNK_SIZE):
        chunk = text[i : i + CHUNK_SIZE]
        if chunk.strip():
            chunks.append({"text": chunk, "source": source})


def _load_chunks() -> list[dict]:
    chunks: list[dict] = []

    if os.path.isdir(DOCUMENTS_FOLDER):
        for filename in sorted(os.listdir(DOCUMENTS_FOLDER)):
            if not filename.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(DOCUMENTS_FOLDER, filename)
            reader = PdfReader(pdf_path)
            full_text = ""

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

            _chunk_text(full_text, filename, chunks)

    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, encoding="utf-8") as file:
            _chunk_text(file.read(), "college_info.txt", chunks)

    if not chunks:
        raise ValueError("No documents found. Add PDFs to documents/ or data/college_info.txt.")

    return chunks


class RAGIndex:
    def __init__(self) -> None:
        self.client = _get_client()
        self.chunks = _load_chunks()
        texts = [chunk["text"] for chunk in self.chunks]

        result = self.client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )

        embeddings = np.array(
            [embedding.values for embedding in result.embeddings],
            dtype="float32",
        )

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def search(self, question: str, number_of_results: int = 3) -> list[dict]:
        query_result = self.client.models.embed_content(
            model=EMBED_MODEL,
            contents=question,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )

        query_embedding = np.array(
            [query_result.embeddings[0].values],
            dtype="float32",
        )

        _, indices = self.index.search(query_embedding, number_of_results)

        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])

        return results


def search_documents(question: str, number_of_results: int = 3) -> list[dict]:
    index = get_rag_index()
    return index.search(question, number_of_results)


_RAG_INDEX: RAGIndex | None = None


def get_rag_index() -> RAGIndex:
    global _RAG_INDEX
    if _RAG_INDEX is None:
        _RAG_INDEX = RAGIndex()
    return _RAG_INDEX


if __name__ == "__main__":
    rag = get_rag_index()
    sample = "What information is available about scholarships?"
    hits = rag.search(sample)

    print(f"Indexed {len(rag.chunks)} chunks\n")
    print(f"Query: {sample}\n")

    for hit in hits:
        print("SOURCE:", hit["source"])
        print(hit["text"][:300], "...\n")
