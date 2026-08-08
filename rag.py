import os
import numpy as np
import faiss

from pypdf import PdfReader
from google import genai
from google.genai import types
from dotenv import load_dotenv


# Load API key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")


# Gemini client
client = genai.Client(api_key=api_key)


# PDF folder
DOCUMENTS_FOLDER = "documents"


# Store our text chunks
chunks = []


# ==========================================
# READ PDF FILES
# ==========================================

for filename in os.listdir(DOCUMENTS_FOLDER):

    if filename.lower().endswith(".pdf"):

        pdf_path = os.path.join(
            DOCUMENTS_FOLDER,
            filename
        )

        print(f"Reading: {filename}")

        reader = PdfReader(pdf_path)

        full_text = ""

        for page in reader.pages:

            text = page.extract_text()

            if text:
                full_text += text + "\n"


        # ==================================
        # SPLIT TEXT INTO CHUNKS
        # ==================================

        chunk_size = 1000

        for i in range(0, len(full_text), chunk_size):

            chunk = full_text[i:i + chunk_size]

            if chunk.strip():

                chunks.append({
                    "text": chunk,
                    "source": filename
                })


print(f"\nTotal chunks created: {len(chunks)}")


# ==========================================
# CREATE EMBEDDINGS
# ==========================================

texts = [chunk["text"] for chunk in chunks]


result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=texts,
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT"
    )
)


embeddings = np.array(
    [embedding.values for embedding in result.embeddings],
    dtype="float32"
)


print("Embeddings created!")


# ==========================================
# CREATE FAISS INDEX
# ==========================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


print(f"FAISS index created with {index.ntotal} vectors")


# ==========================================
# SEARCH FUNCTION
# ==========================================

def search_documents(question, number_of_results=3):

    query_result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )


    query_embedding = np.array(
        [query_result.embeddings[0].values],
        dtype="float32"
    )


    distances, indices = index.search(
        query_embedding,
        number_of_results
    )


    results = []

    for idx in indices[0]:

        if idx < len(chunks):

            results.append(chunks[idx])


    return results
# ==========================================
# TEST SEARCH
# ==========================================

question = "What information is available about scholarships?"

results = search_documents(question)

print("\nRelevant information:\n")

for result in results:

    print("SOURCE:", result["source"])
    print(result["text"])
    print("-" * 50)