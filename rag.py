"""
Minimal RAG (retrieval-augmented generation) demo.
Flow: documents -> chunks -> embed & store in Chroma -> retrieve relevant chunks -> ask Claude.
Embeddings run locally/free via Chroma; only the final answer uses the Anthropic API.
"""

from dotenv import load_dotenv
import chromadb
from langchain_anthropic import ChatAnthropic

load_dotenv()  # loads ANTHROPIC_API_KEY from .env

# 1. SOURCE DOCUMENTS ---------------------------------------------------------
# Sample public content so this repo is safe to share. Swap in your own later.
documents = [
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
    "It was completed in 1889 and stands about 330 metres tall. "
    "For 41 years it was the tallest man-made structure in the world.",

    "The Great Barrier Reef is the world's largest coral reef system, off the "
    "coast of Queensland, Australia. It stretches over 2,300 km and is made up "
    "of nearly 3,000 individual reefs.",

    "Photosynthesis is how green plants convert sunlight, water and carbon "
    "dioxide into glucose and oxygen. It mostly happens in the chloroplasts "
    "of plant cells, using the pigment chlorophyll.",
]

# 2. CHUNKING -----------------------------------------------------------------
# Split text into smaller overlapping pieces so retrieval is precise.
def chunk_text(text, size=300, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

all_chunks = []
for doc in documents:
    all_chunks.extend(chunk_text(doc))

# 3. EMBED & STORE ------------------------------------------------------------
# Chroma turns each chunk into a vector (locally, free) and stores it.
client = chromadb.Client()
collection = client.create_collection(name="docs")
collection.add(
    documents=all_chunks,
    ids=[f"chunk-{i}" for i in range(len(all_chunks))],
)

# 4. RETRIEVE -----------------------------------------------------------------
question = "How tall is the Eiffel Tower?"
results = collection.query(query_texts=[question], n_results=3)
retrieved = results["documents"][0]
context = "\n\n".join(retrieved)

# 5. GENERATE -----------------------------------------------------------------
llm = ChatAnthropic(model="claude-sonnet-4-6")
prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}"""
answer = llm.invoke(prompt).content

# 6. SHOW RESULT --------------------------------------------------------------
print("Question:", question)
print("\nAnswer:", answer)
print("\n--- Sources used ---")
for i, chunk in enumerate(retrieved, 1):
    print(f"[{i}] {chunk[:120]}...")