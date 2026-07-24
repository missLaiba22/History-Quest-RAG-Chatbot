import time

import pinecone
from pinecone import ServerlessSpec
import tiktoken
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

from embeddings import embed_texts, VECTOR_DIMENSION

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError(
        "PINECONE_API_KEY is not set. Copy .env.example to .env and fill it in."
    )

# Serverless region config -- configurable rather than assumed, since which
# cloud/region you have access to depends on your Pinecone account/plan.
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

PDF_PATH = os.getenv(
    "PDF_PATH",
    "01. World History. Cultures, States, and Societies to 1500 author Eugene Berger, "
    "George L. Israel, Charlotte Miller, Brian Parkinson, Andrew Reeves and Nadejda Williams.pdf",
)

INDEX_NAME = "world-history"
UPSERT_BATCH_SIZE = 100  # Pinecone recommends batching upserts rather than one giant call

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# --- Recreate the index from scratch ---
# We deliberately DELETE and rebuild rather than upsert-in-place. The index may
# already contain vectors from the old, broken embedding pipeline (random,
# untrained projections) -- those vectors live in a meaningless space and are
# not safely comparable to correctly-embedded vectors. A partial upsert with
# reused ids would only overwrite as many old vectors as there are new chunks;
# if chunk count/order ever differs between runs, stale vectors are left
# behind silently. Deleting the whole index guarantees a clean, trustworthy
# rebuild every time this script is run.
if INDEX_NAME in pc.list_indexes().names():
    print(f"Deleting existing index '{INDEX_NAME}' for a clean rebuild...")
    pc.delete_index(INDEX_NAME)

print(f"Creating index '{INDEX_NAME}' ({VECTOR_DIMENSION}-dim, cosine)...")
pc.create_index(
    name=INDEX_NAME,
    dimension=VECTOR_DIMENSION,
    metric="cosine",
    spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
)

# Index creation is asynchronous -- wait until Pinecone reports it's ready
# before writing to it.
while not pc.describe_index(INDEX_NAME).status["ready"]:
    print("Waiting for index to be ready...")
    time.sleep(1)

index = pc.Index(INDEX_NAME)

pdf_loader = PyPDFLoader(PDF_PATH)
pdf_documents = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    length_function=lambda text: len(tiktoken.get_encoding('p50k_base').encode(text)),
    separators=["\n\n", "\n", " ", ""]
)

pdf_text = "\n".join([doc.page_content for doc in pdf_documents])
chunks = text_splitter.split_text(pdf_text)
print(f"Split PDF into {len(chunks)} chunks.")

chunk_ids = [f"id_{i}" for i in range(len(chunks))]

print("Generating embeddings...")
embeddings = embed_texts(chunks)

vectors_to_upsert = [
    (chunk_id, embedding, {"text": text})
    for chunk_id, embedding, text in zip(chunk_ids, embeddings, chunks)
]

print(f"Upserting {len(vectors_to_upsert)} vectors in batches of {UPSERT_BATCH_SIZE}...")
for i in range(0, len(vectors_to_upsert), UPSERT_BATCH_SIZE):
    batch = vectors_to_upsert[i : i + UPSERT_BATCH_SIZE]
    index.upsert(vectors=batch)
    print(f"  upserted {min(i + UPSERT_BATCH_SIZE, len(vectors_to_upsert))}/{len(vectors_to_upsert)}")

print("Data processing and indexing complete.")