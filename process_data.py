import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
import pinecone
import tiktoken
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set the environment variables and API keys
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone and model
pc = pinecone.Pinecone(api_key=os.environ['PINECONE_API_KEY'])
index_name = "world-history"

VECTOR_DIMENSION = 384

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=VECTOR_DIMENSION,
        metric='cosine'
    )

index = pc.Index(index_name)

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModel.from_pretrained(model_name)

def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = model(inputs)
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    embeddings = tf.keras.layers.Dense(VECTOR_DIMENSION)(embeddings)
    return embeddings.numpy().tolist()

pdf_loader = PyPDFLoader("01. World History. Cultures, States, and Societies to 1500 author Eugene Berger, George L. Israel, Charlotte Miller, Brian Parkinson, Andrew Reeves and Nadejda Williams.pdf")
pdf_documents = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    length_function=lambda text: len(tiktoken.get_encoding('p50k_base').encode(text)),
    separators=["\n\n", "\n", " ", ""]
)

pdf_text = "\n".join([doc.page_content for doc in pdf_documents])
chunks = text_splitter.split_text(pdf_text)

chunk_ids = [f"id_{i}" for i in range(len(chunks))]
embeddings = get_embeddings(chunks)

vectors_to_upsert = [(chunk_id, embedding, {"text": text}) for chunk_id, embedding, text in zip(chunk_ids, embeddings, chunks)]
index.upsert(vectors=vectors_to_upsert)

print("Data processing and indexing complete.")