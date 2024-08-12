from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import pinecone
import tiktoken
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set the environment variables and API keys
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

# Initialize Pinecone and model
pc = pinecone.Pinecone(api_key=os.environ['PINECONE_API_KEY'])
index_name = "world-history"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension for SentenceTransformer
        metric='cosine'
    )

index = pc.Index(index_name)

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to get embeddings
def get_embeddings(texts):
    return model.encode(texts).tolist()

# Load and split the PDF text
pdf_loader = PyPDFLoader("01. World History. Cultures, States, and Societies to 1500 author Eugene Berger, George L. Israel, Charlotte Miller, Brian Parkinson, Andrew Reeves and Nadejda Williams.pdf")
pdf_text = pdf_loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
    length_function=lambda text: len(tiktoken.get_encoding('p50k_base').encode(text)),
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_text(pdf_text)
embeddings = get_embeddings(chunks)

# Upsert vectors into Pinecone
vectors_to_upsert = [(f"id_{i}", embedding, {"text": text}) for i, (embedding, text) in enumerate(zip(embeddings, chunks))]
index.upsert(vectors=vectors_to_upsert)

# Function to check Pinecone content (for debugging)
def check_pinecone_content():
    try:
        stats = index.describe_index_stats()
        print(f"Total vectors in index: {stats['total_vector_count']}")
        sample_ids = [f'id_{i}' for i in range(min(5, stats['total_vector_count']))]
        fetch_response = index.fetch(ids=sample_ids)
        for id, vector in fetch_response['vectors'].items():
            print(f"\nVector ID: {id}")
            print(f"Metadata: {vector.get('metadata', 'No metadata')}")
            print(f"Values (first 5): {vector['values'][:5] if vector['values'] else 'No values'}")
        return "Pinecone index content check completed"
    except Exception as e:
        return f"Pinecone index content check failed: {str(e)}"

# Call the function to check Pinecone index
print(check_pinecone_content())

# Root endpoint
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Ask endpoint
@app.post("/ask")
async def ask(request: Request, question: str = Form(...)):
    try:
        query_embedding = get_embeddings([question])[0]
        print(f"Query Embedding: {query_embedding}")

        top_matches = index.query(vector=query_embedding, top_k=10, include_metadata=True)
        print(f"Top Matches: {top_matches}")

        if not top_matches or not top_matches.get('matches'):
            return templates.TemplateResponse("index.html", {"request": request, "answer": "Sorry, I couldn't find an answer."})

        contexts = [item['metadata']['text'] for item in top_matches['matches']]
        print(f"Contexts: {contexts}")

        augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + question
        print(f"Augmented Query: {augmented_query}")

        response = genai.generate_text(
            prompt=f"You are an expert on the book titled 'World History'. Answer questions based only on the provided context from this book.\n\n{augmented_query}",
            safety_settings=[{"category": "HARM_CATEGORY_DEROGATORY", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}],
            generation_config={"temperature": 0.7, "max_output_tokens": 150}
        )
        print(f"Generated Response: {response}")

        return templates.TemplateResponse("index.html", {"request": request, "answer": response.text})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "answer": f"An error occurred: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
