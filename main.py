from google import genai
from google.genai import types
import pinecone
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv

from embeddings import embed_query
from models import QuestionModel

# Load environment variables
load_dotenv()

app = FastAPI()

# Templates live in their own top-level folder, kept separate from static/.
# (Previously templates/ sat inside static/, which is mounted as a public
# static route below -- that meant the raw, unrendered HTML was directly
# downloadable at /static/templates/index.html. Keeping them apart is the
# standard FastAPI layout and avoids that.)
templates = Jinja2Templates(directory="templates")

# Serve static assets (CSS/JS) only.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set the environment variables and API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not PINECONE_API_KEY or not gemini_api_key:
    raise RuntimeError(
        "PINECONE_API_KEY and GEMINI_API_KEY must both be set. "
        "Copy .env.example to .env and fill in real values."
    )

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
gemini_client = genai.Client(api_key=gemini_api_key)

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "world-history"
index = pc.Index(index_name)

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.post("/ask")
async def ask(payload: QuestionModel):
    question = payload.question.strip()

    if not question:
        return JSONResponse(status_code=400, content={"error": "Ask something first."})

    try:
        # Handle common greetings and simple queries
        greetings = ["hi", "hello", "hey", "alright", "thanks", "bye"]
        if question.lower() in greetings:
            responses = {
                "hi": "Hello! How can I assist you with world history today?",
                "hello": "Hi there! What would you like to know about world history?",
                "hey": "Hey! Feel free to ask me anything about world history.",
                "alright": "Got it! If you have any questions about world history, just let me know.",
                "thanks": "You're welcome! If you have more questions, I'm here to help.",
                "bye": "Goodbye! Feel free to return if you have more questions."
            }
            response_text = responses.get(question.lower(), "I'm here if you need any help with world history.")
            return JSONResponse(content={"answer": response_text})

        # Process the question normally
        query_embedding = list(embed_query(question))
        top_matches = index.query(vector=query_embedding, top_k=3, include_metadata=True)

        if not top_matches or not top_matches.get('matches'):
            response_text = "Sorry, I couldn't find an answer."
        else:
            contexts = [item['metadata']['text'] for item in top_matches['matches'][:3]]
            combined_contexts = "\n\n-------\n\n".join(contexts)

            # Shorten the combined contexts if needed
            max_length = 2000  # Adjusted for brevity
            if len(combined_contexts) > max_length:
                combined_contexts = combined_contexts[:max_length] + "..."

            # Use clear instructions to avoid confusion
            augmented_query = f"<CONTEXT>\n{combined_contexts}\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n{question}"

            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=(
                    "You are an expert on the book 'World History: Cultures, States, "
                    "and Societies to 1500.' Answer the following question in a brief "
                    f"and conversational manner.\n\n{augmented_query}"
                ),
                config=types.GenerateContentConfig(temperature=0.5),
            )

            response_text = response.text.strip()

        return JSONResponse(content={"answer": response_text})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"An error occurred: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)