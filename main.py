import google.generativeai as genai
import pinecone
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv
from functools import lru_cache
from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf

chat_history = []

# Load environment variables
load_dotenv()

app = FastAPI()

templates = Jinja2Templates(directory="static/templates")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set the environment variables and API keys
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=os.environ['PINECONE_API_KEY'])
index_name = "world-history"
index = pc.Index(index_name)

# Load the model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModel.from_pretrained(model_name)

VECTOR_DIMENSION = 384

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@lru_cache(maxsize=1024)
def get_cached_embedding(question):
    # Tokenize the input question
    inputs = tokenizer(question, return_tensors="tf", padding=True, truncation=True, max_length=512)
    
    # Get the model's output (hidden states)
    outputs = model(inputs)
    
    # Compute the mean of the hidden states to get the embeddings
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    
    # Reduce the dimension to match the Pinecone index
    embeddings = tf.keras.layers.Dense(VECTOR_DIMENSION)(embeddings)
    
    # Return the embeddings as a list
    return embeddings.numpy().tolist()[0]

@app.post("/ask")
async def ask(request: Request, question: str = Form(...)):
    global chat_history
    
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
            # Update chat history
            chat_history.append({"user": question, "bot": response_text})
            return templates.TemplateResponse("index.html", {"request": request, "answer": response_text, "history": chat_history})

        # Process the question normally
        query_embedding = get_cached_embedding(question)
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

            response = genai.generate_text(
                prompt=f"You are an expert on the book 'World History: Cultures, States, and Societies to 1500.' Answer the following question in a brief and conversational manner.\n\n{augmented_query}",
                temperature=0.5,
                safety_settings=[{"category": "HARM_CATEGORY_DEROGATORY", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}]
            )

            # Debug log response
            print("Generated Response:", response)

            generated_text = response.result or response.generated_text or response.choices[0].text
            response_text = generated_text.strip()  # Ensure no extra whitespace

        # Update chat history
        chat_history.append({"user": question, "bot": response_text})

        return templates.TemplateResponse("index.html", {"request": request, "answer": response_text, "history": chat_history})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "answer": f"An error occurred: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
