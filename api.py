from fastapi import FastAPI
from pydantic import BaseModel
from src.vector_db_manager import VectorDBManager
from src.chatbot import Chatbot

app = FastAPI()

db_manager = VectorDBManager()
chatbot = Chatbot(
    project_retriever=db_manager.get_retriever(mode="recommend"),
    service_retriever=db_manager.get_retriever(mode="explain")
)

class ChatRequest(BaseModel):
    query: str
    mode: str

@app.post("/chat")
def handle_chat(request: ChatRequest):
    answer = chatbot.generate_response(request.query, request.mode)
    return {"answer": answer}