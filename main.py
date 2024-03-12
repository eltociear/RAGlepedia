import asyncio
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import HTMLResponse
from httpx import AsyncClient
from chatBot import ChatBot
from utils.config import Config


from collections import deque

import os

from dotenv import load_dotenv

load_dotenv()

cnfg = Config()

# init the fastAPI app
app = FastAPI()

# init VirtualAssistant memory
memory = deque(maxlen=cnfg.MAX_MEMORY_SIZE)

# init the clients
websocket_clients = []

# build the chatBot object
chatbot = ChatBot(embedding_model_name=os.getenv("EMBEDDINGS_MODEL_NAME"))
if not chatbot.check_vector_fullness():
    custom_dataset, custom_dataloader, chunk_splitter = chatbot.build_dataset_objects()
    chatbot.upload_full_data(custom_dataloader, chunk_splitter)


@app.get("/")
async def get():
    template_path = os.path.join(os.path.dirname(__file__), "templates/index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Template file not found.")

    return HTMLResponse(
        content=open("templates/index.html", "r").read(), status_code=200
    )


@app.websocket("/chat")
async def chat_endpoint(websocket: WebSocket):
    # Accept incoming websocket connection
    await websocket.accept()
    # Add the websocket to the list of clients
    websocket_clients.append(websocket)

    try:
        # initialise the role of the system
        messages = list()
        messages.append(
            {
                "role": "system",
                "content": """You are Q&A bot. A highly intelligent system that answers user questions based on the 'CONTEXT' provided by the user above
                            each 'QUESTION'. Be Specific and Descriptive. Order matters, so if for example user tells you 'summarize the following...', you have to summarize the
                            query provided after the word 'following'. If you don't know the answer, please think rationally answer your own knowledge base.""",
            }
        )

        while True:
            # Receive message from the client
            user_message = await websocket.receive_text()
            # use RAG to retrieve relevant text-passages
            retrieved = chatbot.rag_query(query_text=user_message)

            top_k_matches = retrieved["matches"]
            contexts = [m["metadata"]["text"] for m in top_k_matches]

            # create the context message to pass to LLM
            context_str = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n"
            print(context_str)
            memory.append(context_str)

            # Combine RAG contexts with past interactions from memory
            combined_context = "\n\n".join(contexts + list(memory))
            messages.append(
                {"role": "user", "content": f"'CONTEXT': {combined_context}"}
            )

            # now pass the user message to model
            messages.append({"role": "user", "content": f"'QUESTION': {user_message}"})
            memory.append(context_str)
            # Send message to Azure OpenAI and get response
            chatbot_response = await get_openai_response(messages)
            memory.append(chatbot_response)

            # then append to history messages
            messages.append({"role": "assistant", "content": chatbot_response})

            # Broadcast the response to all connected clients
            for client in websocket_clients:
                await client.send_text(chatbot_response)

    finally:
        # Remove the websocket from the list of clients if connection is closed
        websocket_clients.remove(websocket)


async def get_openai_response(message: str) -> str:
    async with AsyncClient() as client:

        response = chatbot.respond(message)
        return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
