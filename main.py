import os
import logging
import uvicorn
import json
from typing import Optional
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from langchain.vectorstores import VectorStore

from manager import ConnectionManager
from models import ChatResponse
from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query import get_chain, get_vector_store

load_dotenv()
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
persist_directory = '.db'
documents_path = 'knowledge_base/sources'

OPENAI_API_KEY = os.getenv('MY_OPENAI_API_KEY')
TRACING = bool(os.getenv('TRACING', 0))

vector_store: Optional[VectorStore] = None

manager = ConnectionManager()


@app.on_event("startup")
def startup_event():
    """runs when fastapi server start up, creates vector_store instance"""
    global vector_store
    vector_store = get_vector_store(persist_directory, documents_path)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """root endpoint to render main page"""

    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/reindex")
async def reindex():
    """reindex the vector db and updates vector store instance"""
    global vector_store
    vector_store = get_vector_store(persist_directory, documents_path,
                                    reindex=True)


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await manager.connect(websocket)
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)

    qa_chain = get_chain(vector_store, question_handler, stream_handler,
                         TRACING)

    # print("QA_CHAIN:", qa_chain)
    while True:
        try:
            # Receive and send back the client message
            socket_message = await websocket.receive_text()
            data = json.loads(socket_message)

            question = data['question']
            resp = ChatResponse(sender="you", message=question, type="stream")
            await manager.send_personal_message_json(resp.dict(), websocket)

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await manager.send_personal_message_json(start_resp.dict(), websocket)

            result = await qa_chain.acall(
                {"question": question}
            )

            source_documents = {i.page_content for i in
                                result['source_documents']}

            end_resp = ChatResponse(sender="bot", message="", type="end",
                                    source_documents=source_documents,
                                    chat_history=[])
            print(end_resp.dict())
            await manager.send_personal_message_json(end_resp.dict(), websocket)
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await manager.send_personal_message_json(resp.dict(), websocket)


if __name__ == "__main__":
    port = os.getenv('PORT', default=8000)
    uvicorn.run(app, host="127.0.0.1", port=int(port), log_level="info")
