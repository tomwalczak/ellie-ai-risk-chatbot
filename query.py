from langchain import PromptTemplate
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.text_splitter import TokenTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import Chroma
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

from chat_vector_db import MyConversationalRetrievalChain
from prompts.my_prompts import DOC_CHAIN_PROMPT

OPENAI_API_KEY = os.getenv("MY_OPENAI_API_KEY")


def get_chain(
        vectorstore: VectorStore, question_handler, stream_handler,
        tracing: bool = False):
    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    memory = ConversationBufferMemory(memory_key="chat_history",
                                      input_key='question',
                                      output_key='answer',
                                      return_messages=True)

    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = OpenAI(
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
        openai_api_key=OPENAI_API_KEY,
    )
    streaming_llm = ChatOpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT,
        callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=DOC_CHAIN_PROMPT,
        callback_manager=manager
    )

    qa = MyConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
        return_source_documents=True,
        # max_sources_limit=3,
        max_tokens_limit = 3, # this name is misleading, it's actually the max number of documents to return
        verbose=False,
        memory=memory,
    )

    return qa


def get_vector_store(persist_directory: str,
                     documents_path: str, reindex=False) -> VectorStore:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    if not reindex and os.path.exists(persist_directory):
        return Chroma(embedding_function=embeddings,
                      persist_directory=persist_directory)

    loader = DirectoryLoader(documents_path, loader_cls=TextLoader)

    documents = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=500,
                                         chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return Chroma.from_documents(texts, embeddings,
                                 persist_directory=persist_directory)
