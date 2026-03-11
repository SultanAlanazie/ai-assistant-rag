import os
import logging
import torch
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

from vectorize_documents import embeddings, load_vectorstore, vectorstore_exists

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Assistant", layout="centered")
st.title("AI Assistant using RAG System")

# GPU/CPU indicator
device_label = "GPU ON" if torch.cuda.is_available() else "CPU ON"
st.sidebar.caption(f" Device: {device_label}")
if not torch.cuda.is_available():
    st.sidebar.warning(
        "Running on CPU. Embeddings will be slow.\n\n"
        "Reinstall PyTorch with CUDA support:\n"
        "`pip install torch --index-url https://download.pytorch.org/whl/cu121`"
    )


if not vectorstore_exists():
    st.error(
        "Vector store not found. "
        "Run (python vectorize_documents.py) first."
    )
    st.stop()


# Setup
@st.cache_resource(show_spinner="Loading vector store.")
def get_vectorstore():
    logger.info("Loading vector store from disk.")
    return load_vectorstore()


@st.cache_resource(show_spinner="Initialising LLM & chain.")
def get_chain(_vectorstore):
    llm = Ollama(
        model="llama3.1",   
        temperature=0,
        num_predict=1500,   
        num_ctx=4096,       
    )

    retriever = _vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20},
    )

    prompt_template = """You are a precise assistant. Answer using ONLY the context provided.

Rules:
- List ALL steps or items completely — never skip or truncate.
- Do not infer or add information not in the context.
- If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Complete Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=prompt_template,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    return chain


# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

vectorstore = get_vectorstore()
chain = get_chain(vectorstore)

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question about your documents.")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Streaming: tokens appear as they generate
        response_box = st.empty()
        full_answer = ""

        with st.spinner("Thinking."):
            response = chain.invoke({"question": user_input})

        answer = response["answer"]
        source_docs = response.get("source_documents", [])

        import time
        words = answer.split(" ")
        for i, word in enumerate(words):
            full_answer += word + " "
            if i % 8 == 0:
                response_box.markdown(full_answer + "▌")
        response_box.markdown(answer)

        # Source citations
        if source_docs:
            with st.expander("Sources ↓↓↓"):
                seen = set()
                for doc in source_docs:
                    src = doc.metadata.get("source", "unknown")
                    page = doc.metadata.get("page", "?")
                    key = f"{src}::{page}"
                    if key not in seen:
                        seen.add(key)
                        st.markdown(f"- `{os.path.basename(src)}` — page {page}")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})