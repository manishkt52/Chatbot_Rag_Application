import streamlit as st
import os
import sqlite3
from dotenv import load_dotenv
from typing import Optional, List

from huggingface_hub import InferenceClient
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---------------- ENV SETUP ---------------- #

# Disable Chroma telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "false"

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    st.error("HUGGINGFACEHUB_API_TOKEN not found in .env")
    st.stop()

data_directory = os.path.join(os.path.dirname(__file__), "data")

if not os.path.exists(data_directory):
    st.error("Vector database not found. Please run main.py first.")
    st.stop()

# ---------------- LLM WRAPPER ---------------- #

class HFRouterLLM(LLM):
    client: InferenceClient
    max_tokens: int = 1024
    temperature: float = 1.0

    @property
    def _llm_type(self) -> str:
        return "huggingface-router"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

# ---------------- VECTOR STORE ---------------- #

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory=data_directory
)

# ---------------- LLM INIT ---------------- #

hf_client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=HF_TOKEN,
)




hf_hub_llm = HFRouterLLM(
    client=hf_client,
    max_tokens=1024,
    temperature=1.0,
)

# ---------------- PROMPT ---------------- #

prompt_template = """
As a highly knowledgeable fashion assistant, your role is to answer questions
ONLY using the provided fashion database context.

Rules:
- Stay strictly within fashion topics
- Do not add external knowledge
- If the question is unrelated to fashion, politely decline

Context:
{context}

Question:
{question}

Answer:
"""

custom_prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

rag_chain = RetrievalQA.from_chain_type(
    llm=hf_hub_llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": custom_prompt}
)

def get_response(question):
    result = rag_chain({"query": question})
    response_text = result["result"]
    answer_start = response_text.find("Answer:") + len("Answer:")
    return response_text[answer_start:].strip()

# ---------------- STREAMLIT UI ---------------- #

st.markdown(
    """
    <style>
        .appview-container .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
<h3 style='text-align: left; color: black; padding-top: 35px; border-bottom: 3px solid red;'>
    Discover the AI Styling Recommendations 👗👠
</h3>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🤖 FashionBot: Your AI Style Companion")
    st.markdown("""
    Ask me about:
    - Fashion trends 👕
    - Styling advice 👢
    - Seasonal wardrobes 🌞
    - Accessories 💍
    """)

initial_message = """
Hi! I'm your FashionBot 🤖  

Try asking:
- What are the top fashion trends this summer?
- Suggest an outfit for a winter wedding
- Must-have accessories for winter
- Shoes for a cocktail dress
"""

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]

st.button("Clear Chat", on_click=clear_chat_history)

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Fetching fashion advice..."):
            response = get_response(prompt)
            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
