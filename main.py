import os
import sqlite3
from dotenv import load_dotenv
from collections import OrderedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Disable Chroma telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# Load environment variables
load_dotenv()

# Load PDF
loader = PyPDFLoader("fashion_data.pdf")
documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=300
)
texts = text_splitter.split_documents(documents)

# Initialize embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2"
)

# Create / persist Chroma vector store
vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory="data"
)

vector_store.add_documents(texts)

# Validate setup
results = vector_store.similarity_search(
    "What are some popular items for winter?",
    k=3
)

print("Vector DB created successfully")
print(results)
