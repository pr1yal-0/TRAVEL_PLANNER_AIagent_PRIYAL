# destination_utils.py

import os
import pickle
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

CSV_PATH = "destinations.csv"
VECTORSTORE_PATH = "vectorstore.pkl"

def create_vector_store():
    print("📄 Loading CSV and generating embeddings...")
    df = pd.read_csv(CSV_PATH)

    docs = []
    for _, row in df.iterrows():
        if pd.notna(row["name"]) and pd.notna(row["description"]):
            title = f"{row['name']}, {row['country']}"
            content = f"{title} - {row['description']}"
            docs.append(Document(page_content=content))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    with open(VECTORSTORE_PATH, "wb") as f:
        pickle.dump(vectorstore, f)

    print("✅ Vectorstore created and saved to:", VECTORSTORE_PATH)

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    with open(VECTORSTORE_PATH, "rb") as f:
        return pickle.load(f)
