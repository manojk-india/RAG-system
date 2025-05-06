import os
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import List
from utils import *



# creating a vector index file 
async def create_vector_store(text: str, index_path: str = "vector_store.index"):
    # Initialize embeddings model (384-dimension)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Generate embeddings
    embeddings = model.encode(chunks)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save index and metadata
    faiss.write_index(index, index_path)
    with open(f"{index_path}.meta", "w", encoding="utf-8") as f:  # ← Add 
        f.write("\n".join(chunks))
    
    return index

# function for querying vector store
async def query_vector_store(query: str, index_path: str = "vector_store.index", k=3):
    # Load index and metadata
    index = faiss.read_index(index_path)
    with open(f"{index_path}.meta", encoding="utf-8") as f:
        chunks = f.read().splitlines()
    
    # Encode query
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    
    # Search
    distances, indices = index.search(query_embedding, k)
    
    return [(chunks[i], distances[0][j]) for j, i in enumerate(indices[0])]

# function to convert pdf to text and to a  vector index file 
async def one_shot_vectorizer():
    # 1. Extract text from PDF
    pdf_text = extract_text_from_pdf("./pdf/sample.pdf")
    
    # 2. Create and save vector store
    await create_vector_store(pdf_text)

