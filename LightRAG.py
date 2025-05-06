from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc
import pdfplumber
from utils import *
from lightrag.kg.shared_storage import initialize_pipeline_status
import asyncio


async def main():
    rag = LightRAG(
        working_dir="./LightRAG_docs",
        llm_model_func=ollama_model_complete,
        llm_model_name="mistral:latest",
        llm_model_kwargs={"options": {"num_ctx": 32768}},  # Essential for large docs
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embedding(texts, embed_model="nomic-embed-text")
        )
    )

    # text=extract_text_from_pdf("pdf/sample.pdf")

    await rag.initialize_storages()  
    await initialize_pipeline_status()  # Creates pipeline_status[1]

    rag.insert('ACH is automatic compliance hardener')


    response = rag.query(
        "what is ACH ?",
        param=QueryParam(mode="hybrid")  # Options: naive/local/global/hybrid
    )
    print(f"Answer: {response.answer}\nSources: {response.sources}")

asyncio.run(main())