import os

from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_model_complete, hf_embed
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer
from lightrag.kg.shared_storage import initialize_pipeline_status

import asyncio
import nest_asyncio

nest_asyncio.apply()

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=hf_model_complete,
        llm_model_name="deepseek-ai/DeepSeek-Prover-V2-671B",
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=5000,
            func=lambda texts: hf_embed(
                texts,
                tokenizer=AutoTokenizer.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2"
                ),
                embed_model=AutoModel.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2"
                ),
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    rag = asyncio.run(initialize_rag())

    rag.insert("ACH is automatic compliance hardener")
    # Perform naive search
    print(
        rag.query(
            "what is ACH?", param=QueryParam(mode="naive")
        )
    )

    # Perform local search
    print(
        rag.query(
            "what is ACH?", param=QueryParam(mode="local")
        )
    )

    # Perform global search
    print(
        rag.query(
            "what is ACH?", param=QueryParam(mode="global")
        )
    )

    # Perform hybrid search
    print(
        rag.query(
            "what is ACH?", param=QueryParam(mode="hybrid")
        )
    )


if __name__ == "__main__":
    main()