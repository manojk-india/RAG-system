import chainlit as cl
import os
import shutil

from entry import *


@cl.step(type="tool")
async def process_pdf_tool():
    await one_shot_vectorizer()
    return "PDF processed and vector store created."

@cl.step(type="tool")
async def answer_question(message:str):
    query = message
    results = await query_vector_store(query, k=3)
    if results:
        answer = "\n\n---\n\n".join([f"**Context:**\n{chunk}\n**Score:** {score:.4f}" for chunk, score in results])
    else:
        answer = "Sorry, I couldn't find relevant information in the PDF."
    await cl.Message(content=answer).send()


@cl.on_chat_start
async def start():
    files = None
    while not files:
        files = await cl.AskFileMessage(
            content="Please upload a single PDF file (max 100 MB) to begin!",
            accept=["application/pdf"],
            max_size_mb=100
        ).send()
    pdf_file = files[0]  # Get the first (and only) file

    # Ensure the pdf directory exists
    os.makedirs("./pdf", exist_ok=True)

    save_path = "./pdf/sample.pdf"

    # Copy the uploaded file from its temp path to your desired location
    shutil.copy(pdf_file.path, save_path)

    await cl.Message(
        content=f"`{save_path}` uploaded and saved!"
    ).send()

    await process_pdf_tool()

    await cl.Message(
        content="You can now ask questions about the PDF!"
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    await answer_question(message.content)