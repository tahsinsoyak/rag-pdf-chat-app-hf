import os
import torch
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from transformers import pipeline

qa_device = 0 if torch.cuda.is_available() else -1
embedding_device = "cuda" if torch.cuda.is_available() else "cpu"


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


vector_store = None


def clean_text(text: str) -> str:
    """
    Removes extra spaces and newlines to produce a cleaner text.
    """
    return " ".join(text.split())


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a given PDF file.
    """
    try:
        reader = PdfReader(file_path)
        text = ""
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text += f"--- Page {page_num} ---\n{page_text}\n"
        return text
    except Exception as e:
        return ""


# --------------------------------------------------------------------------------
# QA Pipeline Setup - English Model with GPU support if available
# --------------------------------------------------------------------------------
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2",
    device=qa_device,
    use_fast=True,
)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Renders the main page.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Processes an uploaded PDF:
      - Saves the file temporarily.
      - Extracts and cleans text.
      - Splits text into chunks.
      - Computes embeddings and updates the global FAISS vector store.
    """
    temp_file_path = f"temp_{file.filename}"

    # Save temporary file
    try:
        content = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        return {"error": "File could not be saved."}

    # Extract and clean text
    raw_text = extract_text_from_pdf(temp_file_path)
    text = clean_text(raw_text)

    # Remove temporary file
    try:
        os.remove(temp_file_path)
    except Exception as e:
        pass

    if not text.strip():
        return {"error": "No text could be extracted, please try another file."}

    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1500, chunk_overlap=300
    )
    texts = text_splitter.split_text(text)

    # Compute embeddings and update FAISS vector store using GPU if available
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": embedding_device},
    )
    global vector_store
    try:
        if vector_store is None:
            vector_store = FAISS.from_texts(texts, embeddings)
        else:
            new_store = FAISS.from_texts(texts, embeddings)
            vector_store.merge_from(new_store)
    except Exception as e:
        print(f"Exception during vector store update: {e}")
        return {"error": f"Error updating vector store: {str(e)}"}

    return RedirectResponse("/", status_code=303)


@app.post("/query/")
async def query_document(request: Request, query: str = Form(...)):
    """
    Generates an answer for a query:
      - Searches the FAISS vector store for relevant text chunks.
      - For each chunk, runs the QA pipeline individually.
      - Returns the answer with the highest confidence score.
    """
    if vector_store is None:
        result = {"answer": "Please upload a PDF first."}
        return templates.TemplateResponse(
            "index.html", {"request": request, "result": result}
        )

    try:
        docs = vector_store.similarity_search(query, k=8)
    except Exception as e:
        result = {"answer": "Error during similarity search."}
        return templates.TemplateResponse(
            "index.html", {"request": request, "result": result}
        )

    if not docs:
        result = {"answer": "No relevant content found."}
        return templates.TemplateResponse(
            "index.html", {"request": request, "result": result}
        )

    # Run QA on each document chunk and select the best answer
    best_answer = None
    best_score = -1.0

    for doc in docs:
        print(doc.page_content)
        context = doc.page_content
        if len(context) > 2048:
            context = context[:2048]
        try:
            qa_result = qa_pipeline(question=query, context=context)
            print(qa_result)
            score = qa_result.get("score", 0)
            if score > best_score:
                best_score = score
                best_answer = qa_result.get("answer", "")
        except Exception as e:
            pass

    if best_answer is None:
        best_answer = "No answer found."

    result = {"answer": best_answer}
    return templates.TemplateResponse(
        "index.html", {"request": request, "result": result}
    )


@app.post("/clear/")
async def clear_vector_store(request: Request):
    """
    Resets the global FAISS vector store.
    """
    global vector_store
    vector_store = None
    result = {"message": "Vector store reset."}
    return templates.TemplateResponse(
        "index.html", {"request": request, "result": result}
    )


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
