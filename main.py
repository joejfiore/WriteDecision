import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------- SETUP ----------

PROJECT_ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")
SRO_DIR = os.path.join(DATA_DIR, "sro_decisions")
VSTORE_DIR = os.path.join(PROJECT_ROOT, "vectorstores")
TEMPLATE_VS_PATH = os.path.join(VSTORE_DIR, "templates")
SRO_VS_PATH = os.path.join(VSTORE_DIR, "sro_decisions")
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")

os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(SRO_DIR, exist_ok=True)
os.makedirs(VSTORE_DIR, exist_ok=True)

# ---------- CREATE .ENV IF MISSING ----------

if not os.path.exists(ENV_FILE):
    openai_key = input("Enter your OpenAI API key: ").strip()
    with open(ENV_FILE, "w") as f:
        f.write(f"OPENAI_API_KEY={openai_key}\n")
    print(".env file created.")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# ---------- EMBED PDFs TO VECTORSTORES ----------

def build_vectorstore_from_pdfs(input_dir, output_path):
    if os.path.exists(output_path):
        print(f"[✓] Vectorstore already exists: {output_path}")
        return FAISS.load_local(output_path, embedding)

    print(f"[•] Processing PDFs in {input_dir}...")
    all_chunks = []
    for file in os.listdir(input_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(input_dir, file))
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
            print(f"  - {file} → {len(chunks)} chunks")
    if not all_chunks:
        raise ValueError(f"No PDF files found in {input_dir}.")
    vs = FAISS.from_documents(all_chunks, embedding)
    vs.save_local(output_path)
    print(f"[✓] Vectorstore saved: {output_path}")
    return vs

template_db = build_vectorstore_from_pdfs(TEMPLATES_DIR, TEMPLATE_VS_PATH)
sro_db = build_vectorstore_from_pdfs(SRO_DIR, SRO_VS_PATH)

# ---------- FASTAPI SERVER ----------

app = FastAPI()

class SearchQuery(BaseModel):
    query: str

@app.post("/search_templates")
def search_templates(query: SearchQuery):
    results = template_db.similarity_search(query.query, k=5)
    return {"results": [r.page_content for r in results]}

@app.post("/search_sro")
def search_sro(query: SearchQuery):
    results = sro_db.similarity_search(query.query, k=5)
    return {"results": [r.page_content for r in results]}

# ---------- LAUNCH SERVER ----------

if __name__ == "__main__":
    print("\n[✓] System ready. Drop PDFs into /data/templates/ or /data/sro_decisions/")
    print("[✓] Then call the API at http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
