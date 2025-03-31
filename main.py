import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------- SETUP ----------

PROJECT_ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")
SRO_DIR = os.path.join(DATA_DIR, "sro_decisions")
VSTORE_DIR = os.path.join(PROJECT_ROOT, "vectorstores")
TEMPLATE_VS_PATH = os.path.join(VSTORE_DIR, "templates")
SRO_VS_PATH = os.path.join(VSTORE_DIR, "sro_decisions")

os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(SRO_DIR, exist_ok=True)
os.makedirs(VSTORE_DIR, exist_ok=True)

# ---------- LOAD API KEY ----------

# Load from .env file if it exists (for local development)
load_dotenv()

# Get API key from environment variable (works with Railway)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

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
        print(f"No PDF files found in {input_dir}. Creating empty vectorstore.")
        # Create a minimal document to avoid errors
        from langchain.schema import Document
        dummy_doc = Document(page_content="Placeholder document", metadata={})
        all_chunks = [dummy_doc]
    vs = FAISS.from_documents(all_chunks, embedding)
    vs.save_local(output_path)
    print(f"[✓] Vectorstore saved: {output_path}")
    return vs

try:
    template_db = build_vectorstore_from_pdfs(TEMPLATES_DIR, TEMPLATE_VS_PATH)
    sro_db = build_vectorstore_from_pdfs(SRO_DIR, SRO_VS_PATH)
except Exception as e:
    print(f"Error building vectorstores: {e}")
    # Create fallback databases
    from langchain.schema import Document
    dummy_doc = Document(page_content="Placeholder document", metadata={})
    template_db = FAISS.from_documents([dummy_doc], embedding)
    sro_db = FAISS.from_documents([dummy_doc], embedding)

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
    print("\n[✓] System ready. API running at http://0.0.0.0:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000)