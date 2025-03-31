import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ---------- SETUP ----------
PROJECT_ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")
SRO_DIR = os.path.join(DATA_DIR, "sro_decisions")
VSTORE_DIR = os.path.join(PROJECT_ROOT, "vectorstores")
TEMPLATE_VS_PATH = os.path.join(VSTORE_DIR, "templates")
SRO_VS_PATH = os.path.join(VSTORE_DIR, "sro_decisions")

# Ensure directories exist
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(SRO_DIR, exist_ok=True)
os.makedirs(VSTORE_DIR, exist_ok=True)

# ---------- FASTAPI APP ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- LOAD API KEY ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
template_db = None
sro_db = None

# ---------- ROOT ENDPOINT ----------
@app.get("/")
async def root():
    return {
        "status": "API is running",
        "vector_search_status": "Available" if OPENAI_API_KEY else "Unavailable (No API Key)",
        "endpoints": ["/search_templates", "/search_sro"]
    }

# ---------- INITIALIZE VECTORSTORES ----------
try:
    if OPENAI_API_KEY:
        print("Initializing OpenAI embeddings...")
        embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        def build_vectorstore_from_pdfs(input_dir, output_path):
            if os.path.exists(output_path):
                print(f"[✓] Loading existing vectorstore: {output_path}")
                return FAISS.load_local(output_path, embedding)

            print(f"[•] Processing PDFs in {input_dir}...")
            all_chunks = []
            pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
            
            if not pdf_files:
                print(f"No PDF files found in {input_dir}. Creating dummy vectorstore.")
                dummy_doc = Document(page_content="Placeholder document", metadata={})
                all_chunks = [dummy_doc]
            else:
                for file in pdf_files:
                    try:
                        file_path = os.path.join(input_dir, file)
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        chunks = splitter.split_documents(docs)
                        all_chunks.extend(chunks)
                        print(f"  - {file} → {len(chunks)} chunks")
                    except Exception as e:
                        print(f"Error processing {file}: {str(e)}")
            
            vs = FAISS.from_documents(all_chunks, embedding)
            vs.save_local(output_path)
            print(f"[✓] Vectorstore saved: {output_path}")
            return vs

        # Create dummy doc for empty vectorstores
        dummy_doc = Document(page_content="Placeholder document", metadata={})
        
        # Initialize template vectorstore
        try:
            print("Building template vectorstore...")
            template_db = build_vectorstore_from_pdfs(TEMPLATES_DIR, TEMPLATE_VS_PATH)
        except Exception as e:
            print(f"Error building template vectorstore: {str(e)}")
            template_db = FAISS.from_documents([dummy_doc], embedding)

        # Initialize SRO vectorstore
        try:
            print("Building SRO vectorstore...")
            sro_db = build_vectorstore_from_pdfs(SRO_DIR, SRO_VS_PATH)
        except Exception as e:
            print(f"Error building SRO vectorstore: {str(e)}")
            sro_db = FAISS.from_documents([dummy_doc], embedding)
            
except Exception as e:
    print(f"Error during initialization: {str(e)}")

# ---------- API ENDPOINTS ----------
class SearchQuery(BaseModel):
    query: str

@app.post("/search_templates")
def search_templates(query: SearchQuery):
    if not template_db:
        raise HTTPException(
            status_code=503,
            detail="Vector search unavailable. OPENAI_API_KEY might not be set."
        )
    try:
        results = template_db.similarity_search(query.query, k=5)
        return {"results": [r.page_content for r in results]}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error performing template search: {str(e)}"
        )

@app.post("/search_sro")
def search_sro(query: SearchQuery):
    if not sro_db:
        raise HTTPException(
            status_code=503,
            detail="Vector search unavailable. OPENAI_API_KEY might not be set."
        )
    try:
        results = sro_db.similarity_search(query.query, k=5)
        return {"results": [r.page_content for r in results]}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error performing SRO search: {str(e)}"
        )