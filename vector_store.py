from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# ── Step 1: Load all documents ───────────────────────
print("Loading documents...")
loader = DirectoryLoader(
    "data/",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
docs = loader.load()
print(f"Loaded {len(docs)} documents")

# ── Step 2: Split into chunks ────────────────────────
print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

# ── Step 3: Embed and store in ChromaDB ──────────────
print("Embedding and storing...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="lc_vector_store"
)
print(f"Stored {vectordb._collection.count()} chunks in ChromaDB")

# ── Step 4: Test retrieval ───────────────────────────
print("\nTesting retrieval...")
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
results   = retriever.invoke("How do I add items to a list?")

print(f"\nTop {len(results)} results:")
for i, doc in enumerate(results):
    print(f"\n[{i+1}] Source: {doc.metadata.get('source','unknown')}")
    print(f"     Preview: {doc.page_content[:80]}...")
