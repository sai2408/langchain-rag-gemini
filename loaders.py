from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, DirectoryLoader
from dotenv import load_dotenv

load_dotenv()

# ── Load a single TXT file ───────────────────────────
print("=== Single TXT file ===")
loader = TextLoader("data/sample.txt", encoding="utf-8")
docs   = loader.load()
print(f"Pages loaded : {len(docs)}")
print(f"Content preview : {docs[0].page_content[:100]}")
print(f"Metadata : {docs[0].metadata}")
print()

# ── Load a single PDF file ───────────────────────────
print("=== Single PDF file ===")
loader = PyMuPDFLoader("data/python_loops.pdf")
docs   = loader.load()
print(f"Pages loaded : {len(docs)}")
print(f"Content preview : {docs[0].page_content[:100]}")
print(f"Metadata : {docs[0].metadata}")
print()

# ── Load ALL files from a folder ─────────────────────
print("=== Entire data/ folder ===")
loader = DirectoryLoader(
    "data/",
    glob="**/*.txt",         # load all .txt files
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
docs = loader.load()
print(f"Total documents loaded : {len(docs)}")
for doc in docs:
    print(f"  {doc.metadata['source']} — {len(doc.page_content)} chars")
