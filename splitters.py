from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Load a document first
loader = TextLoader("data/sample.txt", encoding="utf-8")
docs   = loader.load()
text   = docs[0].page_content

print(f"Original length: {len(text)} chars\n")

# ── Method 1: CharacterTextSplitter ──────────────────
print("=== CharacterTextSplitter ===")
splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separator="\n"
)
chunks = splitter.create_documents([text])
print(f"Chunks created : {len(chunks)}")
print(f"Chunk 0 preview: {chunks[0].page_content[:80]}")
print()

# ── Method 2: RecursiveCharacterTextSplitter ─────────
# This is the RECOMMENDED splitter — respects sentences
print("=== RecursiveCharacterTextSplitter ===")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]  # tries these in order
)
chunks = splitter.create_documents([text])
print(f"Chunks created : {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i}: {len(chunk.page_content)} chars — {chunk.page_content[:50]}...")
