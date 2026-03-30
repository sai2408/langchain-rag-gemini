# 🦜🔗 LangChain RAG Pipeline with Gemini

> A beginner-friendly, end-to-end Retrieval-Augmented Generation (RAG) pipeline built with **LangChain**, **Google Gemini**, and **ChromaDB** — from document loading to intelligent Q&A.

---

## 📌 What This Project Covers

| Module | File | Description |
|---|---|---|
| 🔤 LLM Basics | `basics.py` | Connect to Gemini via LangChain, single & multi-turn prompts |
| 📄 Document Loaders | `loaders.py` | Load TXT, PDF, and entire directories |
| ✂️ Text Splitters | `splitters.py` | `CharacterTextSplitter` vs `RecursiveCharacterTextSplitter` |
| 🗃️ Vector Store | `vector_store.py` | Embed chunks and persist in ChromaDB |
| 🤖 Retrieval QA | `retrieval_qa.py` | Full RAG chain with custom prompt using LCEL |

---

## 🏗️ Architecture

```
Your Documents (TXT / PDF)
        │
        ▼
  Document Loaders
        │
        ▼
   Text Splitters  ──→  Chunks
        │
        ▼
  Gemini Embeddings
        │
        ▼
    ChromaDB (Vector Store)
        │
        ▼
    Retriever (Top-k)
        │
        ▼
  RAG Prompt + Gemini LLM
        │
        ▼
     Answer + Sources
```

---

## 🛠️ Tech Stack

- **[LangChain](https://www.langchain.com/)** — Orchestration framework
- **[Google Gemini](https://ai.google.dev/)** — LLM (`gemini-2.5-flash-lite`) + Embeddings (`gemini-embedding-001`)
- **[ChromaDB](https://www.trychroma.com/)** — Local vector store
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** — PDF loading
- **Python 3.10+**

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/langchain-rag-gemini.git
cd langchain-rag-gemini
```

### 2. Install dependencies

```bash
pip install langchain langchain-community langchain-google-genai langchain-chroma \
            chromadb pymupdf python-dotenv
```

### 3. Set up your API key

Create a `.env` file in the root:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

> 🔑 Get your free Gemini API key at [Google AI Studio](https://aistudio.google.com/)

### 4. Add your documents

Place `.txt` or `.pdf` files inside the `data/` folder:

```
data/
├── sample.txt
├── python_loops.pdf
└── ...
```

### 5. Run the pipeline

```bash
# Step 1 — Try the LLM basics
python basics.py

# Step 2 — Test document loading
python loaders.py

# Step 3 — Experiment with text splitters
python splitters.py

# Step 4 — Build the vector store (run once)
python vector_store.py

# Step 5 — Ask questions!
python retrieval_qa.py
```

---

## 💬 Sample Output

```
Question: How do I add items to a list?
──────────────────────────────────────────────────
Answer: You can add items to a list using the append() method for
        a single item, or extend() to add multiple items at once.

Sources used:
  data/sample.txt — You can use append() to add a single item to...
```

---

## 📂 Project Structure

```
langchain-rag-gemini/
├── data/                    # Your source documents (TXT, PDF)
├── lc_vector_store/         # ChromaDB persisted vector store (auto-created)
├── basics.py                # LLM basics & prompting
├── loaders.py               # Document loading examples
├── splitters.py             # Text splitting strategies
├── vector_store.py          # Embed & store documents in ChromaDB
├── retrieval_qa.py          # Full RAG Q&A chain (LCEL style)
├── .env                     # API keys (not committed)
├── .gitignore
└── README.md
```

---

## 🧠 Key Concepts Demonstrated

- **Document Loading** — TXT, PDF, and directory-level loading
- **Text Splitting** — Chunking strategies with overlap for context continuity
- **Vector Embeddings** — Converting text to semantic vectors with Gemini
- **Similarity Search** — Top-k retrieval from ChromaDB
- **RAG Chain (LCEL)** — Modern LangChain Expression Language pipeline
- **Custom Prompting** — PyBot tutor persona with context-grounded answers
- **Source Tracking** — Trace which documents the answer came from

---


## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

[MIT](LICENSE)

---

> Built with ❤️ to learn and share LangChain + Gemini RAG patterns.
