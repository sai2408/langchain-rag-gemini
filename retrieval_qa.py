from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# ── Step 1: Load existing vector store ───────────────
print("Loading vector store...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vectordb = Chroma(
    persist_directory="lc_vector_store",
    embedding_function=embeddings
)
print(f"Loaded {vectordb._collection.count()} chunks")

# ── Step 2: Setup LLM ────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3,
)

# ── Step 3: Setup retriever ───────────────────────────
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# ── Step 4: Custom RAG prompt ─────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are PyBot, a friendly Python tutor.
Answer the question using ONLY the context below.
If the answer is not in the context say:
"I don't have that in my notes."

Context:
{context}

Question: {question}

Answer:"""
)

# ── Step 5: Format retrieved docs into a string ───────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── Step 6: Build the RAG chain (modern LCEL style) ───
#
# LCEL = LangChain Expression Language
# This is the modern way — replaces RetrievalQA
#
# Flow:
# question → retriever → format_docs → prompt → llm → parse
#
rag_chain = (
    {
        "context":  retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

# ── Step 7: Ask function with source tracking ──────────
def ask(question):
    print(f"\nQuestion: {question}")
    print("─" * 50)

    # Get answer from chain
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")

    # Get sources separately
    docs = retriever.invoke(question)
    print(f"\nSources used:")
    for doc in docs:
        src     = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:60].replace('\n', ' ')
        print(f"  {src} — {preview}...")
    print()


# ── Step 8: Test it ────────────────────────────────────
ask("How do I add items to a list?")
ask("What is the __init__ method?")
ask("How do I handle a ValueError?")
ask("What is asyncio?")
