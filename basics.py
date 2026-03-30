from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

#load_dotenv()

# ── Step 1: Connect to Gemini via LangChain ──────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
)

# ── Step 2: Simple single message ────────────────────
response = llm.invoke("What is a Python list in one sentence?")
print("Simple response:")
print(response.content)
print()

# ── Step 3: With system message ──────────────────────
messages = [
    SystemMessage(content="You are a Python tutor. Always give code examples."),
    HumanMessage(content="What is a dictionary?"),
]
response = llm.invoke(messages)
print("With system message:")
print(response.content)
