import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

_llm = None


def get_llm(temperature: float = 0.1):
    global _llm
    if _llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not found — check your .env file")
        _llm = ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=api_key,
            temperature=temperature,
        )
    return _llm