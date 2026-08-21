import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GROQ_API_KEY")

if key:
    print(f"GROQ_API_KEY loaded: yes (length={len(key)})")
else:
    print("GROQ_API_KEY loaded: NO — check your .env file")