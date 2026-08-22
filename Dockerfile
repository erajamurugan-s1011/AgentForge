FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY agentforge/ ./agentforge/

EXPOSE 8000

CMD ["uvicorn", "agentforge.api.main:app", "--host", "0.0.0.0", "--port", "8000"]