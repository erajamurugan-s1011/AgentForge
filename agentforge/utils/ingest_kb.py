import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from agentforge.data.kb_content import KB_ARTICLES

load_dotenv()

COLLECTION_NAME = "helpdesk_kb"
EMBED_MODEL = "all-MiniLM-L6-v2"


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    cloud_url = os.getenv("QDRANT_CLOUD_URL")
    cloud_key = os.getenv("QDRANT_CLOUD_API_KEY")
    if cloud_url and cloud_key:
        client = QdrantClient(url=cloud_url, api_key=cloud_key)
        print("Connecting to Qdrant Cloud")
    else:
        client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=6333)
        print("Connecting to local Qdrant")

    vector_size = model.get_sentence_embedding_dimension()
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' created (dim={vector_size})")

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="category",
        field_schema="keyword",
    )
    print("Payload index created on 'category' field")

    texts = [f"{a['title']}. {a['content']}" for a in KB_ARTICLES]
    print(f"Embedding {len(texts)} articles...")
    embeddings = model.encode(texts, show_progress_bar=True)

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "kb_id": KB_ARTICLES[i]["id"],
                "category": KB_ARTICLES[i]["category"],
                "title": KB_ARTICLES[i]["title"],
                "content": KB_ARTICLES[i]["content"],
            },
        )
        for i in range(len(KB_ARTICLES))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Upserted {len(points)} points into '{COLLECTION_NAME}'")

    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"Verification: collection now has {count} points")


if __name__ == "__main__":
    main()