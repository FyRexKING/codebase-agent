import uuid
import os
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from app.config import COLLECTION_NAME
import google.generativeai as genai
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")
# Lightweight embedding model
embedding_model = TextEmbedding()
# In-memory vector DB
qdrant = QdrantClient(":memory:")
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
def embed(text):
    return list(embedding_model.embed([text]))[0].tolist()
def index_chunks(chunks):
    points = []

    for chunk in chunks:
        vector = embed(chunk["text"])

        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "path": chunk["path"],
                "text": chunk["text"]
            }
        ))
    print(len(chunks))
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
def explain(query, chunks):
    context = "\n\n".join([c["snippet"] for c in chunks])

    prompt = f"""
You are a senior software engineer.

Answer using the code context.

Context:
{context}

Question:
{query}
"""

    response = model.generate_content(prompt)
    return response.text
def search(query):
    vector = embed(query)

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=3
    )

    chunks= []
    for point in results.points:
        payload = point.payload

        chunks.append({
            "path": payload["path"],
            "snippet": payload["text"][:300]
        })
    explanation=explain(query,chunks)
    return{
        "explanation":explanation,
        "sources":len(chunks)
    }