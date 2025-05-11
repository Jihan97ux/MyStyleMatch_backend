from pinecone import Pinecone, ServerlessSpec
import os
import uuid

def get_pinecone_index(api_key: str, index_name: str = "outfit-index"):
    pc = Pinecone(api_key=api_key)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=512,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    return pc.Index(index_name)

def upsert_embedding(index, vector, img_path, category):
    vector_id = str(uuid.uuid4())
    index.upsert([
        {
            "id": vector_id,
            "values": vector,
            "metadata": {
                "img_path": img_path,
                "category": category
            }
        }
    ])
    return vector_id
