from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image
from rembg import remove
from io import BytesIO
import numpy as np
import uuid
import os

from config import pinecone_key
from config import IMAGE_FOLDER
from utils.pinecone_store import get_pinecone_index, upsert_embedding
from utils.encode_img import encode_image

def process_image(image: Image.Image):
    output_image = remove(image)
    unique_filename = f"{uuid.uuid4()}.png"
    return output_image, unique_filename

def is_similar_image(index, new_vector, similarity_threshold=0.95, top_k=5):

    results = index.query(vector=new_vector, top_k=top_k, include_values=False, include_metadata=True)
    
    for match in results.matches:
        score = match.score
        if score > similarity_threshold:
            return True, match.metadata.get("img_path", None)
    
    return False, None

router = APIRouter()

@router.post("/upload/")
async def upload_images(
    category: str = Form(...),
    files: List[UploadFile] = File(...)
):
    results = []

    index = get_pinecone_index(api_key=pinecone_key)

    for file in files:
        try:
            image = Image.open(BytesIO(await file.read())).convert("RGBA")
            output_image, filename = process_image(image)
            embedding = encode_image(output_image).cpu().numpy().tolist()
            
            is_similar, existing_img = is_similar_image(index, embedding[0])
            if is_similar:
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "reason": f"Image serupa terdapat di database: {existing_img}"
                })
                continue
            
            save_path = os.path.join(IMAGE_FOLDER, filename)
            output_image.save(save_path)
            img_path = os.path.join("images", filename)
            
            vector_id = upsert_embedding(index, embedding[0], img_path, category)

            results.append({
                "vector_id": vector_id,
                "category": category,
                "img_path": img_path,
                "embedding": embedding
            })

        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    return {"status": "success", "data": results}
