from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from collections import defaultdict

from config import pinecone_key
from utils.pinecone_store import get_pinecone_index
from services.outfit_recommender import get_outfit, outfit_picker

router = APIRouter()

class PromptRequest(BaseModel):
    prompt: str
    outfit_options: list[str]

@router.post("/recommend-outfit/")
def get_outfit_from_prompt(request: PromptRequest):
    try:
        results = outfit_picker(request.prompt, request.outfit_options)

        if not results:
            return JSONResponse(status_code=404, content={"error": "No outfit found."})

        image_paths = [
            {
                "image_path": img_path,
                "score": round(score, 4),
                "category": category,
                "prompt": prompt
            }
            for img_path, score, category, prompt in results
        ]

        return {"status": "success", "data": image_paths}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})