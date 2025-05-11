import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="config.env")

IMAGE_FOLDER = "images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

pinecone_key = os.getenv("PINECONE_KEY")
openai_key = os.getenv("OPENAI_KEY")
