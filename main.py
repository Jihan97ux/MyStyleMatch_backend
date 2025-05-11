from fastapi import FastAPI
from routes.upload_img import router as upload_router
from routes.retrieve_data import router as retrieve_router
from utils.prompt_preprocess import prompt_preprocessing

app = FastAPI()
app.include_router(upload_router)
app.include_router(retrieve_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    
    # long_prompt = "pilihkan summer outfit untuk berfoto-foto di taman bunga"
    # outfit_options = ["dress", "sandal"]
    # result = prompt_preprocessing(long_prompt, outfit_options)
    # print(result)