import pinecone
from utils.encode_prompt import encode_text
from utils.pinecone_store import get_pinecone_index
from config import pinecone_key
from utils.prompt_preprocess import prompt_preprocessing

def get_outfit(prompts, outfit_options):
    if not isinstance(prompts, list) or len(prompts) == 0:
        print("Masukkan list prompt minimal satu.")
        return []

    print("Searching your outfit...")

    text_embeddings = [encode_text(p).tolist() for p in prompts]
    index = get_pinecone_index(api_key=pinecone_key)

    top_results = []

    for i, text_emb in enumerate(text_embeddings):
        prompt = prompts[i]
        response = index.query(
            vector=text_emb,
            top_k=2,
            include_metadata=True
        )
        
        # print(f"Query result for prompt: {prompt}")
        # print(response.to_dict())

        if response.matches:
            for match in response.matches:
                metadata = match.metadata
                img_path = metadata.get("img_path")
                category = metadata.get("category", "Uncategorized")
                score = match.score

                if category in outfit_options and img_path:
                    top_results.append((img_path, score, category, prompt))

    print("FINAL RESULTS:", top_results)
    return top_results

def outfit_picker(prompt_from_user, outfit_options):
    short_prompts = prompt_preprocessing(prompt_from_user, outfit_options)
    print(short_prompts)
    prompts = list(short_prompts.values())
    return get_outfit(prompts, outfit_options)