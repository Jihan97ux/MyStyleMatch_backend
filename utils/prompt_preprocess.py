import openai
import time

from config import openai_key

client = openai.OpenAI(api_key=openai_key)

def prompt_preprocessing(long_prompt, categories, max_retries=3):
    """
    Fungsi untuk menyederhanakan prompt pengguna berdasarkan kategori fashion dan mengembalikannya dalam format JSON.

    Args:
    - long_prompt (str): prompt dari user
    - categories (list): daftar kategori fashion, contoh: ["dress", "sandal"]
    - client: instance OpenAI client, misal: `openai`
    - max_retries (int): jumlah maksimum percobaan jika terjadi error

    Returns:
    - dict: hasil dalam format JSON sesuai kategori
    """
    input_example = "hari ini saya ingin ke taman bunga berfoto-foto, saya bingung mau pakai outfit apa, pilihkan outfit yang cocok untuk berjalan-jalan ke taman bunga"
    response_example = {
        "dress": "dress yang cocok digunakan untuk berfoto di taman bunga",
        "sandal": "sandal yang cocok digunakan untuk berjalan-jalan ke taman bunga"
    }

    system_prompt = f"""
Kamu adalah asisten fashion yang bertugas menyederhanakan dan mengubah prompt dari user menjadi kebutuhan fashion yang relevan berdasarkan kategori tertentu.

### Contoh:
User Prompt:
"{input_example}"

Kategori:
{categories}

Output yang diharapkan (format JSON):
{response_example}

### Instruksi:
Berdasarkan prompt yang diberikan user dan daftar kategori, buatlah JSON dengan setiap kategori sebagai key, dan deskripsi singkat sebagai value yang menjelaskan kebutuhan fashion yang sesuai dengan konteks prompt user.

Hasilkan JSON sesuai format, hanya deskripsi pendek tapi jelas dan kontekstual untuk tiap kategori.
"""

    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f'Prompt: "{long_prompt}"\nKategori: {categories}'}
                ],
                temperature=0.2
            )

            content = response.choices[0].message.content.strip()
            short_prompts = eval(content) if content.startswith("{") else {}
            return short_prompts

        except Exception as e:
            print(f"Terjadi kesalahan: {e}")
            retries += 1
            time.sleep(2)

    return {}