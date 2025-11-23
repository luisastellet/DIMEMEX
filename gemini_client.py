import os
import json
import re
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

load_dotenv()
API_KEY = os.getenv('API_KEY')

if not API_KEY:
    raise ValueError("API_KEY not found in environment variables. Please check your .env file.")

client = genai.Client(api_key=API_KEY)

def generate_json(prompt, model="gemini-2.0-flash"):
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=GenerateContentConfig(
            response_mime_type="text/plain"
        )
    )

    raw = response.text
    print("Resposta bruta do modelo:", raw[:500]) 

    json_match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if not json_match:
        raise ValueError("Nenhum JSON encontrado na resposta")

    json_str = json_match.group(0)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print("JSON bruta:", json_str)
        raise ValueError(f"JSON inválido: {e}")
