from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import openai
import base64
import os

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    b64_image = base64.b64encode(image_bytes).decode('utf-8')

    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Определи по фото блюда примерные КБЖУ (белки, жиры, углеводы, калории) и коротко опиши, что на фото. Ответ верни в JSON-структуре со всеми значениями."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]
            }
        ],
        max_tokens=500
    )

    try:
        text = response["choices"][0]["message"]["content"]
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        json_string = text[json_start:json_end]
        return JSONResponse(content=eval(json_string))
    except Exception as e:
        return {"error": "Ошибка парсинга", "raw": response["choices"][0]["message"]["content"]}