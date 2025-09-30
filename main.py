from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import openai
import base64
import os

app = FastAPI()

# Задаём ключ (лучше через переменную окружения, но пока можно так)
openai.api_key = os.getenv("OPENAI_API_KEY") or "your-openai-key-here"

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Чтение и кодирование изображения в base64
    image_bytes = await file.read()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Отправка в GPT‑4V
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Что на этом фото? Скажи приблизительное название блюда и его КБЖУ (калории, белки, жиры, углеводы) в формате JSON."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )

        # Парсим ответ
        result_text = response.choices[0].message["content"]

        return JSONResponse(content={"result": result_text})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
