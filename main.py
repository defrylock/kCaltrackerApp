from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import openai
import base64
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY") or "your-openai-key-here"

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Читаем файл и кодируем в base64
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Запрос к GPT‑4V
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Что на этом фото? Назови блюдо и приблизительное КБЖУ (калории, белки, жиры, углеводы) в формате JSON."},
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

        result_text = response.choices[0].message["content"]
        return JSONResponse(content={"result": result_text})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
