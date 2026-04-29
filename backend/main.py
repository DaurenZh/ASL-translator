import base64
import io

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from model import ASLClassifier


app = FastAPI(title="ASL Translator API")
classifier = ASLClassifier()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+):5173",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    image: str


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_ready": classifier.ready,
        "classes": classifier.classes,
    }


@app.post("/predict")
def predict(payload: PredictionRequest):
    try:
        image_data = payload.image
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        raw = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image") from exc

    return classifier.predict(image)
