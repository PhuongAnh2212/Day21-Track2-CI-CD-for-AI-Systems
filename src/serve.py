from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

GCS_BUCKET = os.environ["GCS_BUCKET"]
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """
    Tải file model.pkl từ GCS về máy khi server khởi động.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_MODEL_KEY)

    if not blob.exists():
        raise FileNotFoundError(
            f"Model not found in GCS: gs://{GCS_BUCKET}/{GCS_MODEL_KEY}"
        )

    blob.download_to_filename(MODEL_PATH)
    print(f"Model downloaded from gs://{GCS_BUCKET}/{GCS_MODEL_KEY} to {MODEL_PATH}")


download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400,
            detail="Expected 12 features (wine quality)",
        )

    pred = model.predict([req.features])[0]
    pred_int = int(pred)

    labels = {
        0: "thap",
        1: "trung_binh",
        2: "cao",
    }

    return {
        "prediction": pred_int,
        "label": labels.get(pred_int, "unknown"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)