from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.special import expit   # sigmoid — needed for PAC confidence
import pickle
import re
import os

app = FastAPI(title="Fake News Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model & vectorizer on startup ──────────────────────────────────────
# Model:      PassiveAggressiveClassifier  (fake_news_model_.pkl)
# Vectorizer: TfidfVectorizer              (tfidf_vectorizer_.pkl)
MODEL_PATH      = os.getenv("MODEL_PATH",      "fake_news_model_.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "tfidf_vectorizer_.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    print("✅ Model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    raise RuntimeError(f"Model file not found: {e}")


# ── Same clean_text used during training ─────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Request / Response schemas ────────────────────────────────────────────────
class ArticleRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    verdict: str          # "FAKE" or "REAL"
    confidence: float     # 0.0 – 1.0
    label: int            # 0 = real, 1 = fake


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Fake News Detector API is running."}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: ArticleRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    cleaned    = clean_text(req.text)
    vectorized = vectorizer.transform([cleaned])
    prediction = int(model.predict(vectorized)[0])

    # PassiveAggressiveClassifier has no predict_proba.
    # Use decision_function score → sigmoid → confidence value.
    score      = float(model.decision_function(vectorized)[0])
    raw_conf   = float(expit(score))           # 0–1 via sigmoid
    # When pred=0 (REAL), flip so confidence reflects the actual class
    confidence = raw_conf if prediction == 1 else 1 - raw_conf
    verdict    = "FAKE" if prediction == 1 else "REAL"

    return PredictionResponse(
        verdict=verdict,
        confidence=round(confidence, 4),
        label=prediction,
    )
