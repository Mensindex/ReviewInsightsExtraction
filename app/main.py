from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from app.schemas import ReviewRequest, ReviewAnalysisModel
from dotenv import load_dotenv
from app.engine import ABSAPredictor
import os

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    base_model = "unsloth/llama-3-8b-bnb-4bit"
    adapter_path = "models/adapters"
    use_mock = os.getenv("USE_MOCK", "False").lower() == "true"

    print("🔄 Loading AI Model...")
    try:
        app.state.predictor = ABSAPredictor(
            base_model_id=base_model,
            adapter_path=adapter_path,
            mock_mode=use_mock
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        app.state.predictor = None

    yield

    # Логика при выключении
    print("🛑 Cleaning up resources...")
    if hasattr(app.state, "predictor"):
        del app.state.predictor


app = FastAPI(title="E-com Review Insights API", lifespan=lifespan)


@app.post("/analyze", response_model=ReviewAnalysisModel)
async def analyze_review(request: Request, body: ReviewRequest):
    predictor = getattr(request.app.state, "predictor", None)

    if not predictor:
        raise HTTPException(status_code=503, detail="Model is not ready")

    result = predictor.predict(body.text)

    if not result:
        raise HTTPException(status_code=422, detail="Could not extract valid JSON")

    return result
