from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class SentimentEnum(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class AspectsModel(BaseModel):
    delivery: Optional[SentimentEnum] = None
    price: Optional[SentimentEnum] = None
    quality: Optional[SentimentEnum] = None
    functionality: Optional[SentimentEnum] = None
    service: Optional[SentimentEnum] = None

class ReviewAnalysisModel(BaseModel):
    sentiment: SentimentEnum
    aspects: AspectsModel
    summary: str = Field(description="Краткая выжимка отзыва")

# Модель для входного запроса
class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Текст отзыва для анализа")