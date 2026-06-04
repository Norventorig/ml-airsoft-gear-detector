from pydantic import BaseModel, HttpUrl
from typing import List


class PhotoItem(BaseModel):
    photo_id: str
    url: HttpUrl


class PostPredictRequest(BaseModel):
    post_id: str
    text: str
    photos: List[PhotoItem]


class PredictionItem(BaseModel):
    object_id: str
    category: str
    subcategory: str
    confidence: float
    photo_ids: List[str]


class PostPredictResponse(BaseModel):
    post_id: str
    predictions: List[PredictionItem]
