from fastapi import FastAPI, Depends
import uvicorn

from project.app.schemas import PostPredictRequest, PostPredictResponse
from project.app.dependencies import verify_api_key
from project.app.models import ml_pipeline

app = FastAPI(
    title="airsoft-detector",
    description="API для классификации страйкбольного оружия и экипировки по тексту и фото",
    version="1.0.0"
)


@app.post(
    "/predict",
    response_model=PostPredictResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Классификация объявления",
    description="Принимает текст объявления и массив ссылок на фото. Возвращает предсказанные категории и подкатегории."
)
async def predict_post(payload: PostPredictRequest):
    photos_data = []
    for photo in payload.photos:
        photos_data.append({
            "photo_id": photo.photo_id,
            "url": str(photo.url)
        })

    raw_predictions = ml_pipeline.predict(
        text=payload.text,
        photos=photos_data
    )

    return PostPredictResponse(
        post_id=payload.post_id,
        predictions=raw_predictions
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
