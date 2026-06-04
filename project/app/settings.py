from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TEXT_MODEL_DIR = BASE_DIR / "models" / "category_text"

CATEGORY_IMAGE_MODEL = (
    BASE_DIR /
    "models" /
    "category_image" /
    "best_image_classifier.pth"
)

SUBCATEGORY_IMAGE_MODEL = (
    BASE_DIR /
    "models" /
    "subcategory_image" /
    "best_model_11_classes.pth"
)

API_KEY = "super-secret-key"