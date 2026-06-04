import os

import requests
from dotenv import load_dotenv
load_dotenv()

url = "http://127.0.0.1:8000/predict"

headers = {
    "X-API-Key": os.getenv("API_KEY"),
    "Content-Type": "application/json"
}

# Тело запроса
data = {
  "post_id": "12345",
  "text": "Продаю новую страйкбольную винтовку ASG и тактический жилет. В комплекте магазин и ремень. Состояние идеальное.",
  "photos": [
    {"photo_id": "1", "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSeLl-pMU0SjQNV8Q3LgngGhJYosh1xkZac4Q&s"},
    {"photo_id": "2", "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTD7T_10aVIgbQmjlf8gvpnYGnRZAajMSBguQ&s"}
  ]
}

# Отправка POST-запроса
response = requests.post(url, json=data, headers=headers)

if response.status_code == 200:
    print("Успешный ответ:")
    print(response.json())
else:
    print(f"Ошибка {response.status_code}:")
    print(response.text)
