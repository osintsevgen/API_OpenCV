import time
from locust import HttpUser, task, between

# Путь к файлу изображения
IMAGE_PATH = "test_images/1.jpg"
# URL вашего API
API_URL = "/predict/"
# Имя модели для использования
MODEL_NAME = "classifier_trt"

# Предзагрузка файла изображения
with open(IMAGE_PATH, "rb") as image_file:
    IMAGE_CONTENT = image_file.read()

class ApiUser(HttpUser):
    wait_time = between(0.5, 1)  # Время ожидания между задачами (в секундах)

    @task
    def predict(self):
        try:
            files = {
                "file": ("1.jpg", IMAGE_CONTENT, "image/jpeg")  # Используем предзагруженное содержимое
            }
            params = {
                "model_name": MODEL_NAME  # Параметр model_name
            }
            # Отправляем POST-запрос
            self.client.post(API_URL, params=params, files=files)
        except Exception as e:
            print(f"Error during request: {e}")