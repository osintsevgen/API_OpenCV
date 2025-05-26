import requests
import time

# Путь к файлу изображения
IMAGE_PATH = "test_images/1.jpg"
# URL вашего API
API_URL = "http://127.0.0.1:5000/predict/"
# Имя модели для использования
MODEL_NAME = "classifier_trt"

def send_request(image_path, model_name):
    """Отправляет POST-запрос к API с использованием multipart/form-data."""
    with open(image_path, "rb") as image_file:
        files = {
            "file": ("1.jpg", image_file, "image/jpeg")  # Файл изображения
        }
        params = {
            "model_name": model_name  # Параметр model_name
        }
        response = requests.post(API_URL, params=params, files=files)
        return response.json()

if __name__ == "__main__":
    start_time = time.time()
    for i in range(1500):
        try:
            # Отправляем запрос
            response = send_request(IMAGE_PATH, MODEL_NAME)
        except Exception as e:
            print(f"Error during request {i + 1}: {e}")
    elapsed_time = time.time() - start_time
    print(f"All requests completed in {elapsed_time:.2f} seconds.")