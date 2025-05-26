import aiohttp
import asyncio
import time

# Путь к файлу изображения
IMAGE_PATH = "test_images/1.jpg"
# URL вашего API
API_URL = "http://127.0.0.1:5000/predict/"
# Имя модели для использования
MODEL_NAME = "classifier_trt"

# Предзагрузка файла изображения
with open(IMAGE_PATH, "rb") as image_file:
    IMAGE_CONTENT = image_file.read()

async def send_request(session, image_content, model_name):
    """Отправляет асинхронный POST-запрос к API с предзагруженным файлом."""
    # Создаем FormData для файловых данных
    form_data = aiohttp.FormData()
    form_data.add_field("file", image_content, filename="1.jpg", content_type="image/jpeg")
    
    # Добавляем параметр model_name в query string
    url_with_params = f"{API_URL}?model_name={model_name}"

    try:
        async with session.post(url_with_params, data=form_data) as response:
            if response.status == 200:
                result = await response.json()
                return result
            else:
                print(f"Request failed with status {response.status}, body: {await response.text()}")
    except Exception as e:
        print(f"Error during request: {e}")

async def main():
    """Основная функция для отправки 1500 асинхронных запросов."""
    tasks = []
    async with aiohttp.ClientSession() as session:
        for i in range(1500):
            task = asyncio.create_task(send_request(session, IMAGE_CONTENT, MODEL_NAME))
            tasks.append(task)
        
        # Ожидаем завершения всех задач
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Обработка результатов
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                print(f"Error in request {i + 1}: {response}")


if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    elapsed_time = time.time() - start_time
    print(f"All requests completed in {elapsed_time:.2f} seconds.")