from fastapi import FastAPI, File, UploadFile, Query
from PIL import Image
import base64
import io
import asyncio
from fastapi import HTTPException
from infer_triton import InferenceModule  # Импортируем ваш модуль инференса

app = FastAPI()

# Load ImageNet class names
with open('imagenet_classes.txt', encoding='utf-8') as f:
    class_names = [line.strip() for line in f.readlines()]

inference_module = InferenceModule()  # Создаем экземпляр модуля инференса

@app.post("/predict/", description="Выполняет классификацию изображения с использованием указанной модели.")
async def predict(
    file: UploadFile = File(..., description="Изображение для классификации."), 
    model_name: str = Query(..., description="Имя модели тритона для использования")
):
    """
    Выполнить классификацию изображения.

    Args:
        file (UploadFile): Загружаемое изображение.
        model_name (str): Имя модели для использования в инференсе.

    Returns:
        dict: Название класса и значение логита.
    """
    try:
        # Конвертация загруженного файла в base64
        contents = await file.read()
        img_base64 = base64.b64encode(contents).decode("utf-8")

        # Выполнение инференса с использованием указанной модели
        result = await inference_module.infer_image(img_base64, model_name=model_name)

        # Получение ID класса и логита
        class_id = result["class_id"]
        logit = round(result["logit"], 3)  # Округление логита до 3 знаков после запятой

        # Получение названия класса из списка ImageNet
        class_name = class_names[class_id]

        return {
            "class_name": class_name,
            "logit": logit
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")