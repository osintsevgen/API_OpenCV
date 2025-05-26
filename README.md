# Triton example

Запуск кода:
```
docker compose up -d
uvicorn api:app --reload --port 5000
```
После запуска документция к API доступна тут - http://127.0.0.1:5000/docs

---
Чтобы попасть в терминал trtexec_container:
```
docker exec -it trtexec_container bash
```
Команда для ONNX -> TensorRT конвертации:
```
trtexec --onnx=model.onnx --saveEngine=model.plan --minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 --maxShapes=input:16x3x224x224 --fp16 --useSpinWait --outputIOFormats=fp16:chw --inputIOFormats=fp16:chw
```
---
Для тестирования производительности вашего API с использованием инструмента Locust:
```
locust -f test_locust.py --host=http://127.0.0.1:5000
```
Прочие файлы для тестов: test_async.py и test_usual.py

---
YouTube-туториал по этому репозиторию доступен по ссылке - [**видео**](https://youtu.be/ljqyuDxd_H0?si=Vpi4PiGrmHKSbKqg)
