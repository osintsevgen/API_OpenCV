import torch
import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# from PIL import Image
# # import torch_tensorrt
#
# # Load ImageNet class names
# with open('imagenet_classes.txt') as f:  # импортируем имена классов из файла imagenet_classes.txt
#     class_names = [line.strip() for line in f.readlines()]
#
# # Load pre-trained ResNet152 with correct API
# model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)  # предобученная модель ResNet 152 слоя
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# model.eval()
#
# # Image preprocessing\n",
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
#
#
# # Загрузите изображение для инференса
# image_path = 'test_images/3.jpg'  # Замените на путь к вашему изображению
# image = Image.open(image_path)
# image_tensor = preprocess(image)
# image_tensor = image_tensor.unsqueeze(0).to(device)  # Добавляем размер батча
#
# # Прогоните изображение через модель
# with torch.no_grad():
#     output = model(image_tensor)
#
# # Примените softmax для получения вероятностей классов\n",
# probabilities = output.cpu()
#
# # Отображение результатов
# top_prob, top_class = torch.topk(probabilities, 1)
# top_prob = top_prob.item()
# top_class = top_class.item()
#
# # Получить имя класса
# class_name = class_names[top_class]
#
# # Отобразить изображение
# plt.imshow(image)
# plt.axis('off')
# plt.title(f'Predicted: {class_name} ({top_prob*100:.3f}%)')
# plt.show()
# #

import torch.nn as nn

# Load pre-trained ResNet152 with correct API
model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)

# Add Softmax layer at the end
model = nn.Sequential(
    model,  # Original ResNet152
    nn.Softmax(dim=1)  # Adds Softmax to convert outputs to probabilities
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

#
# Экспорт модели в формат ONNX
dummy_input = torch.randn(1, 3, 224, 224, device=device)  # Пример входного тензора
# onnx_file_path = "triton/models/classifier_onnx/1/model.onnx"  # Путь для сохранения ONNX файла
onnx_file_path = "triton/models/classifier_onnx/1/model.onnx"  # Путь для сохранения ONNX файла

# Экспорт модели\n",
with torch.no_grad():
    torch.onnx.export(
        model,  # Модель PyTorch
        dummy_input,  # Пример входных данных
        onnx_file_path,  # Путь для сохранения ONNX файла
        export_params=True,  # Экспортировать обученные параметры
        opset_version=14,  # Версия ONNX операторов
        do_constant_folding=True,  # Оптимизация констант
        input_names=["input"],  # Имя входного тензора
        output_names=["output"],  # Имя выходного тензора
        dynamic_axes={
            "input": {0: "batch_size"},  # Динамический размер батча
            "output": {0: "batch_size"},
        },
    )

print(f"Модель успешно экспортирована в файл: {onnx_file_path}")

#
# import tensorrt as trt
#
# logger = trt.Logger(trt.Logger.WARNING)
# builder = trt.Builder(logger)
# network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
#
# # Парсинг ONNX
# parser = trt.OnnxParser(network, logger)
# with open("triton/models/classifier_onnx/1/model.onnx", "rb") as f:
#     if not parser.parse(f.read()):
#         for error in range(parser.num_errors):
#             print(parser.get_error(error))
#         raise ValueError("Failed to parse ONNX")
#
# config = builder.create_builder_config()
# config.set_flag(trt.BuilderFlag.FP16)
#
# # Универсальная настройка памяти
# if hasattr(config, 'max_workspace_size'):  # Для старых версий (<8.4)
#     config.max_workspace_size = 2 << 30  # 2GB
# elif hasattr(config, 'memory_pool_limits'):  # Для версий 8.4+
#     config.memory_pool_limits = {trt.MemoryPoolType.WORKSPACE: 2 << 30}
# else:  # Резервный вариант
#     print("Warning: Не удалось установить лимит памяти - используем значения по умолчанию")
#
# # Построение движка
# engine = builder.build_engine(network, config)
# with open("classifier_trt/1/model.plan", "wb") as f:
#     f.write(engine.serialize())
#

