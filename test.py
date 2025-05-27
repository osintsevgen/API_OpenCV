# import tensorrt as trt
# print(trt.__version__)  # Например: 8.6.1

# import onnxruntime as ort
# sess = ort.InferenceSession('triton/models/classifier_onnx/1/model.onnx')
# print('Модель загружена. Входы:', sess.get_inputs()[0].name)

# import tensorrt as trt
#
# logger = trt.Logger(trt.Logger.VERBOSE)
# builder = trt.Builder(logger)
# network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
# parser = trt.OnnxParser(network, logger)
#
# with open("triton/models/classifier_onnx/1/model.onnx", "rb") as f:
#     if not parser.parse(f.read()):
#         for error in range(parser.num_errors):
#             print("ERROR:", parser.get_error(error))
#         raise ValueError("ONNX parsing failed")
#
# config = builder.create_builder_config()
# config.set_flag(trt.BuilderFlag.FP16)
# config.max_workspace_size = 2 << 30  # 2GB
#
# engine = builder.build_engine(network, config)
# with open("triton/models/classifier_trt/1/model.plan", "wb") as f:
#     f.write(engine.serialize())

# import tensorrt as trt
#
# # 1. Инициализация
# logger = trt.Logger(trt.Logger.VERBOSE)
# builder = trt.Builder(logger)
# network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
#
# # 2. Парсинг ONNX
# parser = trt.OnnxParser(network, logger)
# with open("triton/models/classifier_onnx/1/model.onnx", "rb") as f:
#     if not parser.parse(f.read()):
#         for error in range(parser.num_errors):
#             print("Ошибка парсинга ONNX:", parser.get_error(error))
#         raise ValueError("Не удалось распарсить ONNX")
#
# # 3. Конфигурация (для TensorRT 10.x)
# config = builder.create_builder_config()
#
# # Установка лимита памяти (новый API)
# config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
#
# # Включение FP16 (если поддерживается)
# if builder.platform_has_fast_fp16:
#     config.set_flag(trt.BuilderFlag.FP16)
#
# # 4. Построение и сохранение движка
# try:
#     serialized_engine = builder.build_serialized_network(network, config)
#     with open("model.plan", "wb") as f:
#         f.write(serialized_engine)
#     print("Успешно создан model.plan!")
# except Exception as e:
#     print("Ошибка сборки:", str(e))


import tensorrt as trt

logger = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# 1. Парсинг ONNX
parser = trt.OnnxParser(network, logger)
with open("classifier_trt/model.onnx", "rb") as f:
    if not parser.parse(f.read()):
        for error in range(parser.num_errors):
            print("ONNX Error:", parser.get_error(error))
        raise ValueError("ONNX parsing failed")

config = builder.create_builder_config()

# 2. Отключение TF32 если не поддерживается
if builder.platform_has_tf32:
    config.set_flag(trt.BuilderFlag.TF32)
else:
    print("TF32 не поддерживается, используется FP32")

# 3. Настройка optimization profile для динамических входов
profile = builder.create_optimization_profile()
input_name = network.get_input(0).name

# Укажите ожидаемые размеры (подстройте под вашу модель!)
profile.set_shape(
    input_name,
    min=(1, 3, 224, 224),  # Минимальный размер
    opt=(8, 3, 224, 224),  # Оптимальный размер
    max=(32, 3, 224, 224)  # Максимальный размер
)
config.add_optimization_profile(profile)

# 4. Настройка памяти
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB

# 5. Построение движка
try:
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build engine")

    with open("classifier_trt/1/model.plan", "wb") as f:
        f.write(serialized_engine)
    print("Модель успешно сконвертирована в model.plan")

except Exception as e:
    print("Ошибка сборки:", str(e))
    # Дополнительная диагностика
    print("Проверьте:")
    print("- Совместимость операторов ONNX с TensorRT")
    print("- Достаточность GPU памяти (nvidia-smi)")
    print("- Корректность входных размеров")