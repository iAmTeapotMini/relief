import os
import time

import cv2
import numpy as np
import shutil
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model

import settings
from core.utils import Log


def run_inference(input_dir=None, output_dir=None, model_path=None):
    """
    Загружает модель .keras, делает предсказания для каждого тайла из input_dir,
    раскрашивает маски по классам и сохраняет в output_dir.
    """
    input_dir = input_dir or settings.TILES_DIR
    output_dir = output_dir or settings.PREDICTED_TILES_DIR
    model_path = model_path or settings.MODEL_PATH

    if not os.path.exists(model_path):
        Log.error(f"Модель не найдена по пути: {model_path}")
        return -1

    if not os.path.exists(input_dir):
        Log.error(f"Папка с тайлами не найдена: {input_dir}")
        return -1

    files = [f for f in os.listdir(input_dir) if f.startswith("tile_")]
    if not len(files):
        Log.error("Нет фрагментов для обработки.")
        return -1

    # Подготовка папки для предсказаний
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    Log.info("Загрузка нейронной сети в память")
    try:
        model = load_model(model_path, compile=False)  # compile=False ускоряет загрузку для инференса
    except Exception as e:
        Log.error(f"Ошибка при загрузке модели: {e}")
        return -1

    # === СОЗДАНИЕ ПАЛИТРЫ (LUT) ДЛЯ БЫСТРОЙ РАСКРАСКИ ===
    # Находим максимальный индекс класса (в твоем случае 14)
    max_class_id = max(settings.CLASS_TO_COLOR.keys())

    # Создаем пустую палитру (черный цвет по умолчанию)
    color_lut = np.zeros((max_class_id + 1, 3), dtype=np.uint8)

    # Заполняем палитру цветами из словаря
    for class_id, color_rgb in settings.CLASS_TO_COLOR.items():
        color_lut[class_id] = color_rgb

    Log.info(f"Начало инференса ({len(files)} фрагментов)...")
    time.sleep(0.1)

    # Используем tqdm для красивого прогресс-бара
    for filename in tqdm(files, desc="Предсказание тайлов", unit="шт"):
        filepath = os.path.join(input_dir, filename)

        # 1. Загрузка и предобработка изображения
        img = cv2.imread(filepath)

        # Переводим BGR (OpenCV) -> RGB (Нейросеть)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Нормализация (если твоя модель училась на данных 0..1)
        # Если модель училась на 0..255, убери деление на 255.0
        img_input = img_rgb.astype(np.float32) / 255.0

        # Добавляем размерность батча: (512, 512, 3) -> (1, 512, 512, 3)
        img_input = np.expand_dims(img_input, axis=0)

        # 2. Инференс (verbose=0 отключает вывод логов на каждом шаге, чтобы не ломать tqdm)
        prediction = model.predict(img_input, verbose=0)

        # 3. Постобработка предсказания
        # Обычно U-Net выдает тензор (1, 512, 512, num_classes) с вероятностями.
        # Берем индекс класса с максимальной вероятностью (argmax).
        class_mask = np.argmax(prediction[0], axis=-1)  # Результат: матрица (512, 512) с числами 0, 1, 2, 12, 14

        # 4. Раскраска маски через Numpy LUT (очень быстрая операция)
        colored_mask_rgb = color_lut[class_mask]

        # 5. Сохранение результата
        # Переводим обратно RGB -> BGR для корректного сохранения через OpenCV
        colored_mask_bgr = cv2.cvtColor(colored_mask_rgb, cv2.COLOR_RGB2BGR)

        output_filepath = os.path.join(output_dir, filename)
        cv2.imwrite(output_filepath, colored_mask_bgr)

    Log.success("Инференс завершен! Все фрагменты обработаны и раскрашены.")
    return 0