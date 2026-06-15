import os
import re
import shutil

import cv2
import numpy as np

import settings
from core.utils import Log


def resize_with_pad(input_path, output_path=None):
    """
    Загружает изображение, пропорционально вписывает в целевые размеры (2048x1536)
    с добавлением базового фона. Это первый этап стандартизации.
    """
    if output_path is None:
        output_path = os.path.join(settings.TEMP_DIR, 'resize_image.png')

    if not os.path.exists(input_path):
        Log.error(f"Файл не найден: {input_path}")
        return -1

    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        Log.error(f"Не удалось прочитать файл (возможно, поврежден): {input_path}")
        return -1

    h, w = image.shape[:2]
    target_width, target_height = settings.TARGET_WIDTH, settings.TARGET_HEIGHT

    if h > w:
        Log.info(f"Изображение вертикальное ({w}x{h}). Выполняю поворот на 90 градусов...")
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        h, w = image.shape[:2]

    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    if len(image.shape) >= 3:
        canvas = np.full((target_height, target_width, image.shape[2]), settings.PAD_COLOR, dtype=np.uint8)
    else:
        color = settings.PAD_COLOR[0] if isinstance(settings.PAD_COLOR, tuple) else settings.PAD_COLOR
        canvas = np.full((target_height, target_width), color, dtype=np.uint8)

    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

    success = cv2.imwrite(output_path, canvas)

    if success:
        Log.info(f"Изображение приведено к {target_width}x{target_height} и сохранено: {output_path}")
        return 0
    else:
        Log.error(f"Не удалось сохранить файл: {output_path}")
        return -1


def slice_image_into_tiles(input_path, output_dir=None):
    """
    Нарезает изображение строгого размера (1024x1024) с применением зеркального Padding'а.
    """
    if output_dir is None:
        output_dir = settings.TILES_DIR

    if not os.path.exists(input_path):
        Log.error(f"Файл для нарезки не найден: {input_path}")
        return -1

    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        Log.error(f"Не удалось прочитать изображение: {input_path}")
        return -1

    h, w = image.shape[:2]
    tile_size = settings.TILE_SIZE  # 1024
    step = settings.OVERLAP  # 768
    pad = settings.PAD  # 128

    # ДИНАМИЧЕСКИЙ PADDING:
    # Слева и сверху мы всегда добавляем ровно PAD (128).
    # Справа и снизу мы добавляем PAD + остаток, чтобы ширина и высота
    # стали ИДЕАЛЬНО кратны нашему шагу (768). Это гарантирует, что мы не потеряем крайние пиксели!
    pad_top = pad
    pad_left = pad
    pad_bottom = (step - (h % step)) % step + pad
    pad_right = (step - (w % step)) % step + pad

    # ВОТ ОНО! Зеркальное отражение границ, о котором мы писали в дипломе
    padded_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
    padded_h, padded_w = padded_image.shape[:2]

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    tiles_count = 0

    # Скользим окном по УЖЕ РАСШИРЕННОМУ изображению
    for y in range(0, padded_h - tile_size + 1, step):
        for x in range(0, padded_w - tile_size + 1, step):
            tile = padded_image[y: y + tile_size, x: x + tile_size]

            # Сохраняем абсолютные координаты расширенной карты
            filename = f"tile_y{y}_x{x}.png"
            filepath = os.path.join(output_dir, filename)

            cv2.imwrite(filepath, tile)
            tiles_count += 1

    Log.success(f"Изображение расширено зеркальными полями и нарезано на {tiles_count} фрагментов.")
    return 0


def stitch_patches(input_dir, output_path=None):
    """
    Выполняет обратную сшивку предсказанных масок.
    Алгоритм: берет тайл 1024x1024, отрезает по 128 пикселей "слепой зоны" с каждой стороны,
    оставшийся центр 768x768 вклеивает в холст.
    В конце обрезает искусственные поля, возвращая оригинальный размер.
    """
    if output_path is None:
        output_path = os.path.join(settings.TEMP_DIR, 'stitched_mask.png')

    if not os.path.exists(input_dir):
        Log.error(f"Папка с фрагментами не найдена: {input_dir}")
        return -1

    files = [f for f in os.listdir(input_dir) if f.startswith("tile_")]
    if not files:
        Log.error(f"В папке {input_dir} нет файлов для сшивки!")
        return -1

    orig_h = settings.TARGET_HEIGHT  # 1536
    orig_w = settings.TARGET_WIDTH  # 2048
    tile_size = settings.TILE_SIZE  # 1024
    step = settings.STEP  # 768
    pad = settings.PAD  # 128

    # Восстанавливаем размеры расширенного холста (такие же, как были при нарезке)
    pad_bottom = (step - (orig_h % step)) % step + pad
    pad_right = (step - (orig_w % step)) % step + pad

    padded_h = orig_h + pad + pad_bottom
    padded_w = orig_w + pad + pad_right

    # Читаем первый файл для определения каналов
    first_tile = cv2.imread(os.path.join(input_dir, files[0]), cv2.IMREAD_UNCHANGED)
    if len(first_tile.shape) >= 3:
        canvas_padded = np.zeros((padded_h, padded_w, first_tile.shape[2]), dtype=np.uint8)
    else:
        canvas_padded = np.zeros((padded_h, padded_w), dtype=np.uint8)

    pattern = re.compile(r"tile_y(\d+)_x(\d+)")
    tiles_stitched = 0

    for filename in files:
        match = pattern.search(filename)
        if not match:
            continue

        # Эти x и y - это координаты тайла на РАСШИРЕННОМ холсте
        y = int(match.group(1))
        x = int(match.group(2))

        tile = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_UNCHANGED)

        # Вырезаем ИДЕАЛЬНЫЙ ЦЕНТР (768x768). Отрезаем по 128 пикселей со всех сторон.
        center_crop = tile[pad: tile_size - pad, pad: tile_size - pad]

        # Вклеиваем этот центр на расширенный холст
        # Поскольку мы отрезали pad, координаты на холсте тоже сдвигаются на pad!
        canvas_padded[y + pad: y + pad + step, x + pad: x + pad + step] = center_crop

        tiles_stitched += 1

    # САМЫЙ КРАСИВЫЙ ШАГ:
    # Обрезаем те самые зеркальные поля, которые мы добавляли перед нарезкой.
    # Мы возвращаемся к строго оригинальному размеру 2048x1536!
    final_mask = canvas_padded[pad: pad + orig_h, pad: pad + orig_w]

    success = cv2.imwrite(output_path, final_mask)

    if success:
        Log.success(f"Успешно сшито {tiles_stitched} фрагментов.")
        Log.info(f"Итоговая маска обрезана до оригинальных {orig_w}x{orig_h} и сохранена: {output_path}")
        return 0
    else:
        Log.error(f"Не удалось сохранить сшитое изображение: {output_path}")
        return -1