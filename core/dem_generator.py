import cv2
import numpy as np
from scipy.interpolate import griddata

import settings
from core.utils import Log


def generate_heightmap(image_path, output_path):
    """
    Этап 4.8. Генерация 16-битной карты высот (uint16).
    """
    # ВАЖНО: флаг cv2.IMREAD_ANYDEPTH позволяет OpenCV прочитать 16-битный файл "как есть"
    gray_img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)

    if gray_img is None:
        Log.error(f"Ошибка: Изображение для интерполяции не найдено: {image_path}")
        return -1

    Log.info("Подготовка данных для 16-битной интерполяции ЦМР...")
    height, width = gray_img.shape

    # Ищем пиксели, которые НЕ являются фоном (NO_DATA_VALUE)
    lines_mask = (gray_img != settings.NO_DATA_VALUE)

    y_coords, x_coords = np.where(lines_mask)
    points = np.column_stack([x_coords, y_coords])
    values = gray_img[lines_mask]

    if len(points) == 0:
        Log.error("На изображении не найдено изолиний для интерполяции!")
        return -1

    Log.info(f"Найдено {len(points)} опорных точек. Запуск алгоритма интерполяции...")

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    # 1. Основная интерполяция
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')

    # 2. Обработка краевых пустот (NaN)
    mask_nan = np.isnan(grid_z)
    if np.any(mask_nan):
        Log.info("Заполнение краевых пустот (nearest)...")
        nearest_z = griddata(points, values, (grid_x, grid_y), method='nearest')
        grid_z[mask_nan] = nearest_z[mask_nan]

    # 3. Конвертация в 16-бит (ограничение до 65535)
    grid_z_clipped = np.clip(grid_z, 0, 65535)
    grid_z_uint16 = grid_z_clipped.astype(np.uint16)

    # 4. Медианный фильтр (OpenCV поддерживает ядра 3 и 5 для 16-битных изображений)
    filter_size = getattr(settings, 'MEDIAN_FILTER_SIZE', 5)
    Log.info(f"Применение медианного фильтра (ядро {filter_size}x{filter_size})...")
    final_dem = cv2.medianBlur(grid_z_uint16, filter_size)

    # Сохраняем 16-битный ЦМР
    success = cv2.imwrite(output_path, final_dem)
    Log.info("Создание 8-битной визуализации для презентации...")

    # Находим реальный минимум и максимум на вашей сгенерированной карте
    min_val = final_dem.min()
    max_val = final_dem.max()

    if max_val > min_val:
        # Растягиваем диапазон высот на 0-255 (Нормализация Min-Max)
        dem_8bit = ((final_dem - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    else:
        dem_8bit = np.zeros_like(final_dem, dtype=np.uint8)

    # Формируем имя для визуального файла (добавляем _visual)
    visual_path = output_path.replace(".png", "_visual.png")
    cv2.imwrite(visual_path, dem_8bit)

    Log.success(f"16-битная ЦМР сохранена: {output_path}")
    Log.success(f"8-битная визуализация сохранена: {visual_path}")
    return 0