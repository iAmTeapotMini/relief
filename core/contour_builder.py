import cv2
import numpy as np
from shapely import MultiPoint, concave_hull

import settings
from core.utils import Log


def extract_map_boundary(input_path, output_path):
    """
    Находит контур, расширяет его на expand_pixels,
    заливает всё снаружи чёрным цветом и сохраняет результат.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    h, w, c = img.shape
    colors = [settings.RED, settings.PINK, settings.CYAN, settings.BLUE]

    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for color in colors:
        color_mask = cv2.inRange(img, np.array(color), np.array(color))
        combined_mask = cv2.bitwise_or(combined_mask, color_mask)

    y_indices, x_indices = np.where(combined_mask == 255)
    all_points = np.column_stack((x_indices, y_indices))

    if len(all_points) == 0:
        Log.error("Пиксели целевых классов не найдены.")
        return -1

    multi_point = MultiPoint(all_points)
    boundary_polygon = concave_hull(multi_point, ratio=settings.DELAUNAY_RATIO)
    expanded_polygon = boundary_polygon.buffer(settings.EXPAND_PIXELS, join_style=2)
    object_mask = np.zeros((h, w), dtype=np.uint8)

    if expanded_polygon.geom_type == 'Polygon':
        polygon_coords = np.array(expanded_polygon.exterior.coords, dtype=np.int32)
        cv2.drawContours(object_mask, [polygon_coords], -1, 255, thickness=cv2.FILLED)

    elif expanded_polygon.geom_type == 'MultiPolygon':
        for poly in expanded_polygon.geoms:
            polygon_coords = np.array(poly.exterior.coords, dtype=np.int32)
            cv2.drawContours(object_mask, [polygon_coords], -1, 255, thickness=cv2.FILLED)

    result_img = np.zeros_like(img)
    cv2.bitwise_and(img, img, dst=result_img, mask=object_mask)

    cv2.imwrite(output_path, result_img)
    Log.info(f"Изображение успешно обработано и сохранено в: {output_path}")
    return 0

