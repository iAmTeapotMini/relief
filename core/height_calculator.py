import cv2
import networkx as nx
import numpy as np
from skimage import morphology

import settings
from core.utils import colorToMask, Log


def assign_heights_to_lines(main_image_path, aux_lines_path, output_path):
    """
    Выполняет этапы 4.4, 4.5 и 4.7:
    - Построение топологического графа местности.
    - Вычисление относительных высот.
    - Присвоение значений высот (в оттенках серого) линиям каркаса.
    """

    Log.step("Этапы 4.4 - 4.7: Топологический граф и расчет высот")

    img = cv2.imread(main_image_path)
    if img is None:
        Log.error(f"Не удалось загрузить основное изображение: {main_image_path}")
        return -1

    # Загружаем перевернутые в BGR цвета из настроек
    RED = settings.RED[::-1]
    BLUE = settings.BLUE[::-1]
    BLACK = settings.BLACK[::-1]

    # Маски для цветов
    red_mask = colorToMask(img, color=RED)
    blue_mask = colorToMask(img, color=BLUE)
    black_mask = colorToMask(img, color=BLACK)

    bergs_lines_mask = red_mask | blue_mask
    lines_mask = red_mask

    # ==========================================
    # ЭТАП 4.4. ПОСТРОЕНИЕ ГРАФА
    # ==========================================
    Log.info("Поиск связных областей и построение графа...")
    num_areas, areas, stats, centroids = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(lines_mask | black_mask), connectivity=4
    )

    G = nx.DiGraph()
    G.add_nodes_from(range(1, num_areas))

    kernel = np.ones((5, 5), np.uint8)
    # Массив масок для каждого региона
    areas_mask = [(areas == l).astype(np.uint8) * 255 for l in range(1, num_areas)]
    extended_areas_mask = [cv2.dilate(m, kernel, iterations=1) for m in areas_mask]

    dist_threshold = 1
    for label in range(1, num_areas):
        mask = areas_mask[label - 1]
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        for other_label in range(label + 1, num_areas):
            other_mask = areas_mask[other_label - 1]
            min_dist = np.min(dist[other_mask > 0])

            if min_dist <= dist_threshold:
                # Поиск разделяющей линии с бергштрихом
                separating_line = cv2.bitwise_and(extended_areas_mask[label - 1], extended_areas_mask[other_label - 1])
                separating_line = cv2.bitwise_and(separating_line, bergs_lines_mask)
                border_coords = np.where(separating_line == 255)

                for y, x in zip(border_coords[0], border_coords[1]):
                    if blue_mask[y, x] == 255:
                        if mask[y, x] == 255:
                            G.add_edge(label, other_label)
                        else:
                            G.add_edge(other_label, label)
                        break

    # ==========================================
    # ЭТАП 4.5. ВЫЧИСЛЕНИЕ ОТНОСИТЕЛЬНЫХ ВЫСОТ
    # ==========================================
    Log.info("Анализ графа и вычисление высот...")
    dp = {}

    # СТРОГАЯ ПРОВЕРКА НА АЦИКЛИЧНОСТЬ (DAG)
    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        # Если найден цикл, находим его, чтобы показать пользователю
        cycle = nx.find_cycle(G)
        Log.error("КРИТИЧЕСКАЯ ОШИБКА ТОПОЛОГИИ!")
        Log.error("В графе высот найден цикл (парадокс рельефа). Граф не является ацикличным (DAG).")
        Log.error(f"Проблема в этих областях: {cycle}")
        return -1

    for node in reversed(topo_order):
        successors = list(G.successors(node))
        if successors:
            dp[node] = 1 + max(dp[child] for child in successors)
        else:
            dp[node] = 1

    highest_root = max(dp, key=dp.get)
    Log.info(f"Найдена базовая область (корень). Присвоение стартовой высоты: {settings.START_HEIGHT}")

    UG = G.to_undirected()
    red_areas_color = {}

    # Берем шаг из настроек
    step = settings.HEIGHT_STEP
    queue = [(highest_root, settings.START_HEIGHT)]
    visited = {highest_root}

    while queue:
        node, level = queue.pop(0)
        if node not in red_areas_color:
            red_areas_color[node] = level

        for neighbor in UG.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                new_level = level
                if G.has_edge(node, neighbor):
                    new_level += step
                else:
                    new_level -= step
                queue.append((neighbor, new_level))

    Log.info("Конвертация высот в 16-битный формат (uint16)...")

    # Создаем 16-битный одноканальный холст, залитый значением "NO_DATA" (65535)
    line_result_img = np.full((img.shape[0], img.shape[1]), settings.NO_DATA_VALUE, dtype=np.uint16)

    lines_coords = np.where(lines_mask == 255)
    directions = ((-1, 0), (0, 1), (1, 0), (0, -1))

    # 1. Раскраска ОСНОВНЫХ линий
    for y, x in zip(lines_coords[0], lines_coords[1]):
        pixel_neighbors = set()
        for dy, dx in directions:
            y1, x1 = y + dy, x + dx
            if 0 <= x1 < img.shape[1] and 0 <= y1 < img.shape[0]:
                label = areas[y1, x1]
                if label != 0:
                    pixel_neighbors.add(label)

        pixel_neighbors = tuple(pixel_neighbors)

        if len(pixel_neighbors) == 1:
            color_value = red_areas_color.get(pixel_neighbors[0], settings.NO_DATA_VALUE)
            line_result_img[y, x] = color_value
        elif len(pixel_neighbors) == 2:
            node_color1 = red_areas_color.get(pixel_neighbors[0], settings.NO_DATA_VALUE)
            node_color2 = red_areas_color.get(pixel_neighbors[1], settings.NO_DATA_VALUE)
            if node_color1 != settings.NO_DATA_VALUE and node_color2 != settings.NO_DATA_VALUE:
                color_value = (node_color1 + node_color2) // 2
                line_result_img[y, x] = color_value

    # 2. ИНТЕГРАЦИЯ ВСПОМОГАТЕЛЬНЫХ ЛИНИЙ (Скелета с этапа 4.6)
    skeleton_img = cv2.imread(aux_lines_path)
    if skeleton_img is None:
        Log.error(f"Не удалось загрузить скелет вспомогательных линий: {aux_lines_path}")
        return -1

    PINK_BGR = settings.PINK[::-1]
    PURPLE_BGR = settings.PURPLE[::-1]
    ORANGE_BGR = settings.ORANGE[::-1]
    RED_BGR = settings.RED[::-1]

    pink_result = colorToMask(skeleton_img, PINK_BGR)
    purple_result = colorToMask(skeleton_img, PURPLE_BGR)
    orange_result = colorToMask(skeleton_img, ORANGE_BGR)
    red_skel_result = colorToMask(skeleton_img, RED_BGR)

    quarter_step = settings.HEIGHT_STEP // 4

    # Функция для безопасного ограничения высоты (чтобы не уйти за 16-бит)
    def clip_16bit(val):
        return max(0, min(65534, val))  # 65535 зарезервировано под фон

    # Розовые
    pink_coords = np.where(pink_result > 0)
    for y, x in zip(pink_coords[0], pink_coords[1]):
        area_label = areas[y, x]
        if area_label != 0 and area_label in red_areas_color:
            line_result_img[y, x] = clip_16bit(red_areas_color[area_label])

    # Фиолетовые (+ четверть шага)
    purple_coords = np.where(purple_result > 0)
    for y, x in zip(purple_coords[0], purple_coords[1]):
        area_label = areas[y, x]
        if area_label != 0 and area_label in red_areas_color:
            line_result_img[y, x] = clip_16bit(red_areas_color[area_label] + quarter_step)

    # Оранжевые (- четверть шага)
    orange_coords = np.where(orange_result > 0)
    for y, x in zip(orange_coords[0], orange_coords[1]):
        area_label = areas[y, x]
        if area_label != 0 and area_label in red_areas_color:
            line_result_img[y, x] = clip_16bit(red_areas_color[area_label] - quarter_step)

    # Красные (среднее)
    red_skel_coords = np.where(red_skel_result > 0)
    for y, x in zip(red_skel_coords[0], red_skel_coords[1]):
        neighbor_labels = set()
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny1, nx1 = y + dy, x + dx
                if 0 <= ny1 < areas.shape[0] and 0 <= nx1 < areas.shape[1]:
                    lbl = areas[ny1, nx1]
                    if lbl != 0 and lbl in red_areas_color:
                        neighbor_labels.add(lbl)

        if len(neighbor_labels) >= 2:
            labels_list = list(neighbor_labels)[:2]
            color_value = (red_areas_color[labels_list[0]] + red_areas_color[labels_list[1]]) // 2
            line_result_img[y, x] = clip_16bit(color_value)

    # 3. Границы карты (черный скелет). Присваиваем им высоту 0 (дно)
    black_skeleton = morphology.skeletonize(black_mask > 0)
    line_result_img[black_skeleton] = 0

    # Сохраняем 16-битный одноканальный PNG
    success = cv2.imwrite(output_path, line_result_img)
    if success:
        Log.success(f"16-битные высоты присвоены! Изображение сохранено: {output_path}")
        return 0
    return -1