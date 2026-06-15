import cv2
import numpy as np
import heapq

import settings
from core.utils import colorToMask, Log


def get_line_endpoints_and_directions(binary_mask):
    normalized_mask = (binary_mask > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)

    neighbor_count = cv2.filter2D(normalized_mask, -1, kernel)
    y_indices, x_indices = np.where(neighbor_count == 11)
    endpoints = np.column_stack((x_indices, y_indices))

    all_pixels = np.column_stack(np.where(binary_mask > 0))[:, ::-1]

    valid_endpoints = []
    vectors = []

    for ep in endpoints:
        dists = np.sum((all_pixels - ep) ** 2, axis=1)
        nearby_indices = np.where((dists >= 4) & (dists <= (settings.VECTOR_LOOKBACK ** 2)))

        if len(nearby_indices[0]) > 0:
            inner_pt = all_pixels[nearby_indices[0][np.argmin(dists[nearby_indices[0]])]]
            vec = ep - inner_pt
            norm = np.linalg.norm(vec)
            if norm > 0:
                vectors.append(vec / norm)
                valid_endpoints.append(ep)

    return np.array(valid_endpoints), np.array(vectors)


def grow_direct_line(start_pt, initial_vec, black_mask, obstacle_mask, w, h):
    """
    Продление линии строго по прямой согласно вектору направления (Для Группы А).
    """
    ray_pt = start_pt.astype(float)
    target_pt = None
    reached_target = False

    # Пускаем луч строго по вектору
    for step in range(1, 2000):
        ray_pt += initial_vec
        rx, ry = int(round(ray_pt[0])), int(round(ray_pt[1]))

        # 1. Если вышли за границы или коснулись черной маски (Успех!)
        if rx < 0 or rx >= w or ry < 0 or ry >= h or black_mask[ry, rx] > 0:
            rx = max(0, min(w - 1, rx))
            ry = max(0, min(h - 1, ry))
            target_pt = (rx, ry)
            reached_target = True
            break

        # 2. Если наткнулись на препятствие (например, только что нарисованную линию)
        # Пропускаем первые 5 шагов, чтобы не "врезаться" в самих себя на старте
        if step > 5 and obstacle_mask[ry, rx] > 0:
            # Линия блокирована. Возвращаем False, чтобы она ушла в Группу Б.
            return False, None

    # Если цель успешно достигнута - рисуем идеальную прямую
    if reached_target and target_pt is not None:
        temp_canvas = np.zeros((h, w), dtype=np.uint8)

        start_tuple = (int(round(start_pt[0])), int(round(start_pt[1])))
        end_tuple = (int(round(target_pt[0])), int(round(target_pt[1])))

        # Простое соединение конца линии и границы прямой линией
        cv2.line(temp_canvas, start_tuple, end_tuple, 255, 1)

        return True, temp_canvas

    return False, None


def find_corridor_path(start_pt, initial_vec, black_mask, obstacle_mask, w, h):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    forbidden = cv2.dilate(obstacle_mask, kernel, iterations=1)

    sx, sy = int(round(start_pt[0])), int(round(start_pt[1]))
    cv2.circle(forbidden, (sx, sy), 8, 0, -1)

    safe_area = cv2.bitwise_not(forbidden)
    dist_map = cv2.distanceTransform(safe_area, cv2.DIST_L2, 3)

    queue = []
    heapq.heappush(queue, (0.0, sx, sy))

    costs = np.full((h, w), float('inf'))
    costs[sy, sx] = 0
    came_from = {}

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    target_reached = None
    base_dir_x, base_dir_y = initial_vec[0], initial_vec[1]

    while queue:
        current_cost, cx, cy = heapq.heappop(queue)

        if current_cost > costs[cy, cx]: continue

        if cx <= 0 or cx >= w - 1 or cy <= 0 or cy >= h - 1 or black_mask[cy, cx] > 0:
            target_reached = (cx, cy)
            break

        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy

            if nx < 0 or nx >= w or ny < 0 or ny >= h: continue
            if forbidden[ny, nx] > 0: continue

            step_cost = 1.414 if dx != 0 and dy != 0 else 1.0
            step_len = (dx ** 2 + dy ** 2) ** 0.5
            nx_vec, ny_vec = dx / step_len, dy / step_len

            dot_prod = nx_vec * base_dir_x + ny_vec * base_dir_y
            dir_penalty = (1.0 - dot_prod) * settings.DIRECTIONAL_STIFFNESS
            step_cost += dir_penalty

            clearance = dist_map[ny, nx]
            if clearance < settings.REPULSION_RADIUS:
                step_cost += (settings.REPULSION_RADIUS - clearance) * 3.0

            new_cost = current_cost + step_cost
            if new_cost < costs[ny, nx]:
                costs[ny, nx] = new_cost
                came_from[(nx, ny)] = (cx, cy)
                heapq.heappush(queue, (new_cost, nx, ny))

    if target_reached is None:
        return False, None

    path = []
    curr = target_reached
    while curr in came_from:
        path.append(curr)
        curr = came_from[curr]
    path.append((sx, sy))
    path.reverse()

    temp_canvas = np.zeros((h, w), dtype=np.uint8)
    if len(path) > 2:
        path_arr = np.array(path, dtype=float)
        window = 15
        smoothed = path_arr.copy()

        for i in range(1, len(path_arr) - 1):
            start = max(0, i - window // 2)
            end = min(len(path_arr), i + window // 2 + 1)
            smoothed[i] = np.mean(path_arr[start:end], axis=0)

        smoothed[0] = path_arr[0]
        smoothed[-1] = path_arr[-1]

        for i in range(len(smoothed) - 1):
            p1 = (int(round(smoothed[i][0])), int(round(smoothed[i][1])))
            p2 = (int(round(smoothed[i + 1][0])), int(round(smoothed[i + 1][1])))
            cv2.line(temp_canvas, p1, p2, 255, 1)

        return True, temp_canvas

    return False, None


def extend_lines(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        Log.error(f"Не удалось загрузить изображение для продления линий: {image_path}")
        return -1

    h, w = img.shape[:2]
    output_img = img.copy()

    # ВАЖНО: В OpenCV цвета BGR, поэтому красный это (0, 0, 255), розовый (255, 0, 255)
    red_mask = colorToMask(img, color=settings.RED[::-1])
    pink_mask = colorToMask(img, color=settings.PINK[::-1])
    black_mask = colorToMask(img, color=settings.BLACK[::-1])

    red_binary_mask = red_mask.astype(np.uint8)
    obstacle_mask = cv2.bitwise_or(red_binary_mask, pink_mask.astype(np.uint8))

    endpoints, vectors = get_line_endpoints_and_directions(red_binary_mask)
    if len(endpoints) == 0:
        Log.warning("Не найдено висячих концов линий для продления.")
        cv2.imwrite(output_path, output_img)
        return 0

    direct_lines = []
    deferred_lines = []

    Log.info("Анализ лучей и сортировка направлений...")
    for ep, vec in zip(endpoints, vectors):
        ray_pt = ep.astype(float)
        is_direct = False
        dist = 0

        for step in range(1, 2000):
            ray_pt += vec
            rx, ry = int(round(ray_pt[0])), int(round(ray_pt[1]))

            if rx < 0 or rx >= w or ry < 0 or ry >= h or black_mask[ry, rx] > 0:
                is_direct = True
                dist = step
                break

            if step > 5 and obstacle_mask[ry, rx] > 0:
                break

        if is_direct:
            direct_lines.append((ep, vec, dist))
        else:
            deferred_lines.append((ep, vec))

    direct_lines.sort(key=lambda x: x[2])
    extension_canvas = np.zeros((h, w), dtype=np.uint8)

    Log.info(f"Найдено: прямых лучей (Группа А) = {len(direct_lines)}, сложных (Группа Б) = {len(deferred_lines)}.")

    for ep, vec, _ in direct_lines:
        success, line_canvas = grow_direct_line(ep, vec, black_mask, obstacle_mask, w, h)
        if success:
            extension_canvas = cv2.bitwise_or(extension_canvas, line_canvas)
            obstacle_mask = cv2.bitwise_or(obstacle_mask, line_canvas)
        else:
            deferred_lines.append((ep, vec))

    dropped_count = 0
    if deferred_lines:
        Log.info("Запуск модифицированного алгоритма Дейкстры для Группы Б...")
        for ep, vec in deferred_lines:
            success, line_canvas = find_corridor_path(ep, vec, black_mask, obstacle_mask, w, h)
            if success:
                extension_canvas = cv2.bitwise_or(extension_canvas, line_canvas)
                obstacle_mask = cv2.bitwise_or(obstacle_mask, line_canvas)

    if dropped_count > 0:
        Log.warning(f"Всего отброшено линий без выхода (тупиков): {dropped_count}")

    # Скелетизация (Thining) добавленных линий
    try:
        extension_canvas = cv2.ximgproc.thinning(extension_canvas, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    except AttributeError:
        # Резервный способ, если нет модуля ximgproc (отличная защита от ошибок!)
        skel = np.zeros_like(extension_canvas)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = extension_canvas.copy()
        while True:
            eroded = cv2.erode(temp, element)
            skel = cv2.bitwise_or(skel, cv2.subtract(temp, cv2.dilate(eroded, element)))
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0: break
        extension_canvas = skel

    # Закрашиваем новые линии красным цветом (BGR)
    output_img[extension_canvas == 255] = settings.RED[::-1]

    cv2.imwrite(output_path, output_img)
    Log.success(f"Продление горизонталей завершено! Результат: {output_path}")
    return 0