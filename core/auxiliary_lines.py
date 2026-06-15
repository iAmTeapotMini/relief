import os
import cv2
import numpy as np
from skimage import morphology
from collections import deque
from scipy.ndimage import distance_transform_edt

import settings
from core.utils import Log


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def get_endpoints(mask):
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    res = cv2.filter2D(mask.astype(np.uint8), -1, kernel) * mask
    return np.argwhere(res == 11)


def find_path_bfs_halfplane(skel_mask, start, end, ey, ex, dy, dx):
    """
    Поиск пути строго в передней полуплоскости.
    Защищает огибающую линию от выжигания в обход горы.
    """
    h, w = skel_mask.shape
    start, end = tuple(start), tuple(end)
    if start == end: return [start]

    q = deque([(start, [start])])
    v = {start}

    while q:
        curr, path = q.popleft()
        if curr == end: return path

        for o_y, o_x in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ny, nx = curr[0] + o_y, curr[1] + o_x

            if 0 <= ny < h and 0 <= nx < w and skel_mask[ny, nx] and (ny, nx) not in v:
                # dot >= -5.0 позволяет изгибаться по пикселям, но блокирует уход назад
                dot = (ny - ey) * dy + (nx - ex) * dx
                if dot >= -5.0:
                    v.add((ny, nx))
                    q.append(((ny, nx), path + [(ny, nx)]))

    return []


def get_direction_vector(pink_skel, ey, ex, dist=8):
    h, w = pink_skel.shape
    for d_search in [dist, 3]:
        q = deque([((ey, ex), 0)])
        v = {(ey, ex)}
        target = None
        while q:
            curr, d = q.popleft()
            if d == d_search: target = curr; break
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ny, nx = curr[0] + dy, curr[1] + dx
                if (0 <= ny < h and 0 <= nx < w and pink_skel[ny, nx] and (ny, nx) not in v):
                    v.add((ny, nx))
                    q.append(((ny, nx), d + 1))
        if target:
            return np.array([ey - target[0], ex - target[1]], dtype=float)
    return None


def shoot_thick_ray(skel, ey, ex, dy, dx, thickness=2.0):
    """
    Луч с физической толщиной. Пробивает пиксельные щели и находит стенку 100%.
    """
    candidates = np.argwhere(skel)
    if len(candidates) == 0:
        return None

    vys = candidates[:, 0] - ey
    vxs = candidates[:, 1] - ex

    t = vys * dy + vxs * dx
    dist_ortho = np.abs(vxs * dy - vys * dx)

    # Ищем пиксели спереди (t > 2.0) внутри коридора
    valid_mask = (t > 2.0) & (dist_ortho <= thickness)
    valid_candidates = candidates[valid_mask]
    valid_t = t[valid_mask]

    if len(valid_candidates) == 0:
        return None

    best_idx = np.argmin(valid_t)
    return tuple(valid_candidates[best_idx])


# --- ГЛАВНАЯ ФУНКЦИЯ ---

def build_auxiliary_lines(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        Log.error(f"Ошибка: Изображение не найдено {image_path}")
        return -1

    WHITE = tuple(settings.WHITE[::-1])
    PINK = tuple(settings.PINK[::-1])
    CYAN = tuple(settings.CYAN[::-1])
    PURPLE = tuple(settings.PURPLE[::-1])
    ORANGE = tuple(settings.ORANGE[::-1])

    final_img = img.copy()

    white_mask = np.all(img == WHITE, axis=-1)
    pink_mask = np.all(img == PINK, axis=-1)
    cyan_mask = np.all(img == CYAN, axis=-1)

    _, labels_p = cv2.connectedComponents(pink_mask.astype(np.uint8), connectivity=8)

    Log.info("Анализ белых областей (Геометрическое разделение фрагментов)...")
    num_isl, labels_isl = cv2.connectedComponents(white_mask.astype(np.uint8), connectivity=4)

    for lbl in range(1, num_isl):
        island = (labels_isl == lbl)
        isl_dilated = cv2.dilate(island.astype(np.uint8), np.ones((5, 5))).astype(bool)

        # 1. Локальные розовые линии
        touched_p_labels = np.unique(labels_p[isl_dilated & pink_mask])
        touched_p_labels = touched_p_labels[touched_p_labels != 0]

        if len(touched_p_labels) == 0:
            skel = morphology.skeletonize(island)
            final_img[skel] = PINK
            continue

        local_pink_mask = np.isin(labels_p, touched_p_labels)

        # 2. Локальные циановые пиксели (ТОЛЬКО касающиеся розовой)
        local_pink_dilated = cv2.dilate(local_pink_mask.astype(np.uint8), np.ones((5, 5)))
        local_cyan_mask = cyan_mask & (local_pink_dilated > 0)

        if not np.any(local_cyan_mask):
            local_cyan_mask = cyan_mask

        dist_p = distance_transform_edt(~local_pink_mask)
        dist_c = distance_transform_edt(~local_cyan_mask) if np.any(local_cyan_mask) else np.full_like(dist_p, 9999.0)

        skel = morphology.skeletonize(island)

        base_skel_P = skel & (dist_p < dist_c)
        base_skel_O = skel & (dist_p >= dist_c)

        # 3. ПОИСК И СЖИГАНИЕ ПЕРЕМЫЧЕК
        local_pink_skel = morphology.skeletonize(local_pink_mask)
        isl_endpoints = [ep for ep in get_endpoints(local_pink_skel) if isl_dilated[ep[0], ep[1]]]

        burned_mask = np.zeros_like(skel, dtype=bool)
        burn_events = []

        for ey, ex in isl_endpoints:
            vec = get_direction_vector(local_pink_skel, ey, ex, dist=8)
            if vec is None: continue

            vec_norm = vec / (np.linalg.norm(vec) + 1e-6)
            dy_f, dx_f = vec_norm[0], vec_norm[1]
            dy_l, dx_l = -dx_f, dy_f
            dy_r, dx_r = dx_f, -dy_f

            h_f = shoot_thick_ray(skel, ey, ex, dy_f, dx_f, thickness=2.5)
            h_l = shoot_thick_ray(skel, ey, ex, dy_l, dx_l, thickness=2.0)
            h_r = shoot_thick_ray(skel, ey, ex, dy_r, dx_r, thickness=2.0)

            event_mask = np.zeros_like(skel, dtype=bool)

            if h_f is not None:
                event_mask[h_f[0], h_f[1]] = True

                if h_l is not None:
                    path_l = find_path_bfs_halfplane(skel, h_l, h_f, ey, ex, dy_f, dx_f)
                    for py, px in path_l: event_mask[py, px] = True
                if h_r is not None:
                    path_r = find_path_bfs_halfplane(skel, h_r, h_f, ey, ex, dy_f, dx_f)
                    for py, px in path_r: event_mask[py, px] = True

                burned_mask |= event_mask
                burn_events.append({
                    'ey': ey, 'ex': ex,
                    'dy': dy_f, 'dx': dx_f,
                    'mask': event_mask
                })

        # 4. РАЗДЕЛЕНИЕ НА ОГИБАЮЩИЕ И ХВОСТЫ (ПОСЛЕ РАЗРЫВА)
        skel_cut = skel & ~burned_mask
        n_c, comps = cv2.connectedComponents(skel_cut.astype(np.uint8), connectivity=8)

        envelope_mask = np.zeros_like(skel, dtype=bool)
        colored_tails_img = np.zeros_like(final_img)

        for i in range(1, n_c):
            comp_mask = (comps == i)
            dilated_comp = cv2.dilate(comp_mask.astype(np.uint8), np.ones((3, 3))).astype(bool)

            is_tail = False
            votes = set()

            for event in burn_events:
                contact_pixels = dilated_comp & event['mask']
                if np.any(contact_pixels):
                    # Фрагмент отвалился от этой перемычки! Выясняем: это Хвост или Огибающая?
                    # Считаем Центр Масс фрагмента
                    coords = np.argwhere(comp_mask)
                    cy_mean = np.mean(coords[:, 0])
                    cx_mean = np.mean(coords[:, 1])

                    # Вектор от конца розовой линии до Центра Масс фрагмента
                    vy = cy_mean - event['ey']
                    vx = cx_mean - event['ex']
                    dist = np.hypot(vy, vx)

                    if dist > 0:
                        vy_norm, vx_norm = vy / dist, vx / dist
                        dot = vy_norm * event['dy'] + vx_norm * event['dx']

                        # Если Центр Масс находится ВПЕРЕДИ (по направлению розовой линии) -> ЭТО ХВОСТ
                        if dot > 0.1:
                            is_tail = True

                            # Собираем цвета перемычки, чтобы покрасить хвост целиком
                            contact_expanded = cv2.dilate(contact_pixels.astype(np.uint8), np.ones((5, 5))).astype(bool)
                            actual_contacts = contact_expanded & event['mask']
                            for cy, cx in np.argwhere(actual_contacts):
                                if base_skel_P[cy, cx]: votes.add('P')
                                if base_skel_O[cy, cx]: votes.add('O')

            if not is_tail:
                # Центр масс сбоку или сзади -> ЭТО ОГИБАЮЩАЯ
                envelope_mask |= comp_mask
            else:
                # ЭТО ХВОСТ: Красится как единый монолитный кусок!
                if len(votes) == 2:
                    colored_tails_img[comp_mask] = PINK
                elif 'P' in votes:
                    colored_tails_img[comp_mask] = PURPLE
                elif 'O' in votes:
                    colored_tails_img[comp_mask] = ORANGE
                else:
                    colored_tails_img[comp_mask] = PINK

        # 5. ФИНАЛЬНАЯ ОТРИСОВКА
        # Огибающие сохраняют точные пиксельные расстояния (градиенты)
        final_img[envelope_mask & base_skel_P] = PURPLE
        final_img[envelope_mask & base_skel_O] = ORANGE

        # Хвосты монолитные
        tail_active = np.any(colored_tails_img != 0, axis=-1)
        final_img[tail_active] = colored_tails_img[tail_active]

    cv2.imwrite(output_path, final_img)
    Log.success(f"Этап 4.6 завершен! Результат сохранен в {output_path}")
    return 0