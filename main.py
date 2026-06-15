import os

from core.auxiliary_lines import build_auxiliary_lines
from core.height_calculator import assign_heights_to_lines
from core.dem_generator import generate_heightmap
from core.line_extension import extend_lines

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOGGING_MIN_LOG_LEVEL'] = '3'

import shutil
import settings
from core.contour_builder import extract_map_boundary
from core.image_processor import resize_with_pad, slice_image_into_tiles, stitch_patches
from core.inference import run_inference
from core.utils import Log


def prepare_environment():
    """Очищает папку с результатами предыдущего запуска и создает новую"""
    for folder in [settings.TEMP_DIR, settings.OUTPUT_DIR]:
        if os.path.exists(folder):
            Log.info(f"Удаление старой папки: {folder}")
            shutil.rmtree(folder)

        os.makedirs(folder, exist_ok=True)
        Log.info(f"Создана чистая папка: {folder}")


def select_image_from_folder(folder_path):
    """Сканирует папку и дает пользователю выбрать картинку"""
    # Если папки нет, создаем её и просим пользователя положить туда файлы
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        Log.error(f"Папка {folder_path} была пуста. Мы ее создали.")
        Log.warning("Пожалуйста, положите туда изображения и перезапустите скрипт.")
        return None

    # Ищем только картинки
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

    if not files:
        Log.error(f"В папке {folder_path} нет изображений!")
        return None

    print(f"\nДоступные изображения в [ {folder_path} ]:")
    for i, file_name in enumerate(files):
        print(f"  {Log.CYAN}{i + 1}.{Log.RESET} {file_name}")

    # Запрашиваем ввод, пока пользователь не введет корректное число
    while True:
        try:
            choice = input(f"\nВведите номер изображения (1-{len(files)}): ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(files):
                selected_file = os.path.join(folder_path, files[choice_idx])
                Log.success(f"Выбран файл: {files[choice_idx]}")
                return selected_file
            else:
                Log.warning("Нет такого номера в списке. Попробуйте еще раз.")
        except ValueError:
            Log.warning("Пожалуйста, введите число.")


def run_inference1(image_path):
    """Логика первой части алгоритма"""
    Log.step("ЗАПУСК ЧАСТИ 1: ИНФЕРЕНС И ПРЕДОБРАБОТКА")
    Log.info(f"Обработка исходной карты: {image_path}")

    Log.step("НАЧАТО: Изменение размера изображения")
    temp_map = os.path.join(settings.TEMP_DIR, 'resize_image.png')
    resize_with_pad(image_path, temp_map)
    Log.step("ЗАВЕРШЕНО: Изменение размера изображения")

    Log.step("НАЧАТО: Нарезка изображения на фрагменты 1024х1024 с перекрытием 256")
    slice_image_into_tiles(temp_map)
    Log.step("ЗАВЕРШЕНО: Нарезка изображения на фрагменты 1024х1024 с перекрытием 256")

    Log.step("НАЧАТО: Предсказание модели")
    run_inference()
    Log.step("ЗАВЕРШЕНО: Предсказание модели")

    Log.step("НАЧАТО: Сшивка изображения")
    stitched_mask_path = os.path.join(settings.OUTPUT_DIR, 'stitched_mask.png')
    stitch_patches(settings.PREDICTED_TILES_DIR, stitched_mask_path)
    Log.step("ЗВАЕРШЕНО: Сшивка изображения")

    Log.info("НАЧАТО: Определение границы карты")
    boundary_path = os.path.join(settings.TEMP_DIR, 'boundary.png')
    result_code = extract_map_boundary(stitched_mask_path, boundary_path)
    if result_code != 0:
        return Log.error("Программа остановлена")
    Log.info("ЗАВЕРШЕНО: Определение границы карты")


def run_dem(image_path):
    """Логика второй части алгоритма (Генерация ЦМР)"""
    Log.step("ЗАПУСК ЧАСТИ 2: ГЕНЕРАЦИЯ ЦМР")
    Log.info(f"Использование эталонной маски: {image_path}")

    Log.step("НАЧАТО: Определение границы карты")
    mask_path = os.path.join(settings.TEMP_DIR, 'boundary.png')
    result_code = extract_map_boundary(image_path, mask_path)
    if result_code != 0:
        return Log.error("Программа остановлена")
    Log.success("ЗАВЕРШЕНО: Определение границы карты")

    Log.step("НАЧАТО: Продление изолиний до границ карты")
    extended_mask_path = os.path.join(settings.TEMP_DIR, 'extended_lines.png')
    extend_lines(mask_path, extended_mask_path)
    Log.step("ЗАВЕРШЕНО: Продление изолиний до границ карты")

    Log.step("НАЧАТО: Построение вспомогательных линий")
    aux_lines_path = os.path.join(settings.TEMP_DIR, 'auxiliary_lines.png')
    build_auxiliary_lines(extended_mask_path, aux_lines_path)
    Log.step("ЗАВЕРШЕНО: Построение вспомогательных линий")

    Log.step("НАЧАТО: Построение графа, вычисление относительных высот")
    colored_lines_path = os.path.join(settings.TEMP_DIR, 'colored_lines.png')
    assign_heights_to_lines(
        main_image_path=extended_mask_path,
        aux_lines_path=aux_lines_path,
        output_path=colored_lines_path
    )
    Log.step("ЗАВЕРШЕНО: Построение графа, вычисление относительных высот")

    # 4.8 Генерация карты высот (Интерполяция)
    Log.step("НАЧАТО: Генерация финальной Цифровой Модели Рельефа")
    final_dem_path = os.path.join(settings.OUTPUT_DIR, 'height_map.png')
    generate_heightmap(colored_lines_path, final_dem_path)
    Log.step("ЗАВЕРШЕНО: Генерация финальной Цифровой Модели Рельефа")
    Log.step("ОБРАБОТКА ПОЛНОСТЬЮ ЗАВЕРШЕНА!")



def main():
    Log.step("ГЛАВНОЕ МЕНЮ ПРОГРАММЫ")
    print("Доступные режимы:")
    print(f"  {Log.CYAN}1.{Log.RESET} Инференс нейросети (Спортивная карта -> Маска классов (низкое качество))")
    print(f"  {Log.CYAN}2.{Log.RESET} Генерация цифровой модели рельефа (Эталонная маска -> ЦМР)")
    print(f"  {Log.CYAN}0.{Log.RESET} Выход")

    while True:
        mode = input("\nВыберите режим (0, 1 или 2): ")

        if mode == '0':
            Log.info("Завершение работы программы.")
            break

        elif mode == '1':
            selected_image = select_image_from_folder(settings.INFERENCE_IMAGE_PATH)
            if selected_image:
                prepare_environment()
                run_inference1(selected_image)
            break

        elif mode == '2':
            selected_image = select_image_from_folder(settings.DEMO_REFERENCE_MASK_PATH)
            if selected_image:
                prepare_environment()
                run_dem(selected_image)
            break
        else:
            Log.warning("Неверный ввод. Пожалуйста, введите 0, 1 или 2.")


if __name__ == "__main__":
    main()