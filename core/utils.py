import os

import numpy as np

os.system("")

def colorToMask(image, color):
    return np.all(image == color, axis=2).astype(np.uint8) * 255


class Log:
    """Класс для красивого цветного вывода в консоль"""

    # ANSI-коды цветов
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'

    @staticmethod
    def info(msg):
        print(f"{Log.CYAN}[ИНФО]{Log.RESET} {msg}")

    @staticmethod
    def success(msg):
        print(f"{Log.GREEN}[УСПЕХ]{Log.RESET} {msg}")

    @staticmethod
    def warning(msg):
        print(f"{Log.YELLOW}[ВНИМАНИЕ]{Log.RESET} {msg}")

    @staticmethod
    def error(msg):
        print(f"{Log.RED}[ОШИБКА]{Log.RESET} {msg}")

    @staticmethod
    def step(msg):
        print(f"\n{Log.MAGENTA}=== {msg} ==={Log.RESET}")