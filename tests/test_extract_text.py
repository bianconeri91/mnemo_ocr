import numpy as np
import cv2
import pytest
from unittest.mock import patch

from src.pipeline import extract_text   # поправь название модуля


# ---------------------------------------------------------
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: создаём тестовую картинку
# ---------------------------------------------------------
def make_test_image():
    """
    Создаёт искусственное изображение 200×300 + цветовой диапазон,
    чтобы extract_text мог найти один "сенсор".
    """
    img = np.zeros((200, 300, 3), dtype=np.uint8)

    # Рисуем титул (верхний левый угол) — просто белую полосу
    img[0:45, 0:120] = (255, 255, 255)

    # Рисуем сенсор в центре картинки
    cv2.rectangle(img, (100, 80), (200, 97), (0, 255, 0), -1)

    return img


# ---------------------------------------------------------
# ГЛАВНЫЙ ТЕСТ extract_text
# ---------------------------------------------------------
@patch("src.pipeline.ocr_title")
@patch("src.pipeline.ocr_sensors")
def test_extract_text(mock_ocr_sensors, mock_ocr_title):
    img = make_test_image()

    # Цветовой диапазон, по которому найдётся наш зелёный прямоугольник
    color_ranges = {
        "green": (
            [50, 80, 80],   # нижняя граница HSV
            [80, 255, 255]  # верхняя граница HSV
        )
    }

    # --- подменяем OCR ---
    mock_ocr_title.return_value = "TITLE_OK"

    mock_ocr_sensors.return_value = [
        {"text": "123", "score": 0.95},
    ]

    # --- вызываем тестируемую функцию ---
    title, sensors = extract_text(img, color_ranges)

    # --- ПРОВЕРКИ ---

    # титул
    assert title == "TITLE_OK"

    # сенсоры
    assert isinstance(sensors, list)
    assert len(sensors) == 1

    s = sensors[0]

    assert s["text"] == "123"
    assert s["score"] == 0.95

    # координаты должны быть внутри изображения
    assert 0 <= s["x"] <= img.shape[1]
    assert 0 <= s["y"] <= img.shape[0]
    assert s["w"] > 0
    assert s["h"] > 0