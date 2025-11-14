import numpy as np
import pytest
from unittest.mock import patch

from src.ocr_utils import ocr_title, ocr_sensors

@patch("src.ocr_utils.pytesseract.image_to_string")
def test_ocr_title_basic(mock_tesseract):
    # Подделываем OCR-результат
    mock_tesseract.return_value = "TEST_TITLE"

    # Синтетическое изображение
    img = np.zeros((50, 200, 3), dtype=np.uint8)

    text = ocr_title(img)

    assert text == "TEST_TITLE"
    mock_tesseract.assert_called_once()

@patch("src.ocr_utils.paddle_ocr.predict")
def test_ocr_sensors_basic(mock_predict):
    # Подменяем результат работы paddleocr.predict
    mock_predict.return_value = [
        {"rec_texts": ["123"], "rec_scores": [0.98]}
    ]

    # Одна синтетическая ROI
    img = np.zeros((20, 60, 3), dtype=np.uint8)

    out = ocr_sensors([img])

    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["text"] == "123"
    assert out[0]["score"] == 0.98
    mock_predict.assert_called_once()

@patch("src.ocr_utils.paddle_ocr.predict")
def test_ocr_sensors_empty_roi(mock_predict):
    # paddleOCR должен корректно отработать
    mock_predict.return_value = [
        {"rec_texts": [""], "rec_scores": [0]}
    ]

    img = np.zeros((10, 10, 3), dtype=np.uint8)

    out = ocr_sensors([img])

    assert out[0]["text"] == ""
    assert out[0]["score"] == 0
