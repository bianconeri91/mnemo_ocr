import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from  src.pipeline import iter_input_images   # замените под путь вашего файла

@patch("src.pipeline.cv2.imread")
def test_iter_input_images_with_images(mock_imread, tmp_path):
    # Создаем временные файлы
    img1 = tmp_path / "a.png"
    img2 = tmp_path / "b.jpg"
    img1.write_bytes(b"fake")
    img2.write_bytes(b"fake")

    # Мокаем результат чтения изображений
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_imread.return_value = dummy_img

    results = list(iter_input_images(tmp_path))

    assert len(results) == 2
    assert results[0][0] == "a"     # stem
    assert isinstance(results[0][1], np.ndarray)

    assert results[1][0] == "b"
    assert isinstance(results[1][1], np.ndarray)

    assert mock_imread.call_count == 2

@patch("src.pipeline.cv2.imdecode")
@patch("src.pipeline.Document")
def test_iter_input_images_with_docx(mock_Document, mock_imdecode, tmp_path):
    docx_file = tmp_path / "test.docx"
    docx_file.write_bytes(b"fake")

    # Мокаем Document
    mock_doc = MagicMock()
    mock_Document.return_value = mock_doc

    # Имитация doc.part._rels
    mock_rel = MagicMock()
    mock_rel.target_ref = "/media/image1.png"
    mock_rel.target_part.blob = b"FAKE_IMAGE_DATA"

    mock_doc.part._rels = {"r1": mock_rel}

    # cv2.imdecode → возвращает искусственное изображение
    dummy_img = np.ones((5, 5, 3), dtype=np.uint8)
    mock_imdecode.return_value = dummy_img

    results = list(iter_input_images(tmp_path))

    assert len(results) == 1
    name, img = results[0]

    assert name == "test_img1"
    assert isinstance(img, np.ndarray)
    assert img.shape == (5, 5, 3)

    mock_Document.assert_called_once()
    mock_imdecode.assert_called_once()

def test_iter_input_images_empty_folder(tmp_path, capsys):
    results = list(iter_input_images(tmp_path))

    # Функция ничего не вернёт
    assert results == []

    # Проверяем, что напечатано предупреждение
    captured = capsys.readouterr()
    assert "не содержит ни изображений" in captured.out.lower()
