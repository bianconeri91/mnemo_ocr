import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd
import numpy as np

from src.pipeline import process_all_images_to_excel   # заменить на реальный путь


@patch("src.pipeline.extract_text")
@patch("src.pipeline.iter_input_images")
@patch("src.pipeline.pd.DataFrame.to_excel")
def test_process_all_images_to_excel_ok(mock_to_excel, mock_iter, mock_extract, tmp_path):
    # Подготовим конфиг
    cfg = {
        "paths": {
            "input": str(tmp_path / "input"),
            "output": str(tmp_path / "output")
        },
        "export": {
            "excel_filename": "out.xlsx"
        },
        "colors": {"dummy_color": ([0,0,0],[255,255,255])}
    }

    # Создаём папки
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()

    # --- Мокаем iter_input_images ---
    mock_iter.return_value = [
        ("img1", np.zeros((10,10,3))),
        ("img2", np.zeros((10,10,3))),
    ]

    # --- Мокаем extract_text ---
    mock_extract.side_effect = [
        ("TITLE1", [{"text": "S1", "score": 0.9}]),
        ("TITLE2", [{"text": "S2", "score": 0.95}]),
    ]

    # --- Выполняем функцию ---
    process_all_images_to_excel(cfg)

    # --- ПРОВЕРКИ ---

    # extract_text должен быть вызван 2 раза
    assert mock_extract.call_count == 2

    # Excel действительно формировался
    mock_to_excel.assert_called_once()

    # Проверяем, что путь правильный
    called_path, called_kwargs = mock_to_excel.call_args[0][0], mock_to_excel.call_args[1]
    assert called_kwargs["index"] is False
    assert "openpyxl" in called_kwargs["engine"]

    # Проверяем, что DataFrame содержит нужные строки
    df_created: pd.DataFrame = mock_to_excel.call_args[0][0]  # аргумент-DataFrame

    assert len(df_created) == 2
    assert set(df_created.columns) == {"filename", "title", "sensor_name", "score"}

    assert df_created.iloc[0]["filename"] == "img1"
    assert df_created.iloc[0]["title"] == "TITLE1"
    assert df_created.iloc[0]["sensor_name"] == "S1"
    assert df_created.iloc[0]["score"] == 0.9


@patch("src.pipeline.iter_input_images")
@patch("src.pipeline.pd.DataFrame.to_excel")
def test_process_all_images_no_data(mock_to_excel, mock_iter, tmp_path, capsys):
    cfg = {
        "paths": {
            "input": str(tmp_path / "input"),
            "output": str(tmp_path / "output")
        },
        "export": {"excel_filename": "out.xlsx"},
        "colors": {}
    }

    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()

    # iter_input_images вернёт пустой список
    mock_iter.return_value = []

    process_all_images_to_excel(cfg)

    # Excel не должен создаваться
    mock_to_excel.assert_not_called()

    # Проверяем вывод
    captured = capsys.readouterr()
    assert "нет данных" in captured.out.lower()