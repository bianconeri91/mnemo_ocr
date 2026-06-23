from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline import main, process_full_ocr_to_excel


def _cfg(tmp_path):
    return {
        "paths": {
            "input": str(tmp_path / "input"),
            "output": str(tmp_path / "output")
        },
        "export": {"excel_filename": "out.xlsx"},
        "colors": {}
    }


@patch("src.pipeline.process_all_images_to_excel")
@patch("src.pipeline.process_full_ocr_to_excel")
@patch("src.pipeline.load_config")
def test_main_dispatches_scenario_a(mock_load_config, mock_process_b, mock_process_a):
    cfg = {"ok": True}
    mock_load_config.return_value = cfg

    main(["--scenario", "a", "--config", "custom.yaml"])

    mock_load_config.assert_called_once_with("custom.yaml")
    mock_process_a.assert_called_once_with(cfg, save_images_dir=None)
    mock_process_b.assert_not_called()


@patch("src.pipeline.process_all_images_to_excel")
@patch("src.pipeline.process_full_ocr_to_excel")
@patch("src.pipeline.load_config")
def test_main_dispatches_scenario_b(mock_load_config, mock_process_b, mock_process_a):
    cfg = {"ok": True}
    mock_load_config.return_value = cfg

    main(["--scenario", "b", "--config", "custom.yaml"])

    mock_load_config.assert_called_once_with("custom.yaml")
    mock_process_b.assert_called_once_with(cfg, save_images_dir=None)
    mock_process_a.assert_not_called()


@patch("src.pipeline.process_all_images_to_excel")
@patch("src.pipeline.process_full_ocr_to_excel")
@patch("src.pipeline.load_config")
def test_main_passes_save_images_dir(mock_load_config, mock_process_b, mock_process_a):
    cfg = {"ok": True}
    mock_load_config.return_value = cfg

    main([
        "--scenario",
        "b",
        "--save-images",
        "--images-dir",
        "custom_images"
    ])

    mock_process_b.assert_called_once_with(cfg, save_images_dir=Path("custom_images"))
    mock_process_a.assert_not_called()


@patch("src.pipeline.ocr_full_image")
@patch("src.pipeline.iter_input_images")
@patch("src.pipeline.pd.DataFrame.to_excel")
def test_process_full_ocr_to_excel_ok(mock_to_excel, mock_iter, mock_ocr, tmp_path):
    cfg = _cfg(tmp_path)
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()

    mock_iter.return_value = [
        ("doc_img4", np.zeros((10, 10, 3), dtype=np.uint8)),
    ]
    mock_ocr.return_value = [
        [
            [
                [[54.0, 9.0], [82.0, 9.0], [82.0, 19.0], [54.0, 19.0]],
                ["PST", 0.9835820198059082]
            ],
            [
                [[49.0, 19.0], [98.0, 19.0], [98.0, 62.0], [49.0, 62.0]],
                ["TA", 0.9174954295158386]
            ],
            [
                [[24.0, 69.0], [104.0, 69.0], [104.0, 85.0], [24.0, 85.0]],
                ["352-KV-616A", 0.9943318963050842]
            ],
        ]
    ]

    process_full_ocr_to_excel(cfg)

    mock_to_excel.assert_called_once()
    called_kwargs = mock_to_excel.call_args[1]
    assert called_kwargs["index"] is False
    assert called_kwargs["engine"] == "openpyxl"

    df_created: pd.DataFrame = mock_to_excel.call_args[0][0]
    assert len(df_created) == 3
    assert set(df_created.columns) == {
        "filename",
        "image_postfix",
        "point_1",
        "point_2",
        "point_3",
        "point_4",
        "text",
        "score",
        "position_code",
        "position_code_normalized"
    }
    assert df_created.iloc[0]["filename"] == "doc"
    assert df_created.iloc[0]["image_postfix"] == "_img4.png"
    assert df_created.iloc[0]["point_1"] == "[54.0, 9.0]"
    assert df_created.iloc[0]["point_4"] == "[54.0, 19.0]"
    assert df_created.iloc[0]["text"] == "PST"
    assert df_created.iloc[0]["position_code"] == ""
    assert df_created.iloc[0]["position_code_normalized"] == ""

    assert df_created.iloc[2]["text"] == "352-KV-616A"
    assert df_created.iloc[2]["position_code"] == "352-KV-616A"
    assert df_created.iloc[2]["position_code_normalized"] == "352-KV-616A"


@patch("src.pipeline.ocr_full_image")
@patch("src.pipeline.iter_input_images")
@patch("src.pipeline.pd.DataFrame.to_excel")
def test_process_full_ocr_to_excel_extracts_position_without_suffix(
    mock_to_excel,
    mock_iter,
    mock_ocr,
    tmp_path
):
    cfg = _cfg(tmp_path)
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()

    mock_iter.return_value = [
        ("plain_image", np.zeros((10, 10, 3), dtype=np.uint8)),
    ]
    mock_ocr.return_value = [
        [
            [[0, 0], [1, 0], [1, 1], [0, 1]],
            ["352.LIC.616", 0.93],
        ],
        [
            [[0, 2], [1, 2], [1, 3], [0, 3]],
            ["352LIC616", 0.94],
        ],
        [
            [[0, 4], [1, 4], [1, 5], [0, 5]],
            ["352.ABCDE/616A", 0.95],
        ],
    ]

    process_full_ocr_to_excel(cfg)

    df_created: pd.DataFrame = mock_to_excel.call_args[0][0]
    assert df_created.iloc[0]["filename"] == "plain_image"
    assert df_created.iloc[0]["image_postfix"] == ""
    assert df_created.iloc[0]["position_code"] == "352.LIC.616"
    assert df_created.iloc[0]["position_code_normalized"] == "352-LIC-616"
    assert df_created.iloc[1]["position_code"] == "352LIC616"
    assert df_created.iloc[1]["position_code_normalized"] == "352-LIC-616"
    assert df_created.iloc[2]["position_code"] == "352.ABCDE/616A"
    assert df_created.iloc[2]["position_code_normalized"] == "352-ABCDE-616A"


@patch("src.pipeline.ocr_full_image")
@patch("src.pipeline.iter_input_images")
@patch("src.pipeline.pd.DataFrame.to_excel")
def test_process_full_ocr_to_excel_normalizes_missing_dashes(
    mock_to_excel,
    mock_iter,
    mock_ocr,
    tmp_path
):
    cfg = _cfg(tmp_path)
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()

    mock_iter.return_value = [
        ("img", np.zeros((10, 10, 3), dtype=np.uint8)),
    ]
    mock_ocr.return_value = [
        [
            [[0, 0], [1, 0], [1, 1], [0, 1]],
            ["381LA990", 0.93],
        ],
        [
            [[0, 2], [1, 2], [1, 3], [0, 3]],
            ["381LA-990", 0.94],
        ],
        [
            [[0, 4], [1, 4], [1, 5], [0, 5]],
            ["381-LA990", 0.95],
        ],
    ]

    process_full_ocr_to_excel(cfg)

    df_created: pd.DataFrame = mock_to_excel.call_args[0][0]
    assert list(df_created["position_code"]) == [
        "381LA990",
        "381LA-990",
        "381-LA990",
    ]
    assert list(df_created["position_code_normalized"]) == [
        "381-LA-990",
        "381-LA-990",
        "381-LA-990",
    ]


@patch("src.pipeline.ocr_full_image")
@patch("src.pipeline.iter_input_images")
@patch("src.pipeline.pd.DataFrame.to_excel")
def test_process_full_ocr_to_excel_no_data(
    mock_to_excel,
    mock_iter,
    mock_ocr,
    tmp_path,
    capsys
):
    cfg = _cfg(tmp_path)
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()

    mock_iter.return_value = [
    ]

    process_full_ocr_to_excel(cfg)

    mock_to_excel.assert_not_called()
    captured = capsys.readouterr()
    assert "нет данных" in captured.out.lower()
