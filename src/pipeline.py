"""
Главный pipeline проекта
"""

import argparse
import json
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from zipfile import BadZipFile
from docx import Document

from src.config_loader import load_config
from src.ocr_utils import ocr_title, ocr_sensors, ocr_full_image


# ---------------------------------------------------------
# Итератор входных данных (СЦЕНАРИЙ A)
# ---------------------------------------------------------
def iter_input_images(input_dir: Path, save_images_dir: Path | None = None):
    """
    Универсальный вход:
    - читаем ВСЕ png/jpg/bmp
    - читаем ВСЕ изображения внутри DOCX
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp"}

    image_files = []
    docx_files = []

    for p in input_dir.rglob("*"):
        if not p.is_file():
            continue

        suf = p.suffix.lower()
        if suf in exts:
            image_files.append(p)
        elif suf == ".docx" and not p.name.startswith("~$"):
            docx_files.append(p)

    # ---------- PNG/JPG ----------
    for path in sorted(image_files):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            yield path.stem, img

    if save_images_dir is not None:
        save_images_dir.mkdir(parents=True, exist_ok=True)

    # ---------- DOCX ----------
    for docx_path in sorted(docx_files):
        try:
            doc = Document(docx_path)
        except BadZipFile:
            print(f"⚠ Пропускаю некорректный DOCX: {docx_path}")
            continue

        idx = 0
        for rel in doc.part._rels.values():
            if "image" not in rel.target_ref:
                continue

            idx += 1
            blob = rel.target_part.blob
            arr = np.frombuffer(blob, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img is not None:
                image_name = f"{docx_path.stem}_img{idx}"
                if save_images_dir is not None:
                    cv2.imwrite(str(save_images_dir / f"{image_name}.png"), img)

                yield image_name, img

    # Если не нашли ничего
    if not image_files and not docx_files:
        print("⚠ Папка не содержит ни изображений, ни DOCX.")



# ---------------------------------------------------------
# Извлечение текста титула + датчиков
# ---------------------------------------------------------
def extract_text(img: np.ndarray, color_ranges: dict) -> tuple[str, list[dict]]:
    """
    1. Обрезаем титул (верхняя область)
    2. OCR титула
    3. Извлекаем сенсоры по цветовым маскам
    4. OCR сенсоров
    """

    # ---------- 1. титул ----------
    h, w = img.shape[:2]
    title_roi = img[:45, :int(w / 2.4)]
    title_text = ocr_title(title_roi)

    # ---------- 2. сенсоры ----------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = None
    for (lo, hi) in color_ranges.values():
        lo_np = np.array(lo, dtype=np.uint8)
        hi_np = np.array(hi, dtype=np.uint8)
        cur = cv2.inRange(hsv, lo_np, hi_np)
        mask = cur if mask is None else cv2.bitwise_or(mask, cur)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois, positions = [], []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        if ww < 90 or hh < 17:
            continue

        hh = min(hh, 17)
        rois.append(img[y:y + hh, x:x + ww])
        positions.append((x, y, ww, hh))

    # ---------- 3. OCR сенсоров ----------
    sensors = []
    if rois:
        ocr_results = ocr_sensors(rois)

        for (x, y, ww, hh), r in zip(positions, ocr_results):
            sensors.append({
                "text": r["text"],
                "score": r["score"],
                "x": x,
                "y": y,
                "w": ww,
                "h": hh
            })

    return title_text, sensors


# ---------------------------------------------------------
# Основной pipeline → Excel
# ---------------------------------------------------------
def process_all_images_to_excel(cfg: dict, save_images_dir: Path | None = None):
    input_dir = Path(cfg["paths"]["input"])
    output_dir = Path(cfg["paths"]["output"])
    excel_name = cfg["export"]["excel_filename"]

    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / excel_name

    # Цветовые диапазоны сенсоров
    color_ranges = cfg["colors"]

    results = []

    for name, img in iter_input_images(input_dir, save_images_dir=save_images_dir):
        title_text, sensors = extract_text(img, color_ranges)

        for sen in sensors:
            results.append({
                "filename": name,
                "title": title_text,
                "sensor_name": sen["text"],
                "score": sen["score"]
            })

    if not results:
        print("⚠ Нет данных для записи.")
        return

    df = pd.DataFrame(results)
    pd.DataFrame.to_excel(df, excel_path, index=False, engine="openpyxl")

    print(f"✅ Готово! Excel сохранён: {excel_path.resolve()}")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


POSITION_CODE_RE = re.compile(
    r"(?<![A-Za-zА-Яа-я0-9])"
    r"\d{3}[^A-Za-zА-Яа-я0-9]*[A-Za-zА-Яа-я]{2,5}"
    r"[^A-Za-zА-Яа-я0-9]*\d{3}[A-Za-zА-Яа-я]?"
    r"(?![A-Za-zА-Яа-я0-9])"
)
IMAGE_POSTFIX_RE = re.compile(r"(_img\d+)$")


def _split_image_name(name: str) -> tuple[str, str]:
    match = IMAGE_POSTFIX_RE.search(name)
    if not match:
        return name, ""

    postfix = f"{match.group(1)}.png"
    return name[:match.start()], postfix


def _extract_position_code(text: str) -> str:
    match = POSITION_CODE_RE.search(text)
    return match.group(0) if match else ""


def _normalize_position_code(position_code: str) -> str:
    if not position_code:
        return ""

    compact = re.sub(r"[^A-Za-zА-Яа-я0-9]", "", position_code)
    match = re.fullmatch(
        r"(\d{3})([A-Za-zА-Яа-я]{2,5})(\d{3}[A-Za-zА-Яа-я]?)",
        compact
    )
    if not match:
        return position_code.replace(".", "-")

    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


def _is_ocr_item(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[1], (list, tuple))
        and len(value[1]) >= 2
        and isinstance(value[1][0], str)
    )


def _iter_ocr_items(raw_ocr_result):
    """
    Разворачивает сырой PaddleOCR-ответ до отдельных OCR-блоков.
    Поддерживает формат одной страницы и формат списка страниц.
    """
    if not isinstance(raw_ocr_result, list):
        return

    for item in raw_ocr_result:
        if _is_ocr_item(item):
            yield item
        elif isinstance(item, list):
            for nested_item in item:
                if _is_ocr_item(nested_item):
                    yield nested_item


def _format_point(point) -> str:
    return json.dumps(point, ensure_ascii=False, default=_json_default)


def _rows_from_raw_ocr(name: str, raw_ocr_result) -> list[dict]:
    filename, image_postfix = _split_image_name(name)
    rows = []

    for item in _iter_ocr_items(raw_ocr_result):
        points, text_score = item[0], item[1]
        text = str(text_score[0]) if text_score else ""
        score = float(text_score[1]) if len(text_score) > 1 else 0.0
        position_code = _extract_position_code(text)

        rows.append({
            "filename": filename,
            "image_postfix": image_postfix,
            "point_1": _format_point(points[0]) if len(points) > 0 else "",
            "point_2": _format_point(points[1]) if len(points) > 1 else "",
            "point_3": _format_point(points[2]) if len(points) > 2 else "",
            "point_4": _format_point(points[3]) if len(points) > 3 else "",
            "text": text,
            "score": score,
            "position_code": position_code,
            "position_code_normalized": _normalize_position_code(position_code)
        })

    return rows


# ---------------------------------------------------------
# Сценарий B: полный OCR изображения → Excel
# ---------------------------------------------------------
def process_full_ocr_to_excel(cfg: dict, save_images_dir: Path | None = None):
    input_dir = Path(cfg["paths"]["input"])
    output_dir = Path(cfg["paths"]["output"])
    excel_name = cfg["export"]["excel_filename"]

    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / excel_name

    results = []

    for name, img in iter_input_images(input_dir, save_images_dir=save_images_dir):
        raw_ocr_result = ocr_full_image(img)
        results.extend(_rows_from_raw_ocr(name, raw_ocr_result))

    if not results:
        print("⚠ Нет данных для записи.")
        return

    df = pd.DataFrame(results)
    pd.DataFrame.to_excel(df, excel_path, index=False, engine="openpyxl")

    print(f"✅ Готово! Excel сохранён: {excel_path.resolve()}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run OCR pipeline.")
    parser.add_argument(
        "--scenario",
        choices=("a", "b"),
        default="a",
        help="Сценарий обработки: a — датчики по цветам, b — полный OCR изображения"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Сохранять изображения, извлеченные из DOCX"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Папка для сохранения изображений из DOCX"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    save_images_dir = Path(args.images_dir) if args.save_images else None

    if args.scenario == "a":
        process_all_images_to_excel(cfg, save_images_dir=save_images_dir)
    else:
        process_full_ocr_to_excel(cfg, save_images_dir=save_images_dir)


# ---------------------------------------------------------
# Запуск
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
