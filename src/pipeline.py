"""
Главный pipeline проекта
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from docx import Document

from src.config_loader import load_config
from src.ocr_utils_demo import ocr_title, ocr_sensors


# ---------------------------------------------------------
# Итератор входных данных (СЦЕНАРИЙ A)
# ---------------------------------------------------------
def iter_input_images(input_dir: Path):
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
        elif suf == ".docx":
            docx_files.append(p)

    # ---------- PNG/JPG ----------
    for path in sorted(image_files):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            yield path.stem, img

    # ---------- DOCX ----------
    for docx_path in sorted(docx_files):
        doc = Document(docx_path)

        idx = 0
        for rel in doc.part._rels.values():
            if "image" not in rel.target_ref:
                continue

            idx += 1
            blob = rel.target_part.blob
            arr = np.frombuffer(blob, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img is not None:
                yield f"{docx_path.stem}_img{idx}", img

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
def process_all_images_to_excel(cfg: dict):
    input_dir = Path(cfg["paths"]["input"])
    output_dir = Path(cfg["paths"]["output"])
    excel_name = cfg["export"]["excel_filename"]

    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / excel_name

    # Цветовые диапазоны сенсоров
    color_ranges = cfg["colors"]

    results = []

    for name, img in iter_input_images(input_dir):
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


# ---------------------------------------------------------
# Запуск
# ---------------------------------------------------------
if __name__ == "__main__":
    cfg = load_config("config.yaml")
    process_all_images_to_excel(cfg)