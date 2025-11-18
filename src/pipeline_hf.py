import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from src.config_loader import load_config
from src.ocr_utils_demo import ocr_title, ocr_sensors

def process_single_image(img: np.ndarray, color_ranges: dict):
    """
    Обрабатывает одно изображение:
    - вытаскивает титул;
    - находит и оцифровывает сенсоры;
    - возвращает структуру с результатом.
    """

    # ---------- 1. Оцифровка титула ----------
    h, w = img.shape[:2]
    title_roi = img[:45, :int(w / 2.4)]
    title_text = ocr_title(title_roi)

    # ---------- 2. Оцифровка сенсоров ----------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = None

    for (lo, hi) in color_ranges.values():
        lo_np = np.array(lo, dtype=np.uint8)
        hi_np = np.array(hi, dtype=np.uint8)
        cur = cv2.inRange(hsv, lo_np, hi_np)
        mask = cur if mask is None else cv2.bitwise_or(mask, cur)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Contours found:", len(contours)) # для отладки

    rois, positions = [], []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        print("Contour bbox:", x, y, ww, hh) # для отладки
        if ww < 90 or hh < 17:
            continue
        if hh / ww > 1.5:
            continue

        print("ROI accepted:", ww, hh) # для отладки
        hh_clamped = min(hh, 17)
        roi = img[y:y + hh_clamped, x:x + ww]
        rois.append(roi)
        positions.append((x, y, ww, hh))
        print("ROIs after filtering:", len(rois)) # для отладки

    # ---------- 3. OCR сенсоров ----------
    sensors = []
    if rois:
        ocr_results = ocr_sensors(rois)
        print(ocr_results) # для отладки

        for (x, y, ww, hh), result in zip(positions, ocr_results):
            sensors.append({
                "text": result[0],
                "score": result[1],
                "x": x,
                "y": y,
                "w": ww,
                "h": hh
            })

    return title_text, sensors

# ---------------------------------------------------------
# Основной pipeline → Excel
# ---------------------------------------------------------
def process_image_to_excel(cfg: dict):
    input_dir = Path(cfg["paths"]["input"])
    output_dir = Path(cfg["paths"]["output"])
    excel_name = cfg["export"]["excel_filename"]

    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / excel_name

    color_ranges = cfg["colors"]

    results = []

    for img_path in input_dir.glob("*.png"):
        name = img_path.name
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"Не могу прочитать файл {name}")
            continue

        title_text, sensors = process_single_image(img, color_ranges)

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
    df.to_excel(excel_path, index=False, engine="openpyxl")

    print(f"✅ Готово! Excel сохранён: {excel_path.resolve()}")


# ---------------------------------------------------------
# Запуск
# ---------------------------------------------------------
if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    process_image_to_excel(cfg)

