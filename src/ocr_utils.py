"""
OCR-инструменты: pytesseract для титула и PaddleOCR для датчиков.
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR
import pytesseract


# --------------------------
# OCR титула (pytesseract)
# --------------------------
def ocr_title(img: np.ndarray) -> str:
    """
    OCR верхней области (титул мнемосхемы).
    """
    if img is None or img.size == 0:
        return ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=-40)

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 9
    )

    text = pytesseract.image_to_string(binary, lang="eng", config="--psm 7").strip()
    return text


# --------------------------
# OCR датчиков (PaddleOCR)
# --------------------------
paddle_ocr = PaddleOCR(
    lang="en",
    ocr_version="PP-OCRv4"
)


def ocr_sensors(rois: list[np.ndarray]) -> list[dict]:
    """
    OCR областей сенсоров через PaddleOCR.predict().
    Формат вывода:
        [{"text": str, "score": float}, ...]
    """
    results = []

    if not rois:
        return []

    try:
        ocr_results = paddle_ocr.predict(rois)
    except Exception as e:
        print(f"⚠ Ошибка OCR.predict: {e}")
        return [{"text": "?", "score": 0.0} for _ in rois]

    for out in ocr_results:
        
        texts = out.get("rec_texts", ["?"])
        scores = out.get("rec_scores", [0.0])

        # --- распаковка ---
        text = texts[0] if texts else "?"
        score = scores[0] if scores else 0.0

        results.append({"text": text, "score": round(score, 2)})

    return results

