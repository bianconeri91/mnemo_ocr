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
    use_gpu=False
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

    for roi in rois:
        try:
            ocr_results = paddle_ocr.ocr(roi, cls=False)
        except Exception as e:
            results.append({"text": "?", "score": 0.0})
            continue

    if ocr_results and ocr_results[0]:
        text, score = ocr_results[0][0][1]
    else:
        text, score = "?", 0.0

    results.append({
        "text": text,
        "score": float(score)
    })
    
    return results

