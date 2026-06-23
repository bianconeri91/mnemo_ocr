"""
OCR-инструменты: pytesseract для титула и PaddleOCR для датчиков.
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR
import pytesseract

if not hasattr(np, "int"):
    np.int = int


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
    ocr_version="PP-OCRv3",
    use_angle_cls=False,
    show_log=False
)

if not hasattr(paddle_ocr, "predict"):
    paddle_ocr.predict = paddle_ocr.ocr


def _normalize_paddle_output(out) -> list[dict]:
    """
    Приводит ответ PaddleOCR к формату [{"text": str, "score": float}, ...].
    Поддерживает словарь из тестов и формат PaddleOCR 2.x.
    """
    if isinstance(out, dict):
        texts = out.get("rec_texts", ["?"])
        scores = out.get("rec_scores", [0.0])

        if isinstance(texts, str):
            texts = [texts]
        if isinstance(scores, (int, float)):
            scores = [scores]

        return [
            {"text": text, "score": round(float(score), 2)}
            for text, score in zip(texts, scores)
        ]

    if isinstance(out, list) and out:
        if all(isinstance(line, dict) for line in out):
            results = []
            for line in out:
                results.extend(_normalize_paddle_output(line))
            return results

        results = []
        for line in out:
            if (
                isinstance(line, list)
                and len(line) >= 2
                and isinstance(line[1], tuple)
            ):
                text, score = line[1]
                results.append({"text": text, "score": round(float(score), 2)})

        if results:
            return results

    return [{"text": "?", "score": 0.0}]


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
        normalized = _normalize_paddle_output(out)
        if normalized:
            results.append(normalized[0])

    return results


def ocr_full_image(img: np.ndarray) -> list[dict]:
    """
    OCR полного изображения через PaddleOCR.predict().
    Возвращает сырой результат PaddleOCR без нормализации.
    """
    if img is None or img.size == 0:
        return []

    try:
        ocr_result = paddle_ocr.predict(img)
    except Exception as e:
        print(f"⚠ Ошибка OCR.predict: {e}")
        return []

    return ocr_result
