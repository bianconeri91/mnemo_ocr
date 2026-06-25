"""
Сохранение промежуточных результатов маскирования и выделения ROI.
"""

import re
from pathlib import Path

import cv2
import numpy as np


def build_color_mask(img: np.ndarray, color_ranges: dict) -> np.ndarray:
    """Строит общую бинарную маску для всех заданных HSV-диапазонов."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for lo, hi in color_ranges.values():
        lower = np.array(lo, dtype=np.uint8)
        upper = np.array(hi, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    return mask


def extract_rois(
    img: np.ndarray,
    mask: np.ndarray,
    min_width: int = 90,
    min_height: int = 17,
    max_height: int = 17,
) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    """Вырезает ROI по внешним контурам маски и возвращает их координаты."""
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    rois = []
    positions = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < min_width or height < min_height:
            continue

        height = min(height, max_height)
        roi = img[y:y + height, x:x + width]
        if roi.size == 0:
            continue

        rois.append(roi)
        positions.append((x, y, width, height))

    return rois, positions


def save_mask(mask: np.ndarray, image_name: str, masks_dir: Path) -> Path:
    """Сохраняет бинарную маску исходного изображения."""
    masks_dir.mkdir(parents=True, exist_ok=True)
    output_path = masks_dir / f"{_safe_name(image_name)}_mask.png"
    cv2.imwrite(str(output_path), mask)
    return output_path


def save_rois(
    rois: list[np.ndarray],
    positions: list[tuple[int, int, int, int]],
    image_name: str,
    rois_dir: Path,
) -> list[Path]:
    """Сохраняет каждую вырезанную ROI отдельным PNG-файлом."""
    rois_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_name(image_name)
    saved_paths = []

    for index, (roi, (x, y, width, height)) in enumerate(
        zip(rois, positions),
        start=1,
    ):
        output_path = rois_dir / (
            f"{safe_name}_roi_{index:03d}_"
            f"x{x}_y{y}_w{width}_h{height}.png"
        )
        cv2.imwrite(str(output_path), roi)
        saved_paths.append(output_path)

    return saved_paths


def _safe_name(value: str) -> str:
    """Убирает из имени символы, небезопасные для имени файла."""
    return re.sub(r"[^A-Za-zА-Яа-я0-9_.-]+", "_", value).strip("._") or "image"
