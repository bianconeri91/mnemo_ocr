import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Debug
import os
print("which tesseract:", os.popen("which tesseract").read())
print("version:", os.popen("tesseract --version").read())



import gradio as gr
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.pipeline_hf import process_single_image
from src.config_loader import load_config


# ================================
# Вспомогательная функция: подсветка сенсоров
# ================================
def draw_boxes(img_rgb, sensors):
    img = img_rgb.copy()

    for s in sensors:
        x, y, w, h = s["x"], s["y"], s["w"], s["h"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img


# ================================
# Основная функция демо
# ================================
def hf_process(img_rgb):

    cfg = load_config("configs/config.yaml")

    # Запуск пайплайна
    result = process_single_image(img_rgb, cfg_path="configs/config.yaml")

    title = result["title"]
    sensors = result["sensors"]

    # --- визуализация сенсоров ---
    boxed = draw_boxes(img_rgb, sensors)

    # --- DataFrame сенсоров ---
    df = pd.DataFrame(sensors)[["text", "score", "x", "y", "w", "h"]]

    # --- Excel на скачивание ---
    tmp = NamedTemporaryFile(delete=False, suffix=".xlsx")
    df.to_excel(tmp.name, index=False)

    return (
        boxed,
        title,
        df,
        tmp.name  # путь к скачиваемому файлу
    )


# ================================
# Gradio UI
# ================================
demo = gr.Interface(
    fn=hf_process,
    inputs=gr.Image(type="numpy", label="Upload image"),
    outputs=[
        gr.Image(label="Detected sensors"),
        gr.Textbox(label="Title"),
        gr.Dataframe(label="Recognized sensors"),
        gr.File(label="Download Excel")
    ],
    title="MNEMO OCR — HF Demo",
    description="Продовая демо-версия оцифровщика MNEMO."
)

if __name__ == "__main__":
    demo.launch()