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
def hf_process(img):

    cfg = load_config("configs/config.yaml")
    
    # Имя файла (если доступно)
    filename = getattr(img, "name", "uploaded_image.png")

    # Запуск пайплайна
    title, sensors = process_single_image(img, color_ranges=cfg["colors"])

    # --- визуализация сенсоров ---
    boxed = draw_boxes(img, sensors)

    # --- DataFrame сенсоров ---
    df = pd.DataFrame([
        {
            "filename": filename,
            "title": title,
            "text": s["text"],
            "score": s["score"],
            "x": s["x"],
            "y": s["y"],
            "w": s["w"],
            "h": s["h"]
        }
        for s in sensors
    ])

    # --- Excel на скачивание ---
    tmp = NamedTemporaryFile(delete=False, suffix=".xlsx")
    df.to_excel(tmp.name, index=False)

    return (
        boxed,
        title,
        tmp.name,  # путь к скачиваемому файлу
        df
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