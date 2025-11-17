import gradio as gr
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

# Инициализируем OCR один раз
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en'  # если нужно rus — скажи
)

def run_ocr(image):
    if image is None:
        return "No image uploaded."

    # Конвертируем PIL → numpy
    img = np.array(image)

    # Запуск OCR
    result = ocr.ocr(img)

    # Формируем вывод текстом
    lines = []
    for block in result:
        for line in block:
            text = line[1][0]
            conf = line[1][1]
            lines.append(f"{text} (conf: {conf:.2f})")

    return "\n".join(lines)

demo = gr.Interface(
    fn=run_ocr,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Recognized Text"),
    title="MNEMO OCR Demo",
    description="Upload an image and extract text using PaddleOCR.",
)

if __name__ == "__main__":
    demo.launch(share=True, show_api=False)
