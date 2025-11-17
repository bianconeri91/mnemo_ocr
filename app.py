import streamlit as st
import numpy as np
import cv2
import pandas as pd
from pathlib import Path

from src.config_loader import load_config
from src.pipeline import extract_text
from src.docx_reader import docx_to_images

st.set_page_config(page_title="Mnemo OCR Demo", layout="wide")
st.title("🧠 Mnemo OCR — демонстрация")

CONFIG_PATH = Path("configs/config.yaml")
cfg = load_config(CONFIG_PATH)
color_ranges = cfg["colors"]

uploaded = st.file_uploader("Загрузите PNG/JPG/DOCX файл", type=["png", "jpg", "jpeg", "docx"])

if not uploaded:
    st.info("Загрузите изображение или DOCX-файл.")
    st.stop()

filename = uploaded.name.lower()

# ---- DOCX ----
if filename.endswith(".docx"):
    st.subheader("Документ DOCX")

    try:
        images = docx_to_images(uploaded)
    except Exception as e:
        st.error(f"⚠ DOCX нельзя обработать в этой среде.\n{e}")
        st.stop()

    st.write("Обнаружено страниц:", len(images))

    results_all = []

    for idx, page in enumerate(images):
        st.write(f"### Страница {idx+1}")

        img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

        with st.spinner("OCR..."):
            title_text, sensors = extract_text(img, color_ranges)

        st.write("**Титул:**", title_text)

        if sensors:
            df = pd.DataFrame(sensors)
            st.dataframe(df)
            results_all.append(df)
        else:
            st.info("Сенсоры не найдены.")

    if results_all:
        df_total = pd.concat(results_all, ignore_index=True)
        st.download_button("Скачать CSV", df_total.to_csv(index=False).encode(), "result.csv", "text/csv")

    st.stop()

# ---- PNG/JPG ----
else:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

    with st.spinner("OCR..."):
        title_text, sensors = extract_text(img, color_ranges)

    st.write("### Титул")
    st.write(title_text)

    st.write("### Сенсоры")
    if sensors:
        df = pd.DataFrame(sensors)
        st.dataframe(df)
        st.download_button(
            "Скачать CSV",
            df.to_csv(index=False).encode(),
            "sensors.csv",
            "text/csv"
        )
    else:
        st.info("Сенсоры не найдены.")
