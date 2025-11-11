# Mnemo OCR (demo)
Авто-оцифровка датчиков на технологических мнемосхемах (АСУ ТП) с помощью Python + OCR.

## Быстрый старт
1) `pip install -r requirements.txt`
2) Скопируйте `configs/config.example.yaml` в `configs/config.yaml` и при необходимости поправьте пути.
3) Откройте `notebooks/mnemo_ocr_final_v03_color_v01.ipynb` и запустите демо на `data/sample/input`.

## Что делает
- Обрабатывает изображения мнемосхем → извлекает подписи датчиков (OCR) → сохраняет результат в Excel.
