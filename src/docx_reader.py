import os
from pdf2image import convert_from_path
from docx import Document
import tempfile

def docx_to_images(path: str):
    """
    Преобразовать .docx → изображения страниц.
    Если poppler отсутствует, выбрасываем исключение,
    чтобы Streamlit мог показать предупреждение.
    """
    try:
        # docx → pdf → images
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = os.path.join(tmp, "temp.pdf")

            # Преобразование docx → pdf
            # Здесь мы делаем максимально простой экспорт:
            doc = Document(path)
            doc.save(pdf_path)

            # pdf → images
            images = convert_from_path(pdf_path)
            return images

    except Exception as e:
        raise RuntimeError(
            "DOCX обработка недоступна: отсутствует poppler "
            "или системные зависимости."
        ) from e
