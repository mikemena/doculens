import fitz  # PyMuPDF for PDF
import os
from docx import Document  # python-docx
from logger import setup_logger

logger = setup_logger(__name__, include_location=True)


def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    logger.info(f"file extenstion: {ext}")
    if ext == ".pdf":
        doc = fitz.open(file_path)
        text = "".join(page.get_text() for page in doc)
        logger.info(f"text from pdf: {text}")
        doc.close()
        return text
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    else:
        logger.error("Unsupported file type")
        raise ValueError("Unsupported file type")


if __name__ == "__main__":
    logger.info("\n" + "=" * 70)
    logger.info("Starting text extraction...")
    text = extract_text("../data/raw/Service Agreement 4.pdf")
    logger.info(f"text: {text}")
