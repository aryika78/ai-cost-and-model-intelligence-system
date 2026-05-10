"""Parse uploaded documents (PDF, DOCX, TXT, images) to plain text."""

import io


def parse_document(file_bytes: bytes, file_type: str) -> str:
    """Parse document bytes into plain text.

    Args:
        file_bytes: Raw file content
        file_type: File extension (pdf, docx, txt, png, jpg, etc.)

    Returns:
        Extracted plain text
    """
    file_type = file_type.lower().strip(".")

    if file_type == "pdf":
        return _parse_pdf(file_bytes)
    elif file_type == "docx":
        return _parse_docx(file_bytes)
    elif file_type == "txt":
        return file_bytes.decode("utf-8", errors="replace")
    elif file_type in ("png", "jpg", "jpeg", "tiff", "bmp"):
        return _parse_image(file_bytes)
    else:
        return f"Unsupported file type: {file_type}"


def _parse_pdf(file_bytes: bytes) -> str:
    """Parse PDF using PyPDF2, fallback to OCR for scanned PDFs."""
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

        text = "\n".join(text_parts).strip()

        # If very little text extracted, try OCR
        if len(text) < 50:
            ocr_text = _ocr_pdf(file_bytes)
            if ocr_text and len(ocr_text) > len(text):
                return ocr_text

        return text if text else "Could not extract text from PDF."
    except Exception as e:
        return f"Error parsing PDF: {e}"


def _ocr_pdf(file_bytes: bytes) -> str:
    """OCR a scanned PDF using pytesseract."""
    try:
        from PIL import Image
        import pytesseract
        from PyPDF2 import PdfReader

        # Convert PDF pages to images and OCR
        # This is a simplified approach - for production, use pdf2image
        reader = PdfReader(io.BytesIO(file_bytes))
        text_parts = []

        for page in reader.pages:
            for image_obj in page.images:
                img = Image.open(io.BytesIO(image_obj.data))
                text = pytesseract.image_to_string(img)
                if text.strip():
                    text_parts.append(text)

        return "\n".join(text_parts).strip()
    except Exception:
        return ""


def _parse_docx(file_bytes: bytes) -> str:
    """Parse DOCX using python-docx."""
    try:
        from docx import Document

        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        return f"Error parsing DOCX: {e}"


def _parse_image(file_bytes: bytes) -> str:
    """OCR an image using pytesseract."""
    try:
        from PIL import Image
        import pytesseract

        img = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(img)
        return text.strip() if text.strip() else "No text found in image."
    except ImportError:
        return "OCR not available (pytesseract not installed). Please provide text directly."
    except Exception as e:
        return f"Error parsing image: {e}"
