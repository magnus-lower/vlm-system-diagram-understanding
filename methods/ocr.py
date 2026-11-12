import pytesseract
from PIL import Image


# This function extracts text from a diagram by using OCR
def run_ocr(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image).strip()


# Then, this function adds the OCR extracted text to the prompt
def add_ocr(prompt, ocr_text):
    return (
        f"{prompt}\n\n"
        "OCR text from the diagram:\n"
        f"{ocr_text}\n\n"
        "The OCR text may contain errors. "
        "Use the image as the main source."
    )
