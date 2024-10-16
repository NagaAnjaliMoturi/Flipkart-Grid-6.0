import cv2
from paddleocr import PaddleOCR
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Initialize PaddleOCR
ocr = PaddleOCR()

# Use tkinter to open a file dialog for image upload
Tk().withdraw()  # Prevents the root window from appearing
img_path = askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])

# Check if a file was selected
if img_path:
    # Perform OCR prediction
    result = ocr.ocr(img_path, rec=True)  # Set rec=True to get text recognition
    print("OCR Result:")

    # Accumulate the extracted text
    extracted_text = []
    for row in result[0]:
        text = row[1][0]  # Extract recognized text
        extracted_text.append(text)  # Append each text to the list

    # Join all extracted text into a single paragraph
    paragraph = ' '.join(extracted_text)
    print(paragraph)
else:
    print("No file selected.")
