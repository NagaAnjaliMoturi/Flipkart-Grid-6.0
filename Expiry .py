import cv2
import numpy as np
from paddleocr import PaddleOCR
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import re
from datetime import datetime

# Initialize PaddleOCR
ocr = PaddleOCR()

# Function to parse extracted text
def parse_text(extracted_text):
    date_pattern = r'(\d{1,2}[/-]?\d{2,4})'  # Matches MM/YY, MM/YYYY, and variations

    # Find all potential date matches
    potential_dates = re.findall(date_pattern, extracted_text)
    
    manufacture_date, expiry_date = "Not found", "Not found"
    
    if potential_dates:
        # Try to interpret dates from potential matches
        for date in potential_dates:
            try:
                # Normalize the date for consistent comparison
                if len(date) == 7:  # Format MM/YYYY
                    parsed_date = datetime.strptime(date, '%m/%Y')
                elif len(date) == 5:  # Format MM/YY
                    parsed_date = datetime.strptime(date, '%m/%y')
                else:
                    continue  # Skip if the format is not recognized

                # Assign dates based on some logic
                if manufacture_date == "Not found":
                    manufacture_date = date
                else:
                    expiry_date = date  # Assign later found dates as expiry

            except ValueError:
                continue  # Ignore any parsing errors

    details = {
        'Expiry Date': expiry_date,
        'Manufacture Date': manufacture_date,
    }

    return details

def validate_expiry(expiry_date_str):
    if expiry_date_str == "Not found":
        return "Expiry date not available for validation."

    try:
        if len(expiry_date_str) == 7:  # Format MM/YYYY
            expiry_date = datetime.strptime(expiry_date_str, '%m/%Y')
        elif len(expiry_date_str) == 5:  # Format MM/YY
            expiry_date = datetime.strptime(expiry_date_str, '%m/%y')
        else:
            return "Invalid expiry date format."

    except ValueError:
        return "Invalid expiry date format."

    if expiry_date < datetime.now():
        return "Expired"
    else:
        return "Valid"

# Use tkinter to open a file dialog for image upload
Tk().withdraw()  # Prevents the root window from appearing
img_path = askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])

# Check if a file was selected
if img_path:
    # Perform OCR prediction
    result = ocr.ocr(img_path, rec=True)  # Set rec=True to get text recognition
    print("OCR Result:", result)

    # Read the image using OpenCV
    image = cv2.imread(img_path)

    # Draw bounding boxes and text on the image (optional, if you want to keep this part without saving)
    for row in result[0]:
        bbox = [[int(r[0]), int(r[1])] for r in row[0]]  # Extract bounding box coordinates
        text = row[1][0]  # Extract recognized text
        # Draw bounding box
        cv2.polylines(image, [np.array(bbox)], True, (255, 0, 0), 1)
        # Place text on the image
        cv2.putText(image, text, (bbox[0][0], bbox[0][1] - 10), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 0), 1)

    # Parse the extracted text
    concat_output = "\n".join(row[1][0] for row in result[0])  # Concatenate for parsing
    details = parse_text(concat_output)
    print("Parsed Details:", details)

    # Validate expiry date
    expiry_status = validate_expiry(details['Expiry Date'])
    print("Expiry Status:", expiry_status)

else:
    print("No file selected.")
