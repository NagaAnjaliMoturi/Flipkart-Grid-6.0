import streamlit as st
import os
import numpy as np
from paddleocr import PaddleOCR
import re
from datetime import datetime
from keras.models import load_model
import tensorflow as tf
from PIL import Image
import io
from dateutil.relativedelta import relativedelta
import cv2
from fuzzywuzzy import fuzz

# Disable GPU
tf.config.set_visible_devices([], 'GPU')

# Initialize PaddleOCR
ocr = PaddleOCR()

# List of company names (brands)
BRANDS_LIST = [
    "Himalaya", "Dabur", "Baidyanath", "Patanjali", "Zandu", "Biotique", "Khadi Natural", "The Body Shop", "Forest Essentials", "Lotus Herbals", "VLCC", 
    "Mederma", "Glenmark", "Reckitt Benckiser", "Kama Ayurveda", "Jovees", "Santoor", "Aveda", "Neemli Naturals", "Oriflame", "Livon", "Olay", "Garnier", 
    "Nivea", "Cetaphil", "Pond's", "Vaseline", "Nutralite", "Bioderma", "Lacto Calamine", "Suncross", "Head & Shoulders", "Fair and Lovely", "Dove", "Beardo", 
    "Wild Stone", "Engage", "Old Spice", "Secret Temptation", "Biore", "Amway", "Aroma Magic", "WOW Skin Science", "Hindustan Unilever", "Shahnaz Husain", 
    "Khadi Essentials", "Cleanses", "Neem Active", "Livon Hair Gain Tonic", "L'Oreal", "Fogg", "Fena", "Sunsilk", "Pantene", "TRESemmé", "Parachute", 
    "Godrej No. 1", "Lux", "Revlon", "Lakmé", "Maybelline", "M.A.C", "Elle 18", "Burt's Bees", "Kiehl's", "Sally Hansen", "Clinique", "Avon", 
    "Streetwear", "Rexona", "Lifebuoy", "Vicco", "Amrutanjan", "Tata Salt", "Amul", "Mother Dairy", "Britannia", "Parle", "Britannia Good Day", "MTR Foods", 
    "Haldiram’s", "Bikanervala", "Tata Tea", "Tata", "Tata Coffee", "Lipton", "Nestlé", "Maggi", "Kellogg’s", "Saffola", "Fortune", "Patanjali Ghee", "Vim", 
    "Cif", "Harpic", "Dettol", "Vim Dishwash", "Tata Sampann", "Bunge", "Nature Fresh", "KTC", "Annapurna", "Amul Butter", "Dabur Honey", "Kisan", "Tata Salt", 
    "Fortune Sunflower Oil", "Mahakosh", "Tata Power", "Coca Cola", "PepsiCo", "Bisleri", "Mountain Dew", "Thums Up", "Aquafina", "Kinley", "Tata Water", 
    "Sprite", "Whirlpool", "Panasonic", "Bajaj", "Blue Star", "Voltas", "LG", "Philips", "IFB", "Crompton Greaves", "Daikin", "Bosch", "Reliance Trends", 
    "Lifestyle", "Westside", "Shoppers Stop", "Big Bazaar", "Pantaloon", "H&M", "Zara", "FabIndia", "Max Fashion", "BIBA", "Van Heusen", "Allen Solly", "Peter England", 
    "Arrow", "Adidas", "Nike", "Puma", "Reebok", "Wildcraft", "Campus Shoes", "Skechers", "Liberty", "Bata", "Khadim’s", "Relaxo", "Myntra", "Ajio", 
    "Amazon", "Snapdeal", "ShopClues", "Pepperfry", "UrbanClap", "Paytm", "PhonePe", "Google Pay", "Razorpay", "MobiKwik", "CRED", "FreeCharge", "BharatPe", "PayPal", 
    "Zomato", "Grofers", "BigBasket", "Tata CLiQ", "Zappos", "Bigbazaar", "Hathway", "ACT Fibernet", "Airtel", "Jio", "BSNL", "MTNL", "Ola", "Uber", 
    "Zoomcar", "Johnson & Johnson", "Procter & Gamble", "Unilever", "Colgate-Palmolive", "Danone", "Mondelez", "Gillette", "Oral-B", "Oreo", "Pringles", "Toblerone", 
    "KitKat", "Cadbury", "Chocolates", "Quaker Oats", "Almonds", "Rimmel", "Hershey's", "Kraft", "Lipton", "Tetley", "Lipton Green Tea", "Red Bull", "Vita", "Cadbury Silk", 
    "Mars", "Bournvita", "Peach & Mango", "Nestle Milk", "Del Monte", "McVities", "Kraft Heinz", "Goodday", "Parle-G", "Saffola", "Nature Valley", "Nescafé", "Pillsbury", 
    "Ritz Crackers", "Bikano", "Vijay Enterprises", "Emami"
]

# Functionality from ml.py (text extraction)
def extract_text(img_path):
    try:
        result = ocr.ocr(img_path, rec=True)
        
        # Log result to see what the OCR returns
        if result is None:
            return "OCR returned None. No text could be detected."

        # Check if the result is an empty list
        if len(result) == 0 or len(result[0]) == 0:
            return "No text found in the image."

        extracted_text = []
        for row in result[0]:
            text = row[1][0]
            extracted_text.append(text)

        extracted_text = ' '.join(extracted_text)
        
        # Detect brand name from the extracted text
        brand_name = detect_brand(extracted_text)
        
        return extracted_text, brand_name
    
    except Exception as e:
        # Handle unexpected errors
        return f"An error occurred during text extraction: {str(e)}", None

def clean_text(text):
    # Remove special characters, numbers, and extra spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only alphabetic characters and spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

def detect_brand(extracted_text):
    extracted_text = extracted_text.lower()  # Convert the extracted text to lowercase
    cleaned_text = clean_text(extracted_text)  # Clean up the extracted text
    
    # Check if any brand is directly mentioned in the cleaned text using regex
    for brand in BRANDS_LIST:
        brand_lower = brand.lower()  # Convert each brand name to lowercase
        
        # Direct match using regular expressions
        if re.search(r'\b' + re.escape(brand_lower) + r'\b', cleaned_text):
            return f"Brand name: {brand}"

        # Fuzzy matching as a fallback if regex doesn't find a direct match
        if fuzz.partial_ratio(brand_lower, cleaned_text) > 80:
            return f"Brand name: {brand}"

    return "Brand name: Not found"



def parse_text(extracted_text):
    # List of regular expression patterns for different types of dates
    patterns = [
        r"\b[A-Za-z]{3}\d{4}\b",  # FEB2014, JAN2022 (e.g., month + year)
        r'\b(\d{2})/(\d{2,4})\b',  # MM/YY or MM/YYYY format (e.g., 02/24, 12/2024)
        r'#?(\d{2})/(\d{2,4})[A-Za-z0-9]*\b',  # MM/YY or MM/YYYY followed by a batch code (e.g., #06/24B404)
        r'\b(\d{2})/(\d{2})/(\d{4})\b',  # dd/mm/yyyy format (e.g., 31/12/2024)
        r"BEST BEFORE\s*(three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s*(months?|mo|mos?|mths?)",  # Best before with time (e.g., BEST BEFORE 6 months)
        r"USE BY\s*(three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s*(months?|mo|mos?|mths?)"  # Use by with time (e.g., USE BY 3 months)
    ]
    
    potential_dates = []
    
    # Find all matches based on the patterns
    for pattern in patterns:
        potential_dates.extend(re.findall(pattern, extracted_text, flags=re.IGNORECASE))
    
    # Initialize variables
    manufacture_date, expiry_date = "Not found", "Not found"
    best_before_found = False
    valid_dates = []
    
    # Parse the potential dates and add them to valid_dates
    for date in potential_dates:
        parsed_date = None
        
        try:
            if isinstance(date, str) and len(date) == 7:  # Handle month-year format (e.g., FEB2014, JAN2022)
                parsed_date = datetime.strptime(date, '%b%Y')
            elif isinstance(date, tuple) and len(date) == 2:  # Strict month/year format (e.g., 06/2023)
                month, year = date
                if len(year) == 2:  # If it's YY, convert it to YYYY
                    year = '20' + year
                parsed_date = datetime.strptime(f'{month}/{year}', '%m/%Y')
            elif isinstance(date, tuple) and len(date) == 2:  # Batch code case (e.g., #06/24B404)
                month, year = date
                if len(year) == 2:  # If it's YY, convert it to YYYY
                    year = '20' + year
                parsed_date = datetime.strptime(f'{month}/{year}', '%m/%Y')
            elif isinstance(date, tuple) and len(date) == 3:  # Handle dd/mm/yyyy format (e.g., 31/12/2024)
                day, month, year = date
                parsed_date = datetime.strptime(f'{day}/{month}/{year}', '%d/%m/%Y')
            
            # Append parsed_date to valid_dates if it was successfully parsed
            if parsed_date:
                valid_dates.append(parsed_date)
        except ValueError:
            continue
    
    # If no valid dates were found, return 'Not found'
    if valid_dates:
        manufacture_date = valid_dates[0].strftime('%b%Y')  # Assume the first date is the manufacture date
    else:
        return {'Expiry Date': 'Not found'}

    # If potential "Best Before" or "Use By" information exists
    best_before_match = re.search(r"(BEST BEFORE|USE BY)\s*(three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s*(months?|mo|mos?|mths?)", extracted_text, re.IGNORECASE)
    
    if best_before_match:
        best_before_found = True
        months_to_add_str = best_before_match.group(2)
        
        # Mapping words to months
        month_mapping = {
            'three': 3,
            'four': 4,
            'five': 5,
            'six': 6,
            'seven': 7,
            'eight': 8,
            'nine': 9,
            'ten': 10,
            'eleven': 11,
            'twelve': 12
        }
        
        # Check if it's a word (e.g., 'three') or a number (e.g., '6')
        if months_to_add_str.isdigit():
            months_to_add = int(months_to_add_str)
        else:
            months_to_add = month_mapping.get(months_to_add_str.lower(), 0)  # Default to 0 if not found

        # Calculate expiry date by adding months
        manufacture_date_obj = valid_dates[0]  # Assume the first valid date is the manufacture date
        expiry_date_obj = manufacture_date_obj + relativedelta(months=months_to_add)
        expiry_date = expiry_date_obj.strftime('%b%Y')
    else:
        # If no Best Before/Use By found, choose the most recent date from the valid dates
        expiry_date_obj = max(valid_dates)
        expiry_date = expiry_date_obj.strftime('%b%Y')

    return {'Expiry Date': expiry_date}

# Freshness detection functionality
def classify_freshness(res):
    threshold_fresh = 0.6
    return "The item is FRESH!" if res < threshold_fresh else "The item is NOT FRESH."

# Image preprocessing
def preprocess_image(image):
    # Resize the image to 100x100
    img = np.array(image)
    img = cv2.resize(img, (100, 100))  # Resize to the required dimensions (100x100)
    
    # Normalize pixel values to [0, 1]
    img = img / 255.0
    
    # Ensure the image has the correct shape (1, 100, 100, 3)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    return img

# Model evaluation (freshness detection)
def evaluate_image(image, model_path='rottenvsfresh.keras'):
    model = load_model(model_path)
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)
    return prediction[0][0]

# Streamlit Interface
st.title('Product Insight')

# Notify user about camera permissions
st.write(
    "Please make sure to allow the camera access in your browser when prompted for the camera. "
    "If the camera doesn't work, ensure that no other application is using it and refresh the page."
)

# Option for camera input or file upload
option = st.radio('Choose input method:', ('Upload Image', 'Use Camera'))

if option == 'Use Camera':
    # Streamlit camera input widget
    image = st.camera_input("Capture Image")

    if image is not None:
        # Convert the captured image into a format that can be processed
        img = Image.open(io.BytesIO(image.getvalue()))  # Use .getvalue() to get byte data from the UploadedFile object
        img.save('captured_image.jpg')
        
        # Ask user to choose a task after the image is captured
        task_option = st.radio('Choose a task:', ('Text Extraction', 'Expiry Date Extraction', 'Freshness Detection'))

        # Perform tasks based on the option selected
        if task_option == 'Text Extraction':
            extracted_text, brand_name = extract_text('captured_image.jpg')
            st.write(f"Extracted Text: {extracted_text}")
            st.write(brand_name)

        elif task_option == 'Expiry Date Extraction':
            extracted_text, _ = extract_text('captured_image.jpg')
            details = parse_text(extracted_text)
            st.write(f"Expiry Date: {details['Expiry Date']}")

        elif task_option == 'Freshness Detection':
            image_array = np.array(img)
            prediction = evaluate_image(image_array)
            freshness_result = classify_freshness(prediction)
            st.write(freshness_result)

elif option == 'Upload Image':
    # File upload section for image
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

    if uploaded_file:
        img_path = os.path.join('static', uploaded_file.name)
        with open(img_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Ask user to choose a task after the image is uploaded
        task_option = st.radio('Choose a task:', ('Text Extraction', 'Expiry Date Extraction', 'Freshness Detection'))
        
        # Perform tasks based on the option selected
        if task_option == 'Text Extraction':
            extracted_text, brand_name = extract_text(img_path)
            st.write(f"Extracted Text: {extracted_text}")
            st.write(brand_name)

        elif task_option == 'Expiry Date Extraction':
            extracted_text, _ = extract_text(img_path)
            details = parse_text(extracted_text)
            st.write(f"Expiry Date: {details['Expiry Date']}")

        elif task_option == 'Freshness Detection':
            image = Image.open(img_path)
            image_array = np.array(image)
            prediction = evaluate_image(image_array)
            freshness_result = classify_freshness(prediction)
            st.write(freshness_result)
