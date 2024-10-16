from flask import Flask, request, render_template
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
from datetime import datetime
from keras.models import load_model
import tensorflow as tf

# Disable GPU
tf.config.set_visible_devices([], 'GPU')
app = Flask(__name__)
ocr = PaddleOCR()

# Functionality from ml.py
def extract_text(img_path):
    result = ocr.ocr(img_path, rec=True)
    extracted_text = []
    for row in result[0]:
        text = row[1][0]
        extracted_text.append(text)
    return ' '.join(extracted_text)

# Functionality from mlhive.py
def parse_text(extracted_text):
    date_pattern = r'(\d{1,2}[/-]?\d{2,4})'
    potential_dates = re.findall(date_pattern, extracted_text)
    manufacture_date, expiry_date = "Not found", "Not found"
    
    if potential_dates:
        for date in potential_dates:
            try:
                if len(date) == 7:
                    parsed_date = datetime.strptime(date, '%m/%Y')
                elif len(date) == 5:
                    parsed_date = datetime.strptime(date, '%m/%y')
                else:
                    continue

                if manufacture_date == "Not found":
                    manufacture_date = date
                else:
                    expiry_date = date

            except ValueError:
                continue

    return {'Expiry Date': expiry_date, 'Manufacture Date': manufacture_date}

# Freshness detection functionality
def classify_freshness(res):
    threshold_fresh = 0.6
    return "The item is FRESH!" if res < threshold_fresh else "The item is NOT FRESH."

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def evaluate_image(image_path, model_path='rottenvsfresh.keras'):
    model = load_model(model_path)
    preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    return prediction[0][0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        option = request.form.get('option')
        file = request.files['image']
        if file:
            img_path = os.path.join('static', file.filename)
            file.save(img_path)
            
            if option == 'text_extraction':
                result = extract_text(img_path)
                return render_template('index.html', result=result)
            elif option == 'expiry_date_extraction':
                extracted_text = extract_text(img_path)
                details = parse_text(extracted_text)
                return render_template('index.html', result=details)
            elif option == 'freshness_detection':
                prediction = evaluate_image(img_path)
                freshness_result = classify_freshness(prediction)
                return render_template('index.html', result=freshness_result)

    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)
