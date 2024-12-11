![image](https://github.com/user-attachments/assets/7b0d3b1f-c2f5-43e3-b395-5cb151f7e3de)## **Food Freshness and Expiry Date Detection**

This project utilizes Optical Character Recognition (OCR) and Convolutional Neural Networks (CNNs) to automatically detect the freshness and expiry dates of food items like fruits and vegetables. By improving the efficiency of data processing and ensuring food safety, this project aims to reduce food waste, improve operational efficiency, and enhance consumer protection.

## **Project Overview**
This project addresses two key challenges in the food industry:

Text Extraction from Images: Using PaddleOCR, it extracts relevant text from images, such as expiry dates and product details from food packaging.

Food Freshness Detection: It uses a CNN model (MobileNetV2) to classify food images as either fresh or rotten, helping reduce food waste.

The project integrates both image processing and natural language processing (OCR) to create a comprehensive solution for food safety and sustainability.

## **Key Features**
This application leverages advanced image processing and machine learning techniques to analyze product images and extract valuable insights. Built using Streamlit for the user interface and integrating PaddleOCR for text extraction, Keras for freshness detection, and custom algorithms for brand and expiry date identification, the tool provides three core functionalities:

-Text Extraction: Using PaddleOCR, the application extracts text from uploaded or captured images. It then identifies the presence of a brand from a predefined list using fuzzy matching and regular expressions.

Expiry Date Extraction: The app parses the extracted text to detect potential expiry dates or "Best Before" and "Use By" information. It interprets various date formats and computes expiry dates, factoring in months or years from the extracted date.

Freshness Detection: The application uses a pre-trained Keras model to evaluate the freshness of a product based on its image. The result categorizes the item as either "Fresh" or "Not Fresh."

Users interact with the application through a web interface that allows them to either upload an image or use the camera to capture it. Once an image is provided, users can select the desired analysis task (text extraction, expiry date, or freshness), and the system processes the image to provide detailed insights.

This tool is ideal for businesses in the FMCG, food, and health sectors, enabling them to automate the inspection and verification of products for attributes like brand identification, expiry dates, and freshness—key factors in quality control and product management.

## **Technical Approach**

1. Text Extraction
The PaddleOCR library is used to extract text from food packaging images. This OCR system works in multiple languages and can handle complex layouts.
OpenCV is used to preprocess images (e.g., resizing, noise reduction) before passing them to the OCR model for better accuracy.
Extracted text is concatenated into a single string, which is then parsed to identify expiry dates and other relevant details.

2. Expiry Date Detection
The concatenated text from the OCR process is analyzed using regular expressions (regex) to detect date formats like MM/YY, MM/DD/YYYY, and DD/MM/YYYY.
Valid dates are then converted into datetime objects, enabling comparisons (e.g., checking if the food is expired).
The first detected date is treated as the manufacture date, and subsequent dates are interpreted as expiry dates.

3. Freshness Detection
A MobileNetV2 model, fine-tuned for food freshness classification, is used to predict whether food items are fresh or rotten.
The model is trained on a labeled dataset of food images (fresh vs. rotten).
TensorFlow and Keras are used to build and train the model, utilizing data augmentation techniques for better generalization.

## **Usage**
To run the project locally, follow these steps:

1. Clone the repository:

   git clone https://github.com/NagaAnjaliMoturi/Flipkart-Grid-6.0

   cd Flipkart-Grid-6.0-main

2. Install dependencies:

   pip install -r requirements.txt

3. Run the application:

   python app.py

Once the application is running, you can upload an image via the GUI. The extracted text will be displayed, and the system will attempt to detect expiry dates and classify food freshness.

Example Output:
Extracted Text: "Expiry Date: 2024/12/31"
Freshness Status: "Fresh" or "Rotten"

## Libraries and Dependencies
This project depends on the following libraries:

OpenCV (opencv-python) for image processing.

PaddleOCR for extracting text from images.

NumPy for numerical operations and handling image data.

TensorFlow and Keras for deep learning and model training.

Tkinter for the GUI interface.

re for regular expression-based date extraction.

datetime for date parsing and validation.

To install all dependencies, run:

pip install -r requirements.txt

![image](https://github.com/user-attachments/assets/c86c8156-e202-4371-b4d6-726b595d4aeb)

## Text Extraction
![image](https://github.com/user-attachments/assets/0f22a72c-a3ea-4716-9dba-68d252a95706)

![image](https://github.com/user-attachments/assets/a66fda95-e818-4720-846a-5f9395c71cc2)

![image](https://github.com/user-attachments/assets/2c75a46e-1268-4655-ba17-5c09d36a8c4f)

![image](https://github.com/user-attachments/assets/68434317-1552-45c9-8693-066f203a1a24)



## Expiry date Extraction
![image](https://github.com/user-attachments/assets/a194e277-7eea-4e8a-9434-71664c6b4fb6)

![image](https://github.com/user-attachments/assets/efbcb8fb-59b3-42cd-97b1-36751d843d0b)



## Fruit Freshness
![image](https://github.com/user-attachments/assets/a60b6677-0ac0-496b-b717-5450d0df98c2)

![image](https://github.com/user-attachments/assets/37e148ae-5f0c-4686-9eba-2408adc67996)

![image](https://github.com/user-attachments/assets/b3bb2160-74ff-43a8-b178-fc250af12bdc)

![image](https://github.com/user-attachments/assets/4052205e-df1d-456e-8f1d-c05a42ad61a8)









