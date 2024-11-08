## **Food Freshness and Expiry Date Detection**

This project utilizes Optical Character Recognition (OCR) and Convolutional Neural Networks (CNNs) to automatically detect the freshness and expiry dates of food items like fruits and vegetables. By improving the efficiency of data processing and ensuring food safety, this project aims to reduce food waste, improve operational efficiency, and enhance consumer protection.

## **Project Overview**
This project addresses two key challenges in the food industry:

Text Extraction from Images: Using PaddleOCR, it extracts relevant text from images, such as expiry dates and product details from food packaging.

Food Freshness Detection: It uses a CNN model (MobileNetV2) to classify food images as either fresh or rotten, helping reduce food waste.

The project integrates both image processing and natural language processing (OCR) to create a comprehensive solution for food safety and sustainability.

## **Key Features**
OCR for Text Extraction: Extracts textual information such as expiry dates, product names, and other labels from food packaging images.

Expiry Date Detection: Detects and interprets expiry and manufacture dates using custom regex and date parsing.

Freshness Classification: Uses a CNN-based model to classify food items as "fresh" or "rotten" based on images.

Graphical User Interface (GUI): Allows easy image uploads through Tkinter and displays results directly to the user.

Data Visualization: Displays extracted text, expiry date detection results, and freshness classification directly in the GUI.

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

## Text Extraction
![image](https://github.com/user-attachments/assets/d67ee0c8-85da-4732-914c-225641f1764f)

![image](https://github.com/user-attachments/assets/e54034c6-4ff8-42f4-a08f-2631f91f7302)

![image](https://github.com/user-attachments/assets/a1fb401a-2a9a-4dd9-9e73-2a37e3843c8a)

![image](https://github.com/user-attachments/assets/3e4baf49-1bdf-4b40-9e0d-6e06c9d4dd4c)


## Expiry date Extraction
![image](https://github.com/user-attachments/assets/322e0d54-e514-422e-8a6d-b202c7095a57)

![image](https://github.com/user-attachments/assets/0aba5272-1eda-4fca-95c8-bd96277122bc)

![image](https://github.com/user-attachments/assets/bfe2b1ef-0988-4408-b7e5-a1dffd3dcc0c)

![image](https://github.com/user-attachments/assets/89aac366-1a2c-470d-ba74-e2230dc6cea4)


## Fruit Freshness
![image](https://github.com/user-attachments/assets/a60b6677-0ac0-496b-b717-5450d0df98c2)

![image](https://github.com/user-attachments/assets/fb1b4c09-4bf7-4dbc-ba2f-dea078851998)

![image](https://github.com/user-attachments/assets/5e0ed04d-d023-427e-94dc-d867afd3afa5)

![image](https://github.com/user-attachments/assets/94c6a5ec-7dff-468c-b962-e1107f75d6ae)








