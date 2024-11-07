## **Food Freshness and Expiry Date Detection**

This project utilizes Optical Character Recognition (OCR) and Convolutional Neural Networks (CNNs) to automatically detect the freshness and expiry dates of food items like fruits and vegetables. By improving the efficiency of data processing and ensuring food safety, this project aims to reduce food waste, improve operational efficiency, and enhance consumer protection.

**Table of Contents**

Project Overview

Key Features

Technical Approach

Usage

Libraries and Dependencies

Model Training

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
bash
Copy code
git clone https://github.com/yourusername/food-freshness-expiry-detection.git
cd food-freshness-expiry-detection
2. Create and activate a virtual environment (recommended):
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. Install dependencies:
bash
Copy code
pip install -r requirements.txt
4. Run the application:
bash
Copy code
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

bash
Copy code
pip install -r requirements.txt

## Model Training
1. Dataset Preparation
The dataset contains images of food items labeled as either "fresh" or "rotten."
The images are preprocessed (resized, normalized) to match the input size required by the model.
Data augmentation techniques (e.g., rotation, flipping) are applied to increase the variety of training data and improve the model's robustness.
2. Model Architecture
A pre-trained MobileNetV2 model is used as the backbone for the food freshness classification task. Additional layers (Batch Normalization, Dropout, etc.) are added for fine-tuning the model.
The model is trained using binary cross-entropy loss and the Adam optimizer.
3. Training Process
The model is trained for several epochs, with callbacks used to adjust the learning rate and save the best model during training.
The model is evaluated using a validation set to assess performance and ensure it generalizes well to unseen data.
4. Retraining
If you want to retrain the model with your own dataset, follow the instructions in training.py to prepare the data and retrain the model.
