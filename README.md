## **Food Freshness and Expiry Date Detection**

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





1. Overview
The Product Insight application is a machine learning-based tool built with the help of computer vision and natural language processing (NLP) techniques. The application provides three main functionalities:

Text Extraction – Extracts and processes text (including brand names) from images using Optical Character Recognition (OCR).
Expiry Date Detection – Detects expiry dates from the extracted text and computes the expiry date of the product based on information like manufacturing date and best-before duration.
Freshness Detection – Classifies the freshness of a product (fresh or rotten) based on the visual characteristics of the product image, using a pre-trained machine learning model.
The application uses Streamlit for creating an interactive web interface and integrates several AI models and libraries for text extraction and image classification.

2. Features
Text Extraction:

Extracts readable text from images using OCR (PaddleOCR).
Detects brand names from extracted text.
Expiry Date Detection:

Detects expiry date information from extracted text (e.g., "BEST BEFORE", "USE BY").
Uses natural language processing to infer expiry dates and calculate the product’s expiry from manufacturing dates.
Freshness Detection:

Uses a pre-trained Convolutional Neural Network (CNN) model to classify images as "fresh" or "rotten".
Evaluates the freshness based on a predefined threshold for model output.
3. Functional Workflow
The Product Insight application operates through three distinct tasks, which users can select interactively:

Image Upload or Camera Input:

Users can either upload an image from their local machine or capture a new image using their webcam.
Text Extraction:

The uploaded image is processed using PaddleOCR to extract text.
After extracting the text, the application searches for brand names and displays them.
Expiry Date Extraction:

After text extraction, the system identifies expiry-related text (like "Best Before" or "Use By").
The system processes potential dates and calculates the expiry date based on the found data.
Freshness Detection:

A pre-trained CNN model evaluates the product's image to predict its freshness (whether it's "fresh" or "rotten").
The model’s output is used to classify the item’s freshness.
4. Workflow of the Application
Text Extraction Workflow

Input: Product image.
Process: OCR is applied to extract text from the image.
Output: Extracted text is shown. The application also tries to detect the product's brand name.
Flowchart:

plaintext
Copy code
               Start
                  ↓
        Image Upload or Capture
                  ↓
          Apply OCR to Image
                  ↓
  Text Extracted from Image (OCR Output)
                  ↓
  Check if Text Extraction was Successful
                  ↓
   Is Text Found in the Image? 
    ├── Yes: Extracted Text to Return
    └── No: Return "No Text Found"
                  ↓
           Output Extracted Text
                  ↓
                End
Expiry Date Extraction Workflow

Input: Extracted text from image.
Process: The application searches for patterns related to expiry dates, such as "Best Before", "Use By", and associated dates.
Output: The system computes and displays the expiry date.
Flowchart:

plaintext
Copy code
                   Start
                      ↓
           Image Upload or Capture
                      ↓
             Apply OCR to Extract Text
                      ↓
       Extracted Text Containing Date Info
                      ↓
 Check if Expiry Date Information is Found
                      ↓
   Is Expiry Date Found in Text? 
    ├── Yes: Parse Expiry Date from Text
    └── No: Return "Expiry Date Not Found"
                      ↓
   Calculate Expiry Date based on Text Information
                      ↓
                Output Expiry Date
                      ↓
                    End
Freshness Detection Workflow

Input: Product image.
Process: The image is preprocessed (resized, normalized) and passed through a pre-trained model for classification.
Output: The freshness classification result ("Fresh" or "Not Fresh") is displayed.
Flowchart:

plaintext
Copy code
                  Start
                     ↓
        Image Upload or Capture
                     ↓
        Image Preprocessing (Resize, Normalize)
                     ↓
            Model Inference (Image Prediction)
                     ↓
         Prediction (Fresh or Rotten Probability)
                     ↓
          Thresholding (Is prediction > 0.6?)
       ├── Yes: "The item is FRESH!"
       └── No: "The item is NOT FRESH."
                     ↓
              Output Freshness Result
                     ↓
                   End
5. Libraries and Tools Used
Streamlit: For building the web application and providing an interactive interface.
PaddleOCR: For Optical Character Recognition (OCR) to extract text from images.
OpenCV: For image manipulation, such as resizing and color conversion.
Keras & TensorFlow: For building and training the machine learning models (used for freshness detection).
FuzzyWuzzy: For approximate string matching to detect brand names from the extracted text.
Matplotlib: For visualizing the training progress of the model.
6. Model Details
Freshness Detection Model (CNN):

Type: Convolutional Neural Network (CNN).
Purpose: Classifies images into two categories: fresh and rotten.
Model Architecture: The model utilizes a MobileNetV2 backbone for feature extraction (pre-trained on ImageNet), followed by custom layers (Conv2D, SeparableConv2D, Dense layers) to classify the image.
Training: The model is trained on a labeled dataset of fresh and rotten product images.
Threshold: The prediction probability threshold for classification is set to 0.6.
OCR (PaddleOCR):

Purpose: To extract text from product packaging images.
Output: Extracted text is then processed to detect brand names and expiry date information.
7. Installation and Setup
Clone the repository:

bash
Copy code
git clone <repository-url>
Install the dependencies: You need to install the following libraries:

bash
Copy code
pip install streamlit paddlepaddle opencv-python tensorflow fuzzywuzzy matplotlib
Run the Application: To run the Streamlit app, navigate to the project directory and execute:

bash
Copy code
streamlit run app.py
8. Conclusion
The Product Insight application leverages computer vision and machine learning to extract valuable product information such as text, expiry dates, and freshness directly from product images. It provides a seamless experience for users to upload product images, extract relevant information, and classify the freshness of products in real time.



