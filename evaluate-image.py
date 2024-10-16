import cv2
import numpy as np
from keras.models import load_model

# Classify fresh/rotten based on prediction
def classify_freshness(res):
    threshold_fresh = 0.6  # Adjust according to your model's output range
    if res < threshold_fresh:
        print("The item is FRESH!")
    else:
        print("The item is NOT FRESH ")

# Preprocess the input image for prediction
def preprocess_image(image_path):
    # Read and process the image using OpenCV
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100))  # Resizing to the shape expected by the model
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = img / 255.0  # Normalize pixel values to the range [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Evaluate the model for a given image
def evaluate_image(image_path, model_path='rottenvsfresh.keras'):
    # Load the trained model
    model = load_model(model_path)

    # Preprocess the image
    preprocessed_img = preprocess_image(image_path)

    # Predict the freshness
    prediction = model.predict(preprocessed_img)

    # Return the predicted result
    return prediction[0][0]

# Example usage:
if __name__ == "__main__":
    # Path to the image for testing
    test_img_path = 'greenbearotten.jpg'  # Change to your test image path

    # Evaluate the image using the model
    prediction = evaluate_image(test_img_path)

    # Output the prediction result
    print(f'Raw Prediction Score: {prediction}')
    classify_freshness(prediction)
