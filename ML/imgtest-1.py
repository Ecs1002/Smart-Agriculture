from PIL import Image
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Load the saved model and label encoder
model = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

IMAGE_SIZE = (64, 64)

def predict_insect(image_path):
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB').resize((224, 224))  # resize to (224, 224)
    img_array = np.array(img).reshape(1, -1)

    # Use the model to predict the class of the image
    pred = model.predict(img_array)

    # Convert the predicted class back to the original label
    pred_label = label_encoder.inverse_transform(pred)

    return pred_label[0]

# Print the current working directory
print("Current working directory:", os.getcwd())

# List of image paths to test
image_paths_to_test = ['C:/Users/deepa/Downloads/locust_test_img1.png', 'C:/Users/deepa/Downloads/locust_test_img2.png', 'C:/Users/deepa/Downloads/test_img3.jpeg']

# Test the predict_insect function for each image
for image_path in image_paths_to_test:
    predicted_label = predict_insect(image_path)
    print(f"The predicted insect for {image_path} is: {predicted_label}")
