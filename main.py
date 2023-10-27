import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from drone_control import control_drone
from image_preprocessing import preprocess_image

# Load pretrained models
locust_model = keras.models.load_model('locust_detection_model/locust_model.h5')
disease_model = keras.models.load_model('disease_detection_model/disease_model.h5')

# Initialize the camera
camera = cv2.VideoCapture(0)  # Use the default camera

while True:
    ret, frame = camera.read()

    # Preprocess the image
    target_size = (224, 224)  # Adjust to match model input size
    processed_frame = preprocess_image(frame, target_size)

    # Locust detection
    locust_probability = locust_model.predict(processed_frame)

    if locust_probability > 0.95:  # Adjust threshold as needed
        # Perform disease identification
        disease_probability = disease_model.predict(processed_frame)

        if disease_probability > 0.95:  # Adjust threshold as needed
            # Activate the drone to take action
            control_drone()

    cv2.imshow('Real-time Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
