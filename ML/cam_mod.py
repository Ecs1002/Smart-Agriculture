import time
import picamera
from PIL import Image
import numpy as np
import joblib
from skimage.feature import hog
import RPi.GPIO as GPIO

# Load your machine learning model here
model = joblib.load('model.pkl')

# Set GPIO mode and pin
GPIO.setmode(GPIO.BOARD)
dispenser_pin = 17  # Replace with your actual GPIO pin

# Set up dispenser pin as output
GPIO.setup(dispenser_pin, GPIO.OUT)

# Function to extract HOG features from an image
def extract_hog_features(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((224, 224))
    img_array = np.array(img)
    
    # Calculate HOG features and return flattened array
    features, _ = hog(img_array, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True)
    return features

def detect_locust(image):
    # Extract HOG features
    features = extract_hog_features(image)
    # Predict using the model
    prediction = model.predict([features])
    # Return True if locust is detected, False otherwise
    return prediction[0] == 1

def dispense_liquid():
    GPIO.output(dispenser_pin, GPIO.HIGH)
    print("Dispensing liquid...")
    time.sleep(5)  # Adjust the time as needed
    GPIO.output(dispenser_pin, GPIO.LOW)
    print("Dispensing complete.")

with picamera.PiCamera() as camera:
    camera.start_preview()
    try:
        for i in range(15):  # 30 seconds / 2 seconds = 15 iterations
            # Capture into in-memory stream
            stream = io.BytesIO()
            camera.capture(stream, format='jpeg')
            stream.seek(0)
            image = Image.open(stream)

            if detect_locust(image):
                camera.capture('/home/pi/Desktop/image%s.jpg' % i)
                print("Locust detected. Image captured.")
                dispense_liquid()
            time.sleep(2)
    finally:
        camera.stop_preview()
        GPIO.cleanup()  # Clean up GPIO state
