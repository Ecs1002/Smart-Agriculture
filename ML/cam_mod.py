import io
import time
import picamera
import RPi.GPIO as GPIO
import joblib
from PIL import Image
import numpy as np

# Load the trained model and label encoder
model = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define the predict_insect function
def predict_insect(image):
    # Preprocess the image
    img = image.convert('RGB').resize((224, 224))  # resize to (224, 224)
    img_array = np.array(img).reshape(1, -1)

    # Use the model to predict the class of the image
    pred = model.predict(img_array)

    # Convert the predicted class back to the original label
    pred_label = label_encoder.inverse_transform(pred)
    
    return pred_label[0]

# Define the dispenser pin
dispenser_pin = 18

# Set up the GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(dispenser_pin, GPIO.OUT)

# Start dispensing
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

            if predict_insect(image) == 'locust':
                camera.capture('/home/pi/Desktop/image%s.jpg' % i)
                print("Locust detected. Image captured.")
    finally:
        camera.stop_preview()
