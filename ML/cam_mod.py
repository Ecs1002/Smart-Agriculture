import time
import picamera
from PIL import Image
import numpy as np

# Load your machine learning model here
# model = load_model()

def detect_locust(image):
    # Implement your locust detection logic here
    # For example:
    # prediction = model.predict(image)
    # return prediction == 'locust'
    pass

with picamera.PiCamera() as camera:
    camera.start_preview()
    try:
        for i in range(15):  # 30 seconds / 2 seconds = 15 iterations
            # Capture into in-memory stream
            stream = io.BytesIO()
            camera.capture(stream, format='jpeg')
            stream.seek(0)
            image = Image.open(stream)

            if detect_locust(np.array(image)):
                camera.capture('/home/pi/Desktop/image%s.jpg' % i)
                print("Locust detected. Image captured.")
            time.sleep(2)
    finally:
        camera.stop_preview()