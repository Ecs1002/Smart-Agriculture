# Raspberry Pi to capture images or videos during drone flights.

import time
import picamera

# Initialize the camera
camera = picamera.PiCamera()

# Set the camera resolution and framerate (adjust as needed)
camera.resolution = (1920, 1080)  # Full HD resolution
camera.framerate = 30  # 30 frames per second

# Define the output file (image or video)
output_file = 'drone_flight.mp4'  # You can use .jpg for images

# Start recording video (or capturing images)
camera.start_recording(output_file)

# Define the duration of the flight or capture (adjust as needed)
flight_duration = 60  # 60 seconds

try:
    # Simulate a drone flight (replace this with your actual drone control code)
    print("Starting drone flight...")
    time.sleep(flight_duration)
    print("Drone flight completed.")

finally:
    # Stop recording (or capturing)
    camera.stop_recording()

# Release the camera
camera.close()
