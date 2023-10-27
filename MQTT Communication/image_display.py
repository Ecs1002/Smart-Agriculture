# MQTT subscriber to receive the image and display it

import paho.mqtt.client as mqtt
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# Define the MQTT broker's address and port
broker_address = "mqtt.eclipse.org"  # Replace with your MQTT broker address
port = 1883  # Default MQTT port

# Define the MQTT topic to subscribe to (must match the topic on the Raspberry Pi)
mqtt_topic = "drone_image"  # Replace with your desired topic

# Callback when the client connects to the MQTT broker
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    # Subscribe to the MQTT topic
    client.subscribe(mqtt_topic)

# Callback when a message is received from the MQTT topic
def on_message(client, userdata, msg):
    print("Received image data.")
    try:
        # Convert the received image data to a NumPy array
        image_data = np.frombuffer(msg.payload, dtype=np.uint8)
        
        # Decode the image using OpenCV
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        # Display the image using OpenCV
        cv2.imshow("Received Image", image)
        cv2.waitKey(0)

    except Exception as e:
        print(f"Error: {str(e)}")

# Initialize the MQTT client
client = mqtt.Client("TestServerImageSubscriber")

# Set the callback functions
client.on_connect = on_connect
client.on_message = on_message

# Connect to the MQTT broker
client.connect(broker_address, port)

# Keep the script running to receive messages
client.loop_forever()
