import paho.mqtt.client as mqtt
import time

# Define the MQTT broker's address and port
broker_address = "mqtt.eclipse.org"  # Replace with your MQTT broker address
port = 1883  # Default MQTT port

# Define the MQTT topic to publish to
mqtt_topic = "drone_data"  # Replace with your desired topic

# Initialize the MQTT client
client = mqtt.Client("RaspberryPiPublisher")

# Connect to the MQTT broker
client.connect(broker_address, port)

while True:
    # Capture data from the drone (replace with your actual data)
    drone_data = "Data from the drone"

    # Publish data to the MQTT topic
    client.publish(mqtt_topic, drone_data)

    print(f"Published: {drone_data} to topic {mqtt_topic}")
    
    # Sleep for a specified time interval (adjust as needed)
    time.sleep(5)

# Disconnect from the MQTT broker (optional, as it will happen when the program ends)
# client.disconnect()
