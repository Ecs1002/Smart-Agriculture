import paho.mqtt.client as mqtt

# Define the MQTT broker's address and port
broker_address = "mqtt.eclipse.org"  # Replace with your MQTT broker address
port = 1883  # Default MQTT port

# Define the MQTT topic to subscribe to (must match the topic on the Raspberry Pi)
mqtt_topic = "drone_data"  # Replace with your desired topic

# Callback when the client connects to the MQTT broker
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    # Subscribe to the MQTT topic
    client.subscribe(mqtt_topic)

# Callback when a message is received from the MQTT topic
def on_message(client, userdata, msg):
    print(f"Received message: {msg.payload}")

# Initialize the MQTT client
client = mqtt.Client("TestServerSubscriber")

# Set the callback functions
client.on_connect = on_connect
client.on_message = on_message

# Connect to the MQTT broker
client.connect(broker_address, port)

# Keep the script running to receive messages
client.loop_forever()
