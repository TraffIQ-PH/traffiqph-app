# publisher.py
import json
import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "traffiq/test"

def publish(data: dict):
    """Publishes a Python dictionary as JSON to the MQTT topic."""
    client = mqtt.Client()
    client.connect(BROKER, PORT)
    client.publish(TOPIC, json.dumps(data))
    client.disconnect()
    print(f"Published: {data}")

if __name__ == "__main__":
    # Example usage
    green_a = {
    "a_leds": [0, 0, 1],
    "b_leds": [1, 0, 0],
    "c_leds": [1, 0, 0],
    "d_leds": [1, 0, 0]
    }
    green_b = {
    "a_leds": [1, 0, 0],
    "b_leds": [0, 0, 1],
    "c_leds": [1, 0, 0],
    "d_leds": [1, 0, 0]
    }
    green_c = {
    "a_leds": [1, 0, 0],
    "b_leds": [1, 0, 0],
    "c_leds": [0, 0, 1],
    "d_leds": [1, 0, 0]
    }
    green_d = {
    "a_leds": [1, 0, 0],
    "b_leds": [1, 0, 0],
    "c_leds": [1, 0, 0],
    "d_leds": [0, 0, 1]
    }

    publish(green_b)
