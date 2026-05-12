import paho.mqtt.client as mqtt
import cv2
import numpy as np
from core.logger import logger
from config.settings import settings
from services.frame_queue import push_frame

class MqttIngestionClient:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def start(self):
        try:
            logger.info(f"Connecting to MQTT Broker {settings.MQTT_BROKER}:{settings.MQTT_PORT}...")
            # Use connect_async to prevent blocking the main thread during startup
            self.client.connect_async(settings.MQTT_BROKER, settings.MQTT_PORT, 60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Failed to start MQTT client: {e}")

    def stop(self):
        logger.info("Stopping MQTT client...")
        self.client.loop_stop()
        self.client.disconnect()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"Successfully connected to MQTT Broker.")
            client.subscribe(settings.MQTT_TOPIC)
            logger.info(f"Subscribed to topic: {settings.MQTT_TOPIC}")
        else:
            logger.error(f"MQTT connection failed with code {rc}")

    def on_message(self, client, userdata, msg):
        try:
            # Decode JPEG payload to numpy array
            np_arr = np.frombuffer(msg.payload, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Push immediately to queue without blocking
                push_frame(frame)
        except Exception as e:
            logger.error(f"Error decoding MQTT message: {e}")

mqtt_client = MqttIngestionClient()
