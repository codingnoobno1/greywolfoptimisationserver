import paho.mqtt.client as mqtt
import json
from core.logger import logger
from config.settings import settings
from services.result_store import store

class MqttTelemetryClient:
    """
    Standardized MQTT Client for Telemetry & Commands.
    Video frames are no longer supported over MQTT for stability.
    """
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def start(self):
        try:
            logger.info(f"Connecting to MQTT Broker {settings.MQTT_BROKER}:{settings.MQTT_PORT} (Telemetry Mode)...")
            self.client.connect_async(settings.MQTT_BROKER, settings.MQTT_PORT, 60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Failed to start MQTT client: {e}")

    def stop(self):
        logger.info("Stopping MQTT Telemetry client...")
        self.client.loop_stop()
        self.client.disconnect()

    def publish(self, topic, message):
        """Helper to publish telemetry or commands"""
        if self.client.is_connected():
            self.client.publish(topic, message)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("MQTT Telemetry Connected.")
            # Subscribe to sensor topics or commands
            client.subscribe("greywolf/sensors/#")
            client.subscribe("greywolf/commands/#")
        else:
            logger.error(f"MQTT connection failed with code {rc}")

    def on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode()
            logger.info(f"MQTT Received [{msg.topic}]: {payload}")
            
            # Handle commands or sensor data
            if msg.topic == "greywolf/commands/mode":
                store.set_active_mode(payload)
            elif msg.topic == "greywolf/commands/source":
                # We could implement dynamic switching here too
                pass
                
        except Exception as e:
            logger.error(f"Error handling MQTT message: {e}")

mqtt_client = MqttTelemetryClient()
