import os

class Settings:
    # MQTT Settings
    MQTT_BROKER = os.getenv('MQTT_BROKER', 'localhost')
    MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
    MQTT_TOPIC = os.getenv('MQTT_TOPIC', 'esp32/camera')
    
    # Queue and Processing
    FRAME_QUEUE_MAXSIZE = int(os.getenv('FRAME_QUEUE_MAXSIZE', 10))
    NUM_WORKERS = int(os.getenv('NUM_WORKERS', 2))
    USE_WEBCAM = os.getenv('USE_WEBCAM', 'false').lower() == 'true'
    
    # Model Settings
    DEFAULT_ACTIVE_MODE = os.getenv('DEFAULT_ACTIVE_MODE', 'all') # "yolo", "mediapipe", "all", "none"
    YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'yolov8n.pt')
    
    # Streaming Settings
    FPS_LIMIT = int(os.getenv('FPS_LIMIT', 15))
    JPEG_QUALITY = int(os.getenv('JPEG_QUALITY', 70))
    
    # Mobile Specific
    MOBILE_FPS = int(os.getenv('MOBILE_FPS', 20))
    MOBILE_JPEG_QUALITY = int(os.getenv('MOBILE_JPEG_QUALITY', 80))

settings = Settings()
