from ultralytics import YOLO
from config.settings import settings
from core.logger import logger

class YoloModel:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info(f"Loading YOLO model from {settings.YOLO_MODEL_PATH}...")
            try:
                cls._instance = YOLO(settings.YOLO_MODEL_PATH)
                logger.info("YOLO model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
        return cls._instance

# Pre-load on import
YoloModel.get_instance()
