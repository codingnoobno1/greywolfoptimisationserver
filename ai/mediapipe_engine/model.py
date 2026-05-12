import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from core.logger import logger

class MediaPipeModel:
    _instance = None
    _model_path = os.path.join(os.path.dirname(__file__), 'models', 'hand_landmarker.task')

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            if not os.path.exists(cls._model_path):
                logger.error(f'MediaPipe model file not found at {cls._model_path}')
                return None
                
            logger.info('Loading MediaPipe Hand Landmarker Task...')
            try:
                base_options = python.BaseOptions(model_asset_path=cls._model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                cls._instance = vision.HandLandmarker.create_from_options(options)
                logger.info('MediaPipe Hand Landmarker loaded successfully.')
            except Exception as e:
                logger.error(f'Failed to load MediaPipe model: {e}')
                cls._instance = None
        return cls._instance

# Pre-load on import
MediaPipeModel.get_instance()

