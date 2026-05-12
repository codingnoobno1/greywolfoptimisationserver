import mediapipe as mp
# On this environment, solutions are nested under .python
try:
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_drawing
    import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
except ImportError:
    # Fallback for different versions
    try:
        import mediapipe.solutions.hands as mp_hands
        import mediapipe.solutions.drawing_utils as mp_drawing
        import mediapipe.solutions.drawing_styles as mp_drawing_styles
    except ImportError:
        mp_hands = None
        mp_drawing = None
        mp_drawing_styles = None

from core.logger import logger

class MediaPipeModel:
    _instance = None
    _mp_drawing = mp_drawing
    _mp_drawing_styles = mp_drawing_styles
    _mp_hands = mp_hands

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            if cls._mp_hands is None:
                logger.error('MediaPipe solutions modules not found in any known path.')
                return None, None, None, None
                
            logger.info('Loading MediaPipe Hands model...')
            try:
                cls._instance = cls._mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                logger.info('MediaPipe model loaded successfully.')
            except Exception as e:
                logger.error(f'Failed to load MediaPipe model: {e}')
        return cls._instance, cls._mp_hands, cls._mp_drawing, cls._mp_drawing_styles

# Pre-load on import
MediaPipeModel.get_instance()
