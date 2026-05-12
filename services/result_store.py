from threading import Lock

class ResultStore:
    def __init__(self):
        self._lock = Lock()
        self._latest_result = {
            'frame': None,
            'annotated_frame': None,
            'detections': [],
            'gestures': [],
            'fps': 0.0
        }
        self._active_mode = 'none' # none, yolo, mediapipe, all
        self._input_source = 'unknown' # webcam, mqtt, mobile, unknown

    def update_result(self, result: dict):
        with self._lock:
            self._latest_result.update(result)

    def get_result(self) -> dict:
        with self._lock:
            return {
                'detections': self._latest_result.get('detections', []),
                'gestures': self._latest_result.get('gestures', []),
                'fps': self._latest_result.get('fps', 0.0)
            }
            
    def get_latest_frame(self):
        with self._lock:
            # FIX: Explicit None check to avoid NumPy truth value ambiguity crash
            frame = self._latest_result.get('annotated_frame')
            if frame is None:
                frame = self._latest_result.get('frame')
            return frame

    def set_active_mode(self, mode: str):
        with self._lock:
            self._active_mode = mode

    def get_active_mode(self) -> str:
        with self._lock:
            return self._active_mode

    def set_input_source(self, source: str):
        with self._lock:
            self._input_source = source

    def get_input_source(self) -> str:
        with self._lock:
            return self._input_source

store = ResultStore()
