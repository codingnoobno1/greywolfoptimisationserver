from threading import Lock

class ResultStore:
    def __init__(self):
        self._lock = Lock()
        self._sources = {} # source_id -> result_dict
        self._active_modes = {} # source_id -> mode
        
    def _ensure_source(self, source_id: str):
        if source_id not in self._sources:
            self._sources[source_id] = {
                'frame': None,
                'annotated_frame': None,
                'detections': [],
                'gestures': [],
                'telemetry': {},
                'fps': 0.0
            }
            if source_id not in self._active_modes:
                self._active_modes[source_id] = 'none'

    def update_result(self, source_id: str, result: dict):
        with self._lock:
            self._ensure_source(source_id)
            self._sources[source_id].update(result)

    def get_result(self, source_id: str) -> dict:
        with self._lock:
            if source_id not in self._sources:
                return {'detections': [], 'gestures': [], 'fps': 0.0, 'telemetry': {}}
            res = self._sources[source_id]
            return {
                'detections': res.get('detections', []),
                'gestures': res.get('gestures', []),
                'telemetry': res.get('telemetry', {}),
                'fps': res.get('fps', 0.0)
            }
            
    def get_latest_frame(self, source_id: str):
        with self._lock:
            if source_id not in self._sources:
                return None
            res = self._sources[source_id]
            # FIX: Explicit None check to avoid NumPy truth value ambiguity crash
            frame = res.get('annotated_frame')
            if frame is None:
                frame = res.get('frame')
            return frame

    def set_active_mode(self, source_id: str, mode: str):
        with self._lock:
            self._active_modes[source_id] = mode

    def get_active_mode(self, source_id: str) -> str:
        with self._lock:
            return self._active_modes.get(source_id, 'none')

    def list_sources(self) -> list:
        with self._lock:
            return list(self._sources.keys())

store = ResultStore()
