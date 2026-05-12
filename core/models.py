import time
import numpy as np
from pydantic import BaseModel
from typing import Optional, Dict, Any

class FramePacket:
    """Standardized internal frame object for the Greywolf AI platform."""
    def __init__(
        self, 
        frame: np.ndarray, 
        source_id: str, 
        source_type: str, 
        fps: Optional[int] = None, 
        quality: Optional[int] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.frame = frame
        self.source_id = source_id
        self.source_type = source_type
        self.timestamp = time.time()
        self.fps = fps
        self.quality = quality
        self.metadata = metadata or {}

    def to_dict(self):
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "timestamp": self.timestamp,
            "fps": self.fps,
            "quality": self.quality,
            "metadata": self.metadata
        }
