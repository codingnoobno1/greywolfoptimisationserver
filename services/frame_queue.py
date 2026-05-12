from queue import Queue, Full
from config.settings import settings
from core.logger import logger

frame_queue = Queue(maxsize=settings.FRAME_QUEUE_MAXSIZE)

def push_frame(frame):
    """Push a frame to the queue. If full, drop the oldest frame to prevent blocking."""
    try:
        frame_queue.put_nowait(frame)
    except Full:
        logger.warning("Frame queue is full! Dropping oldest frame.")
        try:
            frame_queue.get_nowait() # Remove oldest
            frame_queue.put_nowait(frame) # Try putting again
        except Exception as e:
            logger.error(f"Error handling full queue: {e}")

def get_queue_size():
    return frame_queue.qsize()
