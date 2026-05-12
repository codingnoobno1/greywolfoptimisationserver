from queue import Queue, Full
from config.settings import settings
from core.logger import logger

from core.models import FramePacket

frame_queue = Queue(maxsize=settings.FRAME_QUEUE_MAXSIZE)

def push_frame(packet: FramePacket):
    """Push a FramePacket to the queue. If full, drop the oldest to prevent blocking."""
    try:
        frame_queue.put_nowait(packet)
    except Full:
        logger.debug(f"Queue full! Dropping frame from {packet.source_id}")
        try:
            frame_queue.get_nowait() 
            frame_queue.put_nowait(packet)
        except Exception as e:
            logger.error(f"Error handling full queue: {e}")

def get_queue_size():
    return frame_queue.qsize()
