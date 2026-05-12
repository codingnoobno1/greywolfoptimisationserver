import threading
from services.frame_queue import frame_queue
from services.processor import process_frame
from core.logger import logger
from config.settings import settings

def worker_loop():
    logger.info(f"Worker thread {threading.current_thread().name} started.")
    while True:
        try:
            frame = frame_queue.get()
            process_frame(frame)
        except Exception as e:
            logger.error(f"Error in worker thread: {e}")

def start_workers(num_workers=settings.NUM_WORKERS):
    logger.info(f"Starting {num_workers} AI worker threads...")
    for i in range(num_workers):
        t = threading.Thread(target=worker_loop, daemon=True, name=f"AI-Worker-{i+1}")
        t.start()
