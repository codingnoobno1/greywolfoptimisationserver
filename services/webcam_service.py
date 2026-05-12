import cv2
import threading
import time
from services.frame_queue import push_frame
from core.logger import logger
from config.settings import settings

class WebcamService:
    def __init__(self):
        self.cap = None
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info('Webcam ingestion started.')

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
        logger.info('Webcam ingestion stopped.')

    def _run(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error('Could not open local webcam.')
            self.running = False
            return

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.error('Failed to grab frame from webcam.')
                break
            
            push_frame(frame)
            time.sleep(1.0 / settings.FPS_LIMIT)

webcam_service = WebcamService()
