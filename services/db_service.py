import csv
import os
from datetime import datetime
from threading import Lock
from core.logger import logger

class CSVDatabase:
    def __init__(self, filename='data_log.csv'):
        self.filename = filename
        self.lock = Lock()
        self._initialize_csv()

    def _initialize_csv(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'event_type', 'details'])
            logger.info(f'Initialized CSV database: {self.filename}')

    def log_event(self, event_type, details):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        with self.lock:
            try:
                with open(self.filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, event_type, details])
            except Exception as e:
                logger.error(f'Failed to log to CSV: {e}')

db = CSVDatabase()
