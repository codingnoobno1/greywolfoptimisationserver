import cv2
import time
import numpy as np
from services.result_store import store
from config.settings import settings

def mjpeg_generator():
    """Generator function for streaming video frames using MJPEG"""
    target_frame_time = 1.0 / settings.FPS_LIMIT
    
    # Placeholder for 'No Signal'
    no_signal = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(no_signal, 'CAMERA OFFLINE / MQTT DISCONNECTED', (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    while True:
        start_time = time.time()
        
        frame = store.get_latest_frame()
        if frame is None:
            # Show no signal frame if no data
            frame = no_signal
            
        # Encode the frame in JPEG format
        flag, encoded_image = cv2.imencode(
            '.jpg', 
            frame, 
            [int(cv2.IMWRITE_JPEG_QUALITY), settings.JPEG_QUALITY]
        )
        
        if not flag:
            continue
            
        # Yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encoded_image) + b'\r\n')
            
        # Sleep to maintain target FPS
        elapsed = time.time() - start_time
        sleep_time = target_frame_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
