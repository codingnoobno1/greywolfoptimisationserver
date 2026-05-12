import cv2
import numpy as np
from fastapi import WebSocket
from services.frame_queue import push_frame
from services.result_store import store
from core.logger import logger

async def browser_ingest_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Auto-activate Gesture AI when a mobile stream starts
    store.set_active_mode("mediapipe")
    logger.info("Mobile Binary Pipeline Started. AI Auto-Activated.")
    
    try:
        while True:
            # 1. Receive raw binary bytes (JPEG)
            data = await websocket.receive_bytes()
            
            # 2. Decode JPEG bytes
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                push_frame(frame)
                
                # 3. Send immediate feedback to phone to keep connection active
                # This prevents 'Abnormal Closure' and provides real-time status
                latest_gestures = store.get_result().get("gestures", [])
                
                status_msg = "LISTENING"
                if latest_gestures:
                    # Just send the label of the first detected gesture
                    status_msg = f"DETECTED: {latest_gestures[0]['detected_gesture']}"
                
                await websocket.send_text(status_msg)
            else:
                await websocket.send_text("ERROR: Invalid Frame")
                
    except Exception as e:
        logger.info(f"Mobile Pipeline Closed: {e}")
