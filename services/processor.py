import time
from ai.yolo_engine.inference import run_yolo
from ai.mediapipe_engine.inference import run_mediapipe
from services.result_store import store
from services.db_service import db
from services.mqtt_client import mqtt_client
from core.logger import logger

def process_frame(frame):
    start_time = time.time()
    active_mode = store.get_active_mode()
    
    result = {
        "frame": frame,
        "annotated_frame": frame,
        "detections": [],
        "gestures": []
    }
    
    try:
        if active_mode == "yolo":
            ai_result = run_yolo(frame)
            result.update(ai_result)
        elif active_mode == "mediapipe":
            ai_result = run_mediapipe(frame)
            result.update(ai_result)
    except Exception as e:
        logger.error(f"Error during inference in mode '{active_mode}': {e}")

    end_time = time.time()
    elapsed = end_time - start_time
    result["fps"] = 1.0 / elapsed if elapsed > 0 else 0.0

    if result.get("gestures"):
        logger.info(f"AI Processed {len(result['gestures'])} gestures")

    # Log results to CSV Database
    if result.get("detections"):
        db.log_event("YOLO_DETECTION", str([d['label'] for d in result['detections']]))
    if result.get("gestures"):
        for g in result['gestures']:
            gesture = g['detected_gesture']
            db.log_event("GESTURE", gesture)
            
            # Publish to Uno R4 via MQTT
            if gesture == "Switch 1":
                mqtt_client.publish("greywolf/switch/1", "ON")
            elif gesture == "Switch 2":
                mqtt_client.publish("greywolf/switch/2", "ON")
            else:
                # If no active gesture, we could optionally send OFF
                # but let's stick to explicit triggers for now
                pass

    store.update_result(result)
