import cv2
import time
from ai.yolo_engine.inference import run_yolo
from ai.mediapipe_engine.inference import run_mediapipe
from services.result_store import store
from services.db_service import db
from services.mqtt_client import mqtt_client
from core.logger import logger
from core.models import FramePacket

def process_frame(packet: FramePacket):
    """
    Standardized AI Processing Pipeline.
    1. Normalize resolution to 640x480
    2. Run inference (YOLO + MediaPipe)
    3. Render & Store results
    """
    start_time = time.time()
    active_mode = store.get_active_mode(packet.source_id)
    
    # 1. Normalize Frame Resolution (Standardization Requirement)
    frame = packet.frame
    h, w = frame.shape[:2]
    if w != 640 or h != 480:
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

    result = {
        "frame": frame,
        "annotated_frame": frame,
        "detections": [],
        "gestures": [],
        "source": packet.source_id
    }
    
    try:
        if active_mode in ["yolo", "all"]:
            yolo_res = run_yolo(result["annotated_frame"])
            result["annotated_frame"] = yolo_res["annotated_frame"]
            result["detections"] = yolo_res["detections"]
            
        if active_mode in ["mediapipe", "all"]:
            mp_res = run_mediapipe(result["annotated_frame"])
            result["annotated_frame"] = mp_res["annotated_frame"]
            result["gestures"] = mp_res["gestures"]
            
    except Exception as e:
        logger.error(f"Error during inference from {packet.source_id}: {e}")

    end_time = time.time()
    elapsed = end_time - start_time
    result["fps"] = 1.0 / elapsed if elapsed > 0 else 0.0

    # Log results to CSV Database
    if result.get("detections"):
        db.log_event("YOLO_DETECTION", f"[{packet.source_id}] {str([d['label'] for d in result['detections']])}")
    if result.get("gestures"):
        for g in result['gestures']:
            db.log_event("GESTURE", f"[{packet.source_id}] {g['detected_gesture']}")
            
            # Publish to Uno R4 via MQTT
            gesture = g['detected_gesture']
            # Standardized Command Format: S1_ON, S1_OFF, S2_ON, S2_OFF, REG_1..5
            if gesture == "Switch 1":
                mqtt_client.publish("greywolf/status/event", f"G_DETECTED_S1_{packet.source_id}")
            elif gesture == "Switch 2":
                mqtt_client.publish("greywolf/status/event", f"G_DETECTED_S2_{packet.source_id}")

    store.update_result(packet.source_id, result)
