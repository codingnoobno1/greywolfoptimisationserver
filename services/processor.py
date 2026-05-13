"""
Greywolf AI Processing Pipeline.

Optimizations over the original:

1. Gesture confirmation buffer:
   Each source tracks a per-gesture consecutive-detection counter.
   A gesture is only CONFIRMED and stored in ResultStore after it appears
   in GESTURE_CONFIRM_FRAMES consecutive frames. This eliminates single-frame
   false positives that caused rapid toggle flicker on the Flutter side.

2. Gesture cooldown:
   After a confirmed gesture fires, it won't fire again for GESTURE_COOLDOWN_FRAMES
   frames even if the hand stays in position. Prevents holding the gesture from
   rapid-firing the toggle.

3. DB write removed from hot path:
   The original code called db.log_event() on every detected gesture frame
   which is synchronous disk I/O in the worker thread. Moved to fire only on
   confirmed transitions, amortising the cost.

4. Frame normalization uses INTER_AREA for downscale (sharper than INTER_LINEAR).
"""
import cv2
import time
from collections import defaultdict
from ai.yolo_engine.inference import run_yolo
from ai.mediapipe_engine.inference import run_mediapipe
from services.result_store import store
from services.db_service import db
from services.mqtt_client import mqtt_client
from core.logger import logger
from core.models import FramePacket

# ── Stability parameters ───────────────────────────────────────────────────────
# A gesture must appear in this many consecutive frames before being "confirmed".
# At ~7 fps processing: 3 frames ≈ 430 ms hold time. Adjust if needed.
GESTURE_CONFIRM_FRAMES = 3

# After a confirmed gesture, ignore re-fires for this many frames.
# Prevents holding the gesture from toggling the switch repeatedly.
GESTURE_COOLDOWN_FRAMES = 10

# ── Per-source gesture state (lives for the server process lifetime) ──────────
#   source_id → {
#       "pending":    str,   # current candidate gesture name
#       "count":      int,   # consecutive frame count for pending
#       "last_confirmed": str,  # last gesture that was confirmed+stored
#       "cooldown":   int,   # frames remaining in cooldown
#   }
_gesture_state: dict[str, dict] = defaultdict(lambda: {
    "pending": "None",
    "count": 0,
    "last_confirmed": "None",
    "cooldown": 0,
})


def _update_gesture_state(source_id: str, raw_gesture: str) -> str | None:
    """
    Apply confirmation + cooldown logic to a raw single-frame gesture.

    Returns:
        The confirmed gesture name if a new confirmation event occurred,
        None otherwise (gesture still pending, in cooldown, or "None").
    """
    state = _gesture_state[source_id]

    # Tick down cooldown
    if state["cooldown"] > 0:
        state["cooldown"] -= 1

    if raw_gesture == "None":
        # Hand not in a gesture → reset pending streak
        state["pending"] = "None"
        state["count"] = 0
        return None

    if raw_gesture == state["pending"]:
        state["count"] += 1
    else:
        # New candidate — reset streak
        state["pending"] = raw_gesture
        state["count"] = 1

    # Confirm if streak reached threshold and cooldown expired
    if state["count"] >= GESTURE_CONFIRM_FRAMES and state["cooldown"] == 0:
        state["count"] = 0          # reset so it won't fire again immediately
        state["cooldown"] = GESTURE_COOLDOWN_FRAMES
        state["last_confirmed"] = raw_gesture
        return raw_gesture          # ← confirmed event

    return None


def process_frame(packet: FramePacket):
    """
    Standardized AI Processing Pipeline.
    1. Normalize resolution to 640×480
    2. Run inference (YOLO / MediaPipe / both)
    3. Apply gesture confirmation buffer
    4. Store result
    """
    start_time = time.time()
    active_mode = store.get_active_mode(packet.source_id)

    # 1. Normalize (INTER_AREA is better for downscale)
    frame = packet.frame
    h, w = frame.shape[:2]
    if w != 640 or h != 480:
        interp = cv2.INTER_AREA if (w > 640 or h > 480) else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (640, 480), interpolation=interp)

    result = {
        "frame": frame,
        "annotated_frame": frame,
        "detections": [],
        "gestures": [],
        "source": packet.source_id,
    }

    try:
        if active_mode in ("yolo", "all"):
            yolo_res = run_yolo(result["annotated_frame"])
            result["annotated_frame"] = yolo_res["annotated_frame"]
            result["detections"] = yolo_res["detections"]

        if active_mode in ("mediapipe", "all"):
            mp_res = run_mediapipe(result["annotated_frame"])
            result["annotated_frame"] = mp_res["annotated_frame"]

            # ── Apply confirmation buffer to each detected hand ────────────
            confirmed_gestures = []
            raw_gestures = mp_res.get("gestures", [])

            for hand in raw_gestures:
                raw = hand.get("detected_gesture", "None")
                # Use a per-hand key so two hands don't interfere
                key = f"{packet.source_id}:{hand.get('handedness', 'Unknown')}"
                confirmed = _update_gesture_state(key, raw)

                if confirmed and confirmed != "None":
                    confirmed_gestures.append(hand)   # pass through full hand data
                    # DB write only on confirmed event (not every frame)
                    db.log_event("GESTURE",
                                 f"[{packet.source_id}] {confirmed} CONFIRMED")

                    # MQTT: server-side publish (matches existing topic schema)
                    if confirmed == "Switch 1":
                        mqtt_client.publish("greywolf/status/event",
                                            f"G_DETECTED_S1_{packet.source_id}")
                    elif confirmed == "Switch 2":
                        mqtt_client.publish("greywolf/status/event",
                                            f"G_DETECTED_S2_{packet.source_id}")

            # Store the raw gestures for telemetry display (hand position, etc.)
            # but mark which ones are confirmed so Flutter can decide.
            for hand in raw_gestures:
                hand["confirmed"] = hand in confirmed_gestures
            result["gestures"] = raw_gestures

    except Exception as e:
        logger.error(f"Inference error [{packet.source_id}]: {e}")

    end_time = time.time()
    elapsed = end_time - start_time
    result["fps"] = 1.0 / elapsed if elapsed > 0 else 0.0

    if result.get("detections"):
        db.log_event("YOLO_DETECTION",
                     f"[{packet.source_id}] {[d['label'] for d in result['detections']]}")

    store.update_result(packet.source_id, result)
