import cv2
import math
import mediapipe as mp
from ai.mediapipe_engine.model import MediaPipeModel

# Hand connection pairs for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17), (0, 17)     # Palm
]

# ── Landmark indices (MediaPipe hand model) ───────────────────────────────────
#  0 = Wrist
#  1,2,3,4  = Thumb  (CMC, MCP, IP, TIP)
#  5,6,7,8  = Index  (MCP, PIP, DIP, TIP)
#  9,10,11,12 = Middle
#  13,14,15,16 = Ring
#  17,18,19,20 = Pinky


def _distance(a, b) -> float:
    """Euclidean distance between two landmarks (normalized coords)."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _detect_gesture(lm, handedness: str) -> str:
    """
    Detect gesture from a single hand's landmark list.

    Gesture catalogue:
      Switch 1  — index only raised
      Switch 2  — index + middle raised
      Switch 3  — index + middle + ring raised
      Switch 4  — index + middle + ring + pinky raised (4 fingers)
      Switch 5  — all 5 fingers open (thumb included)
      Regulator — thumb + index + middle raised, ring + pinky closed
                  (like pinching/holding a rotary dial)

    Detection is ordered from most-specific to least-specific to avoid
    a 5-finger open triggering Switch 4 first.
    """
    MARGIN       = 0.035   # y-margin for finger open/closed (~3.5% of frame)
    THUMB_MARGIN = 0.030   # slightly smaller for thumb (shorter travel)
    PINCH_THRESH = 0.12    # max normalised distance for thumb-index pinch

    # ── Finger state helpers ─────────────────────────────────────────────────
    def f_open(tip, pip):
        return lm[tip].y < lm[pip].y - MARGIN

    def f_closed(tip, pip):
        return lm[tip].y > lm[pip].y + MARGIN

    # Fingers (y-axis — works for vertical finger extension)
    idx_open   = f_open(8, 6)
    mid_open   = f_open(12, 10)
    ring_open  = f_open(16, 14)
    pinky_open = f_open(20, 18)

    ring_closed  = f_closed(16, 14)
    pinky_closed = f_closed(20, 18)
    mid_closed   = f_closed(12, 10)

    # Thumb — moves sideways so use x-axis relative to handedness
    # Right hand: thumb tip.x < thumb MCP.x when extended outward
    # Left  hand: thumb tip.x > thumb MCP.x when extended outward
    # Additionally require tip above IP joint in y (thumb raised up)
    if handedness == "Right":
        thumb_out = lm[4].x < lm[2].x - THUMB_MARGIN
    else:
        thumb_out = lm[4].x > lm[2].x + THUMB_MARGIN
    thumb_up  = lm[4].y < lm[3].y - THUMB_MARGIN   # tip above IP joint

    thumb_open = thumb_out or thumb_up   # either direction counts

    # Regulator pinch: thumb tip close to index tip (holding a knob)
    pinch_dist    = _distance(lm[4], lm[8])
    pinch_formed  = pinch_dist < PINCH_THRESH

    # ── Gesture priority (most specific → least specific) ────────────────────

    # Switch 5 — all 5 fingers clearly open (thumb + 4 fingers)
    if thumb_open and idx_open and mid_open and ring_open and pinky_open:
        return "Switch 5"

    # Regulator — thumb + index + middle in a dial/pinch shape
    # ring and pinky must be clearly closed (hand is NOT fully open)
    if (thumb_open or pinch_formed) and idx_open and mid_open \
            and ring_closed and pinky_closed:
        return "Regulator"

    # Switch 4 — four fingers up (no thumb required)
    if idx_open and mid_open and ring_open and pinky_open:
        return "Switch 4"

    # Switch 3 — index + middle + ring up, pinky closed
    if idx_open and mid_open and ring_open and pinky_closed:
        return "Switch 3"

    # Switch 2 — index + middle, ring + pinky closed
    if idx_open and mid_open and ring_closed and pinky_closed:
        return "Switch 2"

    # Switch 1 — index only, rest clearly closed
    if idx_open and mid_closed and ring_closed and pinky_closed:
        return "Switch 1"

    return "None"


def run_mediapipe(frame):
    landmarker = MediaPipeModel.get_instance()
    if not landmarker:
        return {"annotated_frame": frame, "gestures": []}

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = landmarker.detect(mp_image)

    annotated_frame = frame.copy()
    hands_data      = []

    # Gesture → overlay colour map
    GESTURE_COLORS = {
        "Switch 1":  (0,   255, 255),  # Cyan
        "Switch 2":  (0,   200, 255),  # Light blue
        "Switch 3":  (0,   128, 255),  # Orange-blue
        "Switch 4":  (0,    80, 255),  # Orange
        "Switch 5":  (0,   255,   0),  # Green
        "Regulator": (255, 140,   0),  # Amber
        "None":      (128, 128, 128),  # Grey
    }

    if detection_result.hand_landmarks:
        h, w, _ = frame.shape
        for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):

            # Draw skeleton
            for (s, e) in HAND_CONNECTIONS:
                pt1 = (int(hand_landmarks[s].x * w), int(hand_landmarks[s].y * h))
                pt2 = (int(hand_landmarks[e].x * w), int(hand_landmarks[e].y * h))
                cv2.line(annotated_frame, pt1, pt2, (0, 200, 100), 2)

            for lm in hand_landmarks:
                pt = (int(lm.x * w), int(lm.y * h))
                cv2.circle(annotated_frame, pt, 4, (255, 60, 60), -1)

            # Handedness
            handedness = "Unknown"
            if detection_result.handedness:
                handedness = detection_result.handedness[hand_idx][0].category_name

            # Run gesture classifier
            detected_gesture = _detect_gesture(hand_landmarks, handedness)

            color = GESTURE_COLORS.get(detected_gesture, (180, 180, 180))

            # Draw label with background box for readability
            label = f"{handedness}: {detected_gesture}"
            y_pos = 50 + hand_idx * 45
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(annotated_frame,
                          (8, y_pos - th - 6), (14 + tw, y_pos + 6),
                          (0, 0, 0), -1)
            cv2.putText(annotated_frame, label,
                        (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            # Regulator: draw a circular arc hint to indicate dial mode
            if detected_gesture == "Regulator":
                cx = int(hand_landmarks[9].x * w)   # palm centre
                cy = int(hand_landmarks[9].y * h)
                cv2.circle(annotated_frame, (cx, cy), 35, (255, 140, 0), 2)
                cv2.putText(annotated_frame, "DIAL",
                            (cx - 18, cy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

            # Export (landmarks as plain dicts — JSON-serialisable)
            landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z}
                         for lm in hand_landmarks]

            hands_data.append({
                "handedness":       handedness,
                "detected_gesture": detected_gesture,
                "landmarks":        landmarks,
            })

    return {"annotated_frame": annotated_frame, "gestures": hands_data}
