import cv2
import mediapipe as mp
from ai.mediapipe_engine.model import MediaPipeModel

# Hand connection pairs for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17), (0, 17) # Palm
]

def run_mediapipe(frame):
    landmarker = MediaPipeModel.get_instance()
    
    if not landmarker:
        return {"annotated_frame": frame, "gestures": []}

    # MP Tasks requires mp.Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Run inference
    detection_result = landmarker.detect(mp_image)
    
    annotated_frame = frame.copy()
    hands_data = []
    
    if detection_result.hand_landmarks:
        h, w, _ = frame.shape
        for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            # Draw landmarks and connections manually
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start_lm = hand_landmarks[start_idx]
                end_lm = hand_landmarks[end_idx]
                pt1 = (int(start_lm.x * w), int(start_lm.y * h))
                pt2 = (int(end_lm.x * w), int(end_lm.y * h))
                cv2.line(annotated_frame, pt1, pt2, (0, 255, 0), 2)

            for lm in hand_landmarks:
                pt = (int(lm.x * w), int(lm.y * h))
                cv2.circle(annotated_frame, pt, 5, (255, 0, 0), -1)
            
            # Extract landmarks for gesture logic
            landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks]
            
            # Handedness
            handedness = "Unknown"
            if detection_result.handedness:
                handedness = detection_result.handedness[hand_idx][0].category_name
                
            # --- Gesture Recognition Logic ---
            def is_finger_open(lm_list, tip_idx, pip_idx):
                return lm_list[tip_idx].y < lm_list[pip_idx].y

            index_open = is_finger_open(hand_landmarks, 8, 6)
            middle_open = is_finger_open(hand_landmarks, 12, 10)
            ring_open = is_finger_open(hand_landmarks, 16, 14)
            pinky_open = is_finger_open(hand_landmarks, 20, 18)

            detected_gesture = "None"
            if index_open and not middle_open and not ring_open and not pinky_open:
                detected_gesture = "Switch 1"
            elif index_open and middle_open and not ring_open and not pinky_open:
                detected_gesture = "Switch 2"

            # Draw the gesture text on the frame
            cv2.putText(
                annotated_frame, 
                f"{handedness}: {detected_gesture}", 
                (10, 60 + (hand_idx * 40)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_AA
            )
                
            hands_data.append({
                "handedness": handedness,
                "detected_gesture": detected_gesture,
                "landmarks": landmarks
            })

    return {"annotated_frame": annotated_frame, "gestures": hands_data}

