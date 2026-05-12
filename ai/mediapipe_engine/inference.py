import cv2
from ai.mediapipe_engine.model import MediaPipeModel

def run_mediapipe(frame):
    hands_model, mp_hands, mp_drawing, mp_drawing_styles = MediaPipeModel.get_instance()
    
    if not hands_model:
        return {"annotated_frame": frame, "gestures": []}

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_model.process(rgb_frame)
    
    annotated_frame = frame.copy()
    hands_data = []
    
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark]
            
            handedness = "Unknown"
            if results.multi_handedness:
                handedness = results.multi_handedness[hand_idx].classification[0].label
                
            # --- Gesture Recognition Logic ---
            def is_finger_open(lm_list, tip_idx, pip_idx):
                return lm_list[tip_idx].y < lm_list[pip_idx].y

            lms = hand_landmarks.landmark
            index_open = is_finger_open(lms, 8, 6)
            middle_open = is_finger_open(lms, 12, 10)
            ring_open = is_finger_open(lms, 16, 14)
            pinky_open = is_finger_open(lms, 20, 18)

            detected_gesture = "None"
            if index_open and not middle_open and not ring_open and not pinky_open:
                detected_gesture = "Switch 1"
            elif index_open and middle_open and not ring_open and not pinky_open:
                detected_gesture = "Switch 2"

            # Draw the gesture text on the frame
            cv2.putText(
                annotated_frame, 
                f"{handedness}: {detected_gesture}", 
                (10, 30 + (hand_idx * 40)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2, 
                cv2.LINE_AA
            )
                
            hands_data.append({
                "handedness": handedness,
                "detected_gesture": detected_gesture,
                "landmarks": landmarks
            })

    return {"annotated_frame": annotated_frame, "gestures": hands_data}
