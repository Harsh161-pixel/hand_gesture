import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque, Counter
from pythonosc import udp_client

IP = "127.0.0.1"
PORT = 9000

with open("gestures_model.pkl", "rb") as f:
    model = pickle.load(f)

osc_client = udp_client.SimpleUDPClient(IP, PORT)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6,
                       model_complexity=1
                       )
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

gesture_history = deque(maxlen=10)

current_mode = "object"

print("starting Hand Gesture Recogination(trained model)...")
print("Victory wiht both hands = toggle mode(object <-> viewport)")
print("press 'q' to quit...")

def is_victory_both_hands(results):
    if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) < 2:
        return False

    victory_count = 0

    for hand_landmarks in results.multi_hand_landmarks:

        fingers_up = [
            hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y,
            hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,
            hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y,
            hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y,
            hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
            ]

        if fingers_up[1] and fingers_up[2] and not fingers_up[3] and not fingers_up[4]:
            victory_count += 1


    return victory_count >= 2

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    gesture_text = "no hand detected"
    confidence = 0.0
    mode_text = "Object Mode" if current_mode == "object" else "Viewport Mode"

    if is_victory_both_hands(results):
        current_mode = "viewport" if current_mode == "object" else "object"
        osc_client.send_message("/mode", current_mode)
        print(f"mode switch to: {current_mode.upper()}")

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            row = np.array(row).reshape(1,-1)

            probs = model.predict_proba(row)[0]
            confidence = max(probs)
            prediction = model.classes_[np.argmax(probs)]

            if confidence > 0.70:
                gesture_history.append(prediction)

            if gesture_history:
                gesture_text = Counter(gesture_history).most_common(1)[0][0]

            osc_client.send_message("/gesture", gesture_text)
            osc_client.send_message("/mode", current_mode)

    cv2.putText(frame, f"Gesture: {gesture_text} ({confidence:.2f})", (10,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"MODE: {mode_text}", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,165,0), 2)

    cv2.imshow("hand gesture control - blender projects", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

