import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque, Counter

with open("gestures_model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6,
                       model_complexity=1
                       )
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

gesture_history = deque(maxlen=15)

print("starting Hand Gesture Recogination(trained model)...")
print("press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    gesture_text = "no hand dectected"
    confidence = 0.0

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

    cv2.putText(frame, f"Gesture: {gesture_text} ({confidence:.2f})", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("hand gesture control - blender projects", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

