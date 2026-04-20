import cv2
import mediapipe as mp
from collections import deque, Counter

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6,
                       model_complexity=1
                       )
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

gesture_history = deque(maxlen=15)

def get_gesture(landmarks):
    if not landmarks:
        return "no hand"

    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]
    middle_tip = landmarks.landmark[12]
    ring_tip = landmarks.landmark[16]
    pinky_tip = landmarks.landmark[20]

    fingers_up = [
        thumb_tip.x < landmarks.landmark[3].x,
        index_tip.y < landmarks.landmark[6].y,
        middle_tip.y < landmarks.landmark[10].y,
        ring_tip.y < landmarks.landmark[14].y,
        pinky_tip.y < landmarks.landmark[18].y
    ]

    pinch_dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5

    if pinch_dist < 0.05:
        return "pinch"
    elif sum(fingers_up) == 0:
        return "fist"
    elif sum(fingers_up) == 5:
        return "Open palm"
    elif fingers_up[1] and not any(fingers_up[2:]):
        return "pointing"
    elif fingers_up[0] and not any(fingers_up[1:]):
        return "thumbs up"
    elif fingers_up[1] and fingers_up[2] and not any(fingers_up[3:]):
        return "Victory"
    else:
        return "unknown"




while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    gesture_text = "no hand dectected"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            current_gesture = get_gesture(hand_landmarks)

            gesture_history.append(current_gesture)

            if gesture_history:
                gesture_text = Counter(gesture_history).most_common(1)[0][0]


    cv2.putText(frame, f"Gesture: {gesture_text}", (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("hand gesture control - blender projects", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

