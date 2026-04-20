import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.75,
                       min_tracking_confidence=0.75
                       )

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

gesture_label = input("enter gesture name: ").strip()

file_exists = os.path.isfile("gestures.csv")


with open("gestures.csv", "a", newline="") as file:
    writer = csv.writer(file)

    if not file_exists:
        header = [f"x{i}" for i  in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)] + ["label"]
        writer.writerow(header)

    print(f"collecting data for gesture: **{gesture_label}**")
    print("press 'q' to stop collecting. keep doing gesture manually")

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                row = []

                for lm in hand_landmarks.landmark:
                    row.append(lm.x)
                    row.append(lm.y)
                    row.append(lm.z)

                row.append(gesture_label)
                writer.writerow(row)
                frame_count += 1

        cv2.putText(frame, f"Gesture: {gesture_label}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

        cv2.putText(frame, f"Samples: {frame_count}" ,  (10,90),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.imshow("collecting data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print(f"\n collected {frame_count} frames samples for '{gesture_label}'")
print("data saved to 'gestures.csv'")

cap.release()
cv2.destroyAllWindows()