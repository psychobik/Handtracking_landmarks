import cv2
import mediapipe as mp
import time
import math
import csv
import os
from datetime import datetime
import winsound

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
recording = False
start_time = None
recorded_data = []
hand_closure_count = 0
prev_hand_state = "open"
record_duration = 60  # Max recording duration in seconds
current_date = datetime.now().strftime("%Y-%m-%d")
last_logged_date = None

# CSV setup
csv_file = "hand_tracking_data.csv"
header = [
    'date', 'time', 'hand', 'landmark', 'x', 'y', 'z', 'hand_closing_count',
    'index_dist', 'middle_dist', 'ring_dist', 'pinky_dist'
]

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

# Function to calculate Euclidean distance
def calculate_distance(point1, point2):
    return round(math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2), 4)

# Video capture setup
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and process the frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Process hand landmarks if detected
    frame_height, frame_width, _ = frame.shape
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[hand_idx].classification[0].label
            landmarks = hand_landmarks.landmark

            # Calculate distances for hand closure detection
            distances = {
                "index": calculate_distance(landmarks[8], landmarks[5]),
                "middle": calculate_distance(landmarks[12], landmarks[9]),
                "ring": calculate_distance(landmarks[16], landmarks[13]),
                "pinky": calculate_distance(landmarks[20], landmarks[17])
            }
            fully_open = all(dist > 0.1 for dist in distances.values())
            fully_closed = all(dist < 0.05 for dist in distances.values())

            # Detect hand closure
            if prev_hand_state == "open" and fully_closed:
                hand_closure_count += 1
                prev_hand_state = "closed"
                winsound.Beep(1000, 200)  # Beep sound for feedback
            elif fully_open:
                prev_hand_state = "open"

            # Draw bounding box and landmarks
            x_min = int(min([landmark.x for landmark in landmarks]) * frame_width)
            y_min = int(min([landmark.y for landmark in landmarks]) * frame_height)
            x_max = int(max([landmark.x for landmark in landmarks]) * frame_width)
            y_max = int(max([landmark.y for landmark in landmarks]) * frame_height)
            box_color = (0, 0, 255) if prev_hand_state == "closed" else (0, 255, 0)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
            cv2.putText(frame, f"Hand: {hand_label}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            # Record data if recording
            if recording:
                timestamp = datetime.now().strftime("%H:%M:%S")
                date_to_log = current_date if last_logged_date != current_date else ""
                if date_to_log:
                    last_logged_date = current_date

                for i, landmark in enumerate(landmarks):
                    recorded_data.append([
                        date_to_log, timestamp, hand_label, f"Landmark_{i}",
                        round(landmark.x, 4), round(landmark.y, 4), round(landmark.z, 4),
                        hand_closure_count,
                        distances["index"], distances["middle"],
                        distances["ring"], distances["pinky"]
                    ])

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display timer and hand closure count
    if recording:
        elapsed_time = time.time() - start_time
        cv2.putText(frame, f"Recording: {int(elapsed_time)}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if elapsed_time >= record_duration:
            recording = False
            out.release()
            print(f"Recording saved as {video_filename}")

    # Write video frames
    if recording and out is not None:
        out.write(frame)

    # Display hand closure count
    cv2.putText(frame, f"Closures: {hand_closure_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Hand Tracking", frame)

    # Start/stop recording on pressing 'R'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        if not recording:
            recording = True
            start_time = time.time()
            hand_closure_count = 0
            recorded_data = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"recording_{timestamp}.avi"
            out = cv2.VideoWriter(video_filename, fourcc, 60.0, (frame.shape[1], frame.shape[0]))
        else:
            recording = False
            if out:
                out.release()
                print(f"Recording saved as {video_filename}")
                # Write CSV data
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(recorded_data)
                recorded_data = []

    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
if out:
    out.release()
