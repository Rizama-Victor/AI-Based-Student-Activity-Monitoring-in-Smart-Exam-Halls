import cv2
import imutils
import os
import time
from datetime import datetime

# Create folder for saving detections
save_path = "detections"
os.makedirs(save_path, exist_ok=True)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Capture video from Pi camera or USB webcam
cap = cv2.VideoCapture(0)  # Use 0 for default camera

# Desired frame dimensions
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Fallback center
center_x = frame_width // 2
center_y = frame_height // 2
margin = 50

# Flags for recording
recording = False
record_start_time = None
cheating_start_time = None
out = None

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print(" Warning: Failed to grab frame.")
        continue

    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2

    try:
        frame = imutils.resize(frame, width=frame_width)
    except Exception as e:
        print(f"Resize error: {e}")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    cheating_detected = False

    for (x, y, w, h) in faces:
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        dx = face_center_x - center_x
        dy = face_center_y - center_y

        # Decide face direction
        if abs(dx) < margin and abs(dy) < margin:
            direction = "good sitting posture"
        elif abs(dx) > abs(dy):
            direction = "cheating"
        else:
            if dy > margin:
                direction = "good sitting posture"
            elif dy < -margin:
                direction = "warning"
            else:
                direction = "good sitting posture"

        # Assign colors
        if direction == "cheating":
            box_color = (0, 0, 255)
            cheating_detected = True
        elif direction == "warning":
            box_color = (0, 255, 255)
        else:
            box_color = (0, 255, 0)

        # Draw bounding box & label on live view
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(frame, direction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, box_color, 2)

    # Check cheating timing
    if cheating_detected:
        if cheating_start_time is None:
            cheating_start_time = time.time()
        elif time.time() - cheating_start_time >= 2 and not recording:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(save_path, f"cheating_{timestamp_str}.jpeg")

            # Add timestamp to snapshot
            snapshot_frame = frame.copy()
            cv2.putText(snapshot_frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            cv2.imwrite(screenshot_path, snapshot_frame)
            print(f"📸 Full-frame screenshot saved: {screenshot_path}")

            # Prepare video recording of full frame
            video_path = os.path.join(save_path, f"cheating_{timestamp_str}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))
            record_start_time = time.time()
            recording = True
            print(f"🎥 Recording full frame: {video_path}")
    else:
        cheating_start_time = None

    # Continue recording if active
    if recording:
        frame_with_ts = frame.copy()
        cv2.putText(frame_with_ts, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        out.write(frame_with_ts)

        if time.time() - record_start_time >= 10:  # 10 seconds duration
            recording = False
            out.release()
            print("✅ Recording stopped after 10 seconds.")

    # Draw reference center
   # cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)

    cv2.imshow("Face Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
