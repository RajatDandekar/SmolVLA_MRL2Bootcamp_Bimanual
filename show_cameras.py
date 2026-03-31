"""Live triple camera feed viewer. Press Q to quit."""
import cv2
import numpy as np

caps = [cv2.VideoCapture(i) for i in range(3)]
labels = ["Arm Cam Left", "Arm Cam Right", "Webcam (Kreo Owl)"]

for cap in caps:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Showing 3 live feeds. Press Q to quit.")

while True:
    frames = []
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Camera {i} unavailable", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"[{i}] {labels[i]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        frames.append(frame)

    combined = np.hstack(frames)
    cv2.imshow("SO-101 Bimanual Cameras — Press Q to quit", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
