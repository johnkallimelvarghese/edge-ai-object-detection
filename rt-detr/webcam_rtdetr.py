import cv2
import time
from ultralytics import RTDETR

# Load model (pretrained)
model = RTDETR("rtdetr-l.pt")  # you can try rtdetr-n.pt for faster

cap = cv2.VideoCapture(0)

total_time = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    results = model(frame, imgsz=640)[0]

    end = time.time()
    inference_time = end - start

    fps = 1 / inference_time
    total_time += inference_time
    frame_count += 1
    avg_fps = frame_count / total_time

    print(f"FPS: {fps:.2f} | Avg FPS: {avg_fps:.2f} | Latency: {inference_time*1000:.2f} ms")

    # Draw results
    annotated_frame = results.plot()

    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("RT-DETR Webcam", annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()