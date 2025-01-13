
# Basic YOLO object detection


import torch
import cv2
from ultralytics import YOLO  # Use YOLO from the Ultralytics library

# Load a pre-trained YOLO model
# You can specify "yolov5s.pt" or "yolov8s.pt" (small model versions) or other model sizes for different performance
#model = YOLO("yolov8n.pt")  # 'n' for nano model, fast and lightweight for real-time detection
model = YOLO("yolo11s.pt")  # 'n' for nano model, fast and lightweight for real-time detection
#model = YOLO("/Volumes/Safari/PhyzAI_RemoteControl/runs/detect/train2/weights/best.pt")


# Open the webcam
cap = cv2.VideoCapture(0)  # Change 0 if using an external webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model on the frame
    results = model(frame)  # Detect objects in the frame

    # Parse and draw bounding boxes
    for result in results:
        boxes = result.boxes  # Each detection result has 'boxes', 'conf', and 'class' attributes
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()  # Get coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = box.cls[0].item()  # Class ID

            # Draw the bounding box with label and confidence score
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show the frame with detections
    cv2.imshow("Object Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
