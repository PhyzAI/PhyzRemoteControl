# Capture images for creating a model


import cv2
import time
import os

image_path = "./Images/"
base_name = "microphone"

if not os.path.exists(image_path):
    os.mkdir(image_path)

# Open the webcam
cap = cv2.VideoCapture(0)  # Change 0 if using an external webcam

count = 100

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    fname = f"{image_path}{base_name}{count:03}.jpg"
    while os.path.exists(fname):
        count += 1
        fname = f"{image_path}{base_name}{count:03}.jpg"
        
    # save the frame
    cv2.imwrite(fname, frame)

    count += 1

    # Show the frame with detections
    cv2.imshow("Object Detection", frame)

    time.sleep(2.5)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
