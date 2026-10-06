import os
import sys
import time
import numpy as np
import cv2
import pygame

# 1. Initialize OpenCV YuNet Face Detector
MODEL_PATH = "face_detection_yunet_2023mar.onnx"
if not os.path.exists(MODEL_PATH):
    sys.exit(f"Missing {MODEL_PATH}. Download it via curl/wget first.")

# Set target resolution for detector
input_size = (1280, 720)
yunet = cv2.FaceDetectorYN.create(
    model=MODEL_PATH,
    config="",
    input_size=input_size,
    score_threshold=0.6,    # Detection confidence
    nms_threshold=0.3,      # Non-maximum suppression
    top_k=5000
)

# 2. Setup Camera and Pygame
pygame.init()
pygame.display.set_caption("PhyzAI Remote Control with Vision (YuNet Long-Range)")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("Error: Could not open video device.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_size[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_size[1])

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Update YuNet input size to match actual camera output
yunet.setInputSize((frame_width, frame_height))

screen = pygame.display.set_mode((frame_width, frame_height))
clock = pygame.time.Clock()

print(f"YuNet Vision loop active ({frame_width}x{frame_height}). Press ESC to exit.")

running = True
try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        success, frame = cap.read()
        if not success:
            break

        # YuNet performs detection directly on OpenCV BGR frames
        _, faces = yunet.detect(frame)

        # Convert frame to RGB for Pygame display drawing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if faces is not None:
            for face in faces:
                # YuNet bounding box output: [x, y, w, h, ...]
                box = list(map(int, face[:4]))
                score = face[-1]

                x, y, w, h = box[0], box[1], box[2], box[3]
                
                # Draw bounding box & confidence label
                cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    rgb_frame, 
                    f"Face: {score:.2f}", 
                    (x, max(y - 10, 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 0), 
                    2
                )

        # Render image array to Pygame
        surface_data = np.rot90(rgb_frame)
        surface_data = np.flipud(surface_data)
        surface = pygame.surfarray.make_surface(surface_data)

        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(30)

finally:
    cap.release()
    pygame.quit()
    sys.exit(0)