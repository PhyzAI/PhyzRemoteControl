import os
import sys
import numpy as np
import cv2
import pygame
import insightface
from insightface.app import FaceAnalysis

# Prevent Cocoa/Objective-C fork warnings on macOS
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# -----------------------------------------------------------------------------
# 1. Initialize InsightFace App
# -----------------------------------------------------------------------------
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def cosine_similarity(emb1, emb2):
    """Calculates cosine similarity between two normalized feature vectors."""
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

# -----------------------------------------------------------------------------
# 2. Motion Detector Class
# -----------------------------------------------------------------------------
class MotionDetector:
    def __init__(self, min_area=1500, threshold=25):
        self.prev_frame = None
        self.min_area = min_area        # Min contour area in pixels to count as motion
        self.threshold = threshold      # Pixel difference sensitivity (0-255)

    def process(self, frame):
        """
        Detects motion relative to the previous frame.
        Returns: list of bounding boxes [(x, y, w, h), ...] and overall motion centroid (cx, cy).
        """
        # Downscale and blur for speed and noise reduction
        small_frame = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return [], None

        # Compute absolute difference between current frame and previous frame
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        self.prev_frame = gray

        # Find contours of moving regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_boxes = []
        total_x, total_y, total_count = 0, 0, 0

        # Scale factors back to original resolution
        scale_x = frame.shape[1] / 320.0
        scale_y = frame.shape[0] / 180.0

        for c in contours:
            if cv2.contourArea(c) < (self.min_area / (scale_x * scale_y)):
                continue
            
            x, y, w, h = cv2.boundingRect(c)
            # Rescale box back to full resolution
            full_box = (int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y))
            motion_boxes.append(full_box)

            total_x += full_box[0] + full_box[2] // 2
            total_y += full_box[1] + full_box[3] // 2
            total_count += 1

        motion_centroid = None
        if total_count > 0:
            motion_centroid = (int(total_x / total_count), int(total_y / total_count))

        return motion_boxes, motion_centroid

# -----------------------------------------------------------------------------
# 3. Load Reference Embeddings
# -----------------------------------------------------------------------------
KNOWN_FACES_DIR = "KnownFaces"
known_db = {}

if os.path.exists(KNOWN_FACES_DIR):
    print(f"Loading reference embeddings from '{KNOWN_FACES_DIR}'...")
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0]
            img_path = os.path.join(KNOWN_FACES_DIR, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                continue

            faces = app.get(img)
            if len(faces) > 0:
                ref_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                known_db[name] = ref_face.embedding
                print(f" Registered identity: '{name}'")
            else:
                print(f" Warning: No face found in '{filename}'. Skipping.")

RECOGNITION_THRESHOLD = 0.45

# -----------------------------------------------------------------------------
# 4. Initialize Pygame, Motion Detector & Camera Feed
# -----------------------------------------------------------------------------
pygame.init()
pygame.display.set_caption("PhyzAI Remote Control - Vision (Faces + Motion)")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("Error: Could not open video device.")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

screen = pygame.display.set_mode((frame_width, frame_height))
clock = pygame.time.Clock()
motion_detector = MotionDetector(min_area=2000, threshold=25)

print(f"\nVision loop active at {frame_width}x{frame_height}. Press ESC to exit.")

# -----------------------------------------------------------------------------
# 5. Main Processing Loop
# -----------------------------------------------------------------------------
running = True

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        success, frame = cap.read()
        if not success:
            break

        # Base frame to RGB for rendering
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- A. Motion Detection ---
        motion_boxes, motion_centroid = motion_detector.process(frame)

        for (mx, my, mw, mh) in motion_boxes:
            # Draw cyan dashed-style motion regions
            cv2.rectangle(rgb_frame, (mx, my), (mx + mw, my + mh), (255, 255, 0), 1)

        if motion_centroid:
            mcx, mcy = motion_centroid
            # Draw motion focus target indicator
            cv2.circle(rgb_frame, (mcx, mcy), 6, (255, 255, 0), -1)
            cv2.putText(
                rgb_frame,
                f"Motion ({mcx}, {mcy})",
                (mcx + 10, mcy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
            )

        # --- B. Face Recognition ---
        faces = app.get(frame)

        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            
            label = "Unknown"
            color = (0, 165, 255) # Orange for unknown

            if known_db and face.embedding is not None:
                best_score = -1.0
                best_name = None
                
                for name, ref_emb in known_db.items():
                    score = cosine_similarity(face.embedding, ref_emb)
                    if score > best_score:
                        best_score = score
                        best_name = name

                if best_score >= RECOGNITION_THRESHOLD:
                    label = f"{best_name} ({best_score:.2f})"
                    color = (0, 255, 0) # Green for verified
                else:
                    label = f"Unknown ({best_score:.2f})"

            # Draw thick face rectangle over motion layer
            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                rgb_frame, 
                label, 
                (x1, max(y1 - 10, 20)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                color, 
                2
            )

        # --- C. Render to Pygame Window ---
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