import os
import sys
import time
import math
import random
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
        self.min_area = min_area
        self.threshold = threshold

    def process(self, frame):
        small_frame = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return [], None

        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        self.prev_frame = gray

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_boxes = []
        total_x, total_y, total_count = 0, 0, 0

        scale_x = frame.shape[1] / 320.0
        scale_y = frame.shape[0] / 180.0

        for c in contours:
            if cv2.contourArea(c) < (self.min_area / (scale_x * scale_y)):
                continue
            
            x, y, w, h = cv2.boundingRect(c)
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
# 3. Gaze & Fake-Face Target Controller
# -----------------------------------------------------------------------------
class GazeController:
    def __init__(self, frame_w, frame_h, num_fake_faces=3):
        self.frame_w = frame_w
        self.frame_h = frame_h
        
        # Configurable fake target positions (simulating other people/points of interest)
        self.fake_targets = [
            (random.randint(100, frame_w - 100), random.randint(100, frame_h - 100))
            for _ in range(num_fake_faces)
        ]
        
        # Smooth gaze tracking variables
        self.current_gaze = [frame_w // 2, frame_h // 2]
        self.target_gaze = [frame_w // 2, frame_h // 2]
        self.last_switch_time = time.time()
        self.hold_duration = 2.5  # Seconds to dwell on a target

    def update(self, real_face_centers, motion_centroid):
        now = time.time()
        
        # Periodically select a new focus target
        if now - self.last_switch_time > self.hold_duration:
            self.last_switch_time = now
            self.hold_duration = random.uniform(1.8, 3.5)
            
            # Occasionally drift/update simulated target locations
            idx = random.randint(0, len(self.fake_targets) - 1)
            self.fake_targets[idx] = (
                random.randint(100, self.frame_w - 100),
                random.randint(100, self.frame_h - 100)
            )

            # Target priority: Real Faces > Motion Centroid > Fake Targets
            if real_face_centers:
                self.target_gaze = list(random.choice(real_face_centers))
            elif motion_centroid and random.random() < 0.7:
                self.target_gaze = list(motion_centroid)
            else:
                self.target_gaze = list(random.choice(self.fake_targets))

        # Exponential smoothing toward active gaze target
        smooth_factor = 0.12
        self.current_gaze[0] += (self.target_gaze[0] - self.current_gaze[0]) * smooth_factor
        self.current_gaze[1] += (self.target_gaze[1] - self.current_gaze[1]) * smooth_factor

        return int(self.current_gaze[0]), int(self.current_gaze[1])

# -----------------------------------------------------------------------------
# 4. Drawing Functions (Overlay Eyes directly on gaze point)
# -----------------------------------------------------------------------------
def draw_phyzy_eyes(img, gaze_pt, eye_radius=48, pupil_radius=20):
    """
    Draws a pair of stylized cartoon eyes centered directly at gaze_pt (x, y).
    """
    if gaze_pt is None:
        return

    gx, gy = gaze_pt
    spacing = eye_radius + 8

    left_eye = (gx - spacing, gy)
    right_eye = (gx + spacing, gy)

    for eye_center in [left_eye, right_eye]:
        # White Sclera
        cv2.circle(img, eye_center, eye_radius, (255, 255, 255), -1)
        # Black Outline
        cv2.circle(img, eye_center, eye_radius, (0, 0, 0), 3)
        # Pupil
        cv2.circle(img, eye_center, pupil_radius, (20, 20, 20), -1)
        # Specular Highlight (Catchlight)
        cv2.circle(img, (eye_center[0] - 4, eye_center[1] - 5), 4, (255, 255, 255), -1)

# -----------------------------------------------------------------------------
# 5. Load Known Face Database
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

RECOGNITION_THRESHOLD = 0.45

# -----------------------------------------------------------------------------
# 6. Initialize Pygame, Camera Feed & Gaze Control
# -----------------------------------------------------------------------------
pygame.init()
pygame.display.set_caption("PhyzAI Remote Control - Vision & Eye Overlay")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("Error: Could not open video device.")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

screen = pygame.display.set_mode((frame_width, frame_height))
clock = pygame.time.Clock()

motion_detector = MotionDetector(min_area=2000, threshold=25)
gaze_controller = GazeController(frame_width, frame_height, num_fake_faces=3)

print(f"\nVision loop active at {frame_width}x{frame_height}. Press ESC to exit.")

# -----------------------------------------------------------------------------
# 7. Main Processing Loop
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

        # Working BGR copy for drawing overlays
        display_frame = frame.copy()

        # --- A. Motion Detection ---
        motion_boxes, motion_centroid = motion_detector.process(frame)
        for (mx, my, mw, mh) in motion_boxes:
            cv2.rectangle(display_frame, (mx, my), (mx + mw, my + mh), (255, 255, 0), 1)

        # --- B. Face Recognition ---
        faces = app.get(frame)
        real_face_centers = []

        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            real_face_centers.append((center_x, center_y))

            label = "Unknown"
            color = (0, 165, 255) # Orange

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
                    color = (0, 255, 0) # Green
                else:
                    label = f"Unknown ({best_score:.2f})"

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- C. Update Gaze Target & Draw Fake Targets ---
        gaze_x, gaze_y = gaze_controller.update(real_face_centers, motion_centroid)

        # Draw quiet markers for configured fake-face targets
        for fx, fy in gaze_controller.fake_targets:
            cv2.circle(display_frame, (fx, fy), 14, (180, 180, 180), 1)
            cv2.putText(display_frame, "Sim Target", (fx - 30, fy + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # --- D. Overlay Cartoon Eyes at Current Look Location ---
        draw_phyzy_eyes(display_frame, (gaze_x, gaze_y))

        # --- E. Render to Pygame Surface ---
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
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