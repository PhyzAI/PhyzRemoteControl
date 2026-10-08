# PHYZAI New Vision Processing Pipeline
# with simplified libraries, and simple saliency
# Oct 2026: initial version by TheRFengineer@gmail.com

import os
import sys
import time
import random
import threading
import numpy as np
import cv2  # pip install opencv-contrib-python-headless
import pygame
from insightface.app import FaceAnalysis

# Prevent Cocoa/Objective-C fork warnings on macOS
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# Prevent ONNX Runtime / OpenMP from maxing out all CPU threads on Linux/Intel
# (Prevents starving Pygame display rendering threads)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

# -----------------------------------------------------------------------------
# Global Feature Switches
# -----------------------------------------------------------------------------
ENABLE_SALIENCY = True          # Set to False to disable saliency detection pipeline
SALIENCY_MODE = "U2NETP"  # Options: 'SPECTRAL_RESIDUAL', 'FINE_GRAINED', 'LAB_COLOR', 'U2NETP'
NUM_SALIENCY_POINTS = 3         # Number of top salient points to extract per frame
ENABLE_COREML = False           # Set to False on Linux/Intel machines without CoreML support
ENABLE_FACE_DETECTION = True    # Set to False to disable face detection & tracking entirely
ENABLE_FACE_RECOGNITION = True  # Set to False to disable identity matching against KnownFaces DB
INSIGHTFACE_MODEL = "buffalo_sc" # Options: 'buffalo_sc', 'buffalo_s' (fast/lightweight), 'buffalo_l' (heavy/accurate)
FACE_DET_INTERVAL = 4           # Run heavy face detection every N frames (caching results in between)
DET_SIZE = (320, 320)           # Reduced detection grid for faster processing

# Gaze Cooldown Settings (Inhibition of Return)
COOLDOWN_DURATION = 2.5  # Seconds to ignore a recently selected target region
COOLDOWN_RADIUS = 120    # Radius in pixels around recent target points to ignore

# -----------------------------------------------------------------------------
# 1. Threaded Camera Capture (Fixes V4L2 Frame Buffering/Bursting on Linux)
# -----------------------------------------------------------------------------
class ThreadedCamera:
    """
    Runs OpenCV video capture in a dedicated background thread.
    Constantly flushes hardware camera buffers so the main loop always
    gets the newest available frame without V4L2 burst delays.
    """
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            sys.exit("Error: Could not open video device.")

        # Attempt buffer size hint
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if not grabbed:
                self.stopped = True
                break
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.grabbed, self.frame.copy()

    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()

# -----------------------------------------------------------------------------
# 2. Initialize InsightFace App & Saliency Extractor
# -----------------------------------------------------------------------------
app = None
if ENABLE_FACE_DETECTION:
    providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider'] if ENABLE_COREML else ['CPUExecutionProvider']
    app = FaceAnalysis(name=INSIGHTFACE_MODEL, providers=providers)
    app.prepare(ctx_id=0, det_size=DET_SIZE)

# Saliency Detectors Setup
saliency_detector = None
u2netp_net = None

if ENABLE_SALIENCY:
    if SALIENCY_MODE == "SPECTRAL_RESIDUAL":
        saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()
    elif SALIENCY_MODE == "FINE_GRAINED":
        saliency_detector = cv2.saliency.StaticSaliencyFineGrained_create()
    elif SALIENCY_MODE == "U2NETP":
        # Attempts to load u2netp.onnx if present; falls back to FINE_GRAINED if missing
        if os.path.exists("u2netp.onnx"):
            u2netp_net = cv2.dnn.readNetFromONNX("u2netp.onnx")
            print("Successfully loaded U-2-Net-p ONNX model.")
        else:
            print("Warning: 'u2netp.onnx' not found in working directory. Falling back to FINE_GRAINED saliency.")
            saliency_detector = cv2.saliency.StaticSaliencyFineGrained_create()
            SALIENCY_MODE = "FINE_GRAINED"

def cosine_similarity(emb1, emb2):
    """Calculates cosine similarity between two normalized feature vectors."""
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

def extract_salient_points(frame, num_points=NUM_SALIENCY_POINTS):
    """
    Computes saliency map using the chosen algorithm and extracts top peak interest locations.
    Supported modes: 'SPECTRAL_RESIDUAL', 'FINE_GRAINED', 'LAB_COLOR', 'U2NETP'
    """
    if not ENABLE_SALIENCY:
        return []

    h, w = frame.shape[:2]
    small_frame = cv2.resize(frame, (160, 90))
    sal_map = None

    if SALIENCY_MODE in ("SPECTRAL_RESIDUAL", "FINE_GRAINED"):
        if saliency_detector is not None:
            success, sal_map = saliency_detector.computeSaliency(small_frame)
            if success:
                sal_map = (sal_map * 255).astype(np.uint8)

    elif SALIENCY_MODE == "LAB_COLOR":
        # Center-surround color/luminance contrast map in CIELAB color space
        lab = cv2.cvtColor(small_frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        l_blur = cv2.GaussianBlur(l, (21, 21), 0)
        a_blur = cv2.GaussianBlur(a, (21, 21), 0)
        b_blur = cv2.GaussianBlur(b, (21, 21), 0)
        
        diff_l = cv2.absdiff(l, l_blur)
        diff_a = cv2.absdiff(a, a_blur)
        diff_b = cv2.absdiff(b, b_blur)
        
        combined = diff_l * 0.4 + diff_a * 0.3 + diff_b * 0.3
        sal_map = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    elif SALIENCY_MODE == "U2NETP" and u2netp_net is not None:
        # Lightweight U-2-Net-p neural saliency inference
        blob = cv2.dnn.blobFromImage(small_frame, 1.0/255.0, (160, 160), (0.485, 0.456, 0.406), swapRB=True)
        u2netp_net.setInput(blob)
        out = u2netp_net.forward()
        sal_out = out[0][0]
        sal_out = cv2.resize(sal_out, (160, 90))
        sal_map = cv2.normalize(sal_out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if sal_map is None:
        return []

    # Threshold map to find highest contrast/interest regions
    _, thresh = cv2.threshold(sal_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    points = []
    scale_x = w / 160.0
    scale_y = h / 90.0

    # Sort contours by area to find distinct visual regions
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours[:num_points]:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int((M["m10"] / M["m00"]) * scale_x)
            cy = int((M["m01"] / M["m00"]) * scale_y)
            points.append((cx, cy))

    return points

# -----------------------------------------------------------------------------
# 3. Motion Detector Class
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
# 4. Gaze & Reflex Target Controller with Inhibition-of-Return (Cooldown)
# -----------------------------------------------------------------------------
class GazeController:
    def __init__(self, frame_w, frame_h, cooldown_duration=COOLDOWN_DURATION, cooldown_radius=COOLDOWN_RADIUS):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.salient_targets = []
        
        # Smooth gaze tracking variables
        self.current_gaze = [frame_w // 2, frame_h // 2]
        self.target_gaze = [frame_w // 2, frame_h // 2]
        self.active_target_type = "IDLE"  # Track target type ("FACE", "SALIENT", "MOTION", "IDLE")
        self.last_switch_time = time.time()
        self.hold_duration = 2.5

        # Motion Quick-Glance Reflex state
        self.in_motion_glance = False
        self.glance_start_time = 0.0
        self.glance_duration = 0.7
        self.last_glance_time = 0.0
        self.glance_cooldown = 1.5

        # Target Cooldown (Inhibition of Return) variables
        self.cooldown_duration = cooldown_duration
        self.cooldown_radius = cooldown_radius
        self.recent_targets = []  # List of tuples: (x, y, timestamp)

    def _is_in_cooldown(self, point, now):
        """Checks if a point falls within the spatial radius of any active cooldown zone."""
        self._prune_expired_cooldowns(now)
        px, py = point
        for rx, ry, _ in self.recent_targets:
            if np.hypot(px - rx, py - ry) <= self.cooldown_radius:
                return True
        return False

    def _add_cooldown_zone(self, point, now):
        """Registers a target location to be suppressed for `cooldown_duration` seconds."""
        self.recent_targets.append((point[0], point[1], now))

    def _prune_expired_cooldowns(self, now):
        """Removes cooldown targets older than `cooldown_duration` seconds."""
        self.recent_targets = [t for t in self.recent_targets if (now - t[2]) < self.cooldown_duration]

    def draw_debug_cooldowns(self, img, now):
        """Draws active suppression zones on the video frame."""
        self._prune_expired_cooldowns(now)
        for rx, ry, ts in self.recent_targets:
            remaining = self.cooldown_duration - (now - ts)
            alpha = max(0.2, remaining / self.cooldown_duration)
            cv2.circle(img, (int(rx), int(ry)), self.cooldown_radius, (0, 0, 200), 1, cv2.LINE_AA)

    def update(self, real_face_centers, motion_centroid, salient_points):
        now = time.time()
        self.salient_targets = salient_points

        # Motion Quick Glance Trigger (Only if not in a cooldown zone)
        if motion_centroid is not None and not self._is_in_cooldown(motion_centroid, now):
            if not self.in_motion_glance and (now - self.last_glance_time > self.glance_cooldown):
                self.in_motion_glance = True
                self.glance_start_time = now
                self.last_glance_time = now
                self.target_gaze = list(motion_centroid)
                self.active_target_type = "MOTION"
                self._add_cooldown_zone(motion_centroid, now)

        # Handle active motion glance duration
        if self.in_motion_glance:
            if now - self.glance_start_time > self.glance_duration:
                self.in_motion_glance = False
                self.last_switch_time = 0

        # Continuous Face Tracking: Dynamically move target point as face moves
        if not self.in_motion_glance and self.active_target_type == "FACE" and real_face_centers:
            closest_face = min(real_face_centers, key=lambda f: np.hypot(f[0] - self.target_gaze[0], f[1] - self.target_gaze[1]))
            self.target_gaze = list(closest_face)

        # Ambient Target Selection
        if not self.in_motion_glance:
            if now - self.last_switch_time > self.hold_duration:
                self.last_switch_time = now
                self.hold_duration = random.uniform(1.8, 3.5)

                # Filter out candidates that fall inside active cooldown regions
                valid_salient = [p for p in salient_points if not self._is_in_cooldown(p, now)]
                valid_faces = [f for f in real_face_centers if not self._is_in_cooldown(f, now)]

                # Prioritization: 70% chance face (if available), 30% salient point (or fallback)
                if valid_faces and random.random() < 0.70:
                    chosen = random.choice(valid_faces)
                    self.target_gaze = list(chosen)
                    self.active_target_type = "FACE"
                    self._add_cooldown_zone(chosen, now)
                elif valid_salient:
                    chosen = random.choice(valid_salient)
                    self.target_gaze = list(chosen)
                    self.active_target_type = "SALIENT"
                    self._add_cooldown_zone(chosen, now)
                elif real_face_centers:
                    # Fallback if all face regions are cooling down
                    chosen = random.choice(real_face_centers)
                    self.target_gaze = list(chosen)
                    self.active_target_type = "FACE"
                else:
                    self.target_gaze = [self.frame_w // 2, self.frame_h // 2]
                    self.active_target_type = "IDLE"

        # Exponential smoothing
        smooth_factor = 0.28 if self.in_motion_glance else 0.12
        self.current_gaze[0] += (self.target_gaze[0] - self.current_gaze[0]) * smooth_factor
        self.current_gaze[1] += (self.target_gaze[1] - self.current_gaze[1]) * smooth_factor

        return int(self.current_gaze[0]), int(self.current_gaze[1])

# -----------------------------------------------------------------------------
# 5. Drawing Functions
# -----------------------------------------------------------------------------
def draw_phyzy_eyes(img, gaze_pt, is_glancing=False, eye_radius=48, pupil_radius=20):
    if gaze_pt is None:
        return

    gx, gy = gaze_pt
    spacing = eye_radius + 8

    left_eye = (gx - spacing, gy)
    right_eye = (gx + spacing, gy)
    border_color = (0, 165, 255) if is_glancing else (0, 0, 0)

    for eye_center in [left_eye, right_eye]:
        cv2.circle(img, eye_center, eye_radius, (255, 255, 255), -1)
        cv2.circle(img, eye_center, eye_radius, border_color, 3)
        cv2.circle(img, eye_center, pupil_radius, (20, 20, 20), -1)
        cv2.circle(img, (eye_center[0] - 4, eye_center[1] - 5), 4, (255, 255, 255), -1)

# -----------------------------------------------------------------------------
# 6. Load Known Face Database
# -----------------------------------------------------------------------------
KNOWN_FACES_DIR = "KnownFaces"
known_db = {}

if ENABLE_FACE_DETECTION and ENABLE_FACE_RECOGNITION and os.path.exists(KNOWN_FACES_DIR):
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
# 7. Initialize Pygame & Threaded Camera Feed
# -----------------------------------------------------------------------------
pygame.init()
pygame.display.set_caption("PhyzAI Remote Control - Saliency & Vision")

camera = ThreadedCamera(0)

# Allow camera driver a moment to initialize frame dimensions
time.sleep(0.5)
success, init_frame = camera.read()

frame_width = init_frame.shape[1] if success and init_frame is not None else 1280
frame_height = init_frame.shape[0] if success and init_frame is not None else 720

screen = pygame.display.set_mode((frame_width, frame_height))
clock = pygame.time.Clock()

motion_detector = MotionDetector(min_area=2000, threshold=25)
gaze_controller = GazeController(frame_width, frame_height)

print(f"\nVision loop active at {frame_width}x{frame_height}. Press ESC to exit.")

# -----------------------------------------------------------------------------
# 8. Main Processing Loop
# -----------------------------------------------------------------------------
running = True
prev_frame_time = time.time()
frame_count = 0
cached_face_centers = []
cached_face_draw_data = []

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        success, frame = camera.read()
        if not success or frame is None:
            time.sleep(0.01)
            continue

        display_frame = frame.copy()
        now = time.time()

        # --- A. Calculate FPS ---
        delta_time = now - prev_frame_time
        prev_frame_time = now
        fps = 1.0 / delta_time if delta_time > 0 else 0.0

        # --- B. Motion Detection ---
        motion_boxes, motion_centroid = motion_detector.process(frame)
        for (mx, my, mw, mh) in motion_boxes:
            cv2.rectangle(display_frame, (mx, my), (mx + mw, my + mh), (255, 255, 0), 1)

        # --- C. Saliency Extraction ---
        salient_points = extract_salient_points(frame, num_points=NUM_SALIENCY_POINTS)
        for sx, sy in salient_points:
            cv2.circle(display_frame, (sx, sy), 8, (255, 0, 255), 1)
            cv2.putText(display_frame, "Salient Point", (sx - 35, sy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # --- D. Face Detection & Recognition (Cached every N frames) ---
        if ENABLE_FACE_DETECTION:
            if frame_count % FACE_DET_INTERVAL == 0:
                faces = app.get(frame)
                cached_face_centers = []
                cached_face_draw_data = []

                for face in faces:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    cached_face_centers.append((center_x, center_y))

                    label = "Face"
                    color = (0, 255, 0)

                    # Identity matching only runs if recognition switch is active
                    if ENABLE_FACE_RECOGNITION and known_db and face.embedding is not None:
                        best_score = -1.0
                        best_name = None
                        for name, ref_emb in known_db.items():
                            score = cosine_similarity(face.embedding, ref_emb)
                            if score > best_score:
                                best_score = score
                                best_name = name

                        if best_score >= RECOGNITION_THRESHOLD:
                            label = f"{best_name} ({best_score:.2f})"
                            color = (0, 255, 0)
                        else:
                            label = f"Unknown ({best_score:.2f})"
                            color = (0, 165, 255)

                    cached_face_draw_data.append((x1, y1, x2, y2, label, color))
        else:
            cached_face_centers = []
            cached_face_draw_data = []

        # Render face boxes/labels using cached data on every frame
        real_face_centers = cached_face_centers
        for x1, y1, x2, y2, label, color in cached_face_draw_data:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame_count += 1

        # --- E. Update Gaze Position ---
        gaze_x, gaze_y = gaze_controller.update(real_face_centers, motion_centroid, salient_points)

        # --- F. Draw Active Cooldown Circles & Overlay Eyes ---
        gaze_controller.draw_debug_cooldowns(display_frame, now)
        draw_phyzy_eyes(display_frame, (gaze_x, gaze_y), is_glancing=gaze_controller.in_motion_glance)

        if gaze_controller.in_motion_glance:
            cv2.putText(display_frame, "[REFLEX GLANCE]", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # --- G. Render FPS Overlay (Top-Right Corner) ---
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(display_frame, fps_text, (frame_width - 130, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --- H. Render Screen ---
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        surface_data = np.rot90(rgb_frame)
        surface_data = np.flipud(surface_data)
        surface = pygame.surfarray.make_surface(surface_data)

        screen.blit(surface, (0, 0))
        pygame.display.flip()

        clock.tick(30)

finally:
    camera.release()
    pygame.quit()
    sys.exit(0)