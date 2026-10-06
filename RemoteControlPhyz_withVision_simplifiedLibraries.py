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
# 1. Initialize InsightFace App (Buffalo_l includes detection + recognition)
# -----------------------------------------------------------------------------
# First run will automatically download buffalo_l (~300MB) to ~/.insightface/models/
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def cosine_similarity(emb1, emb2):
    """Calculates cosine similarity between two normalized feature vectors."""
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

# -----------------------------------------------------------------------------
# 2. Build Reference Embeddings from `KnownFaces` Directory
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

            # Extract faces using InsightFace
            faces = app.get(img)
            if len(faces) > 0:
                # Take the largest face found in reference photo
                ref_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                known_db[name] = ref_face.embedding
                print(f" Registered identity: '{name}'")
            else:
                print(f" Warning: No face found in '{filename}'. Skipping.")
else:
    print(f"Directory '{KNOWN_FACES_DIR}' not found. Running in live detection mode only.")

# Recognition threshold (InsightFace cosine similarity usually targets ~0.40–0.50)
RECOGNITION_THRESHOLD = 0.45

# -----------------------------------------------------------------------------
# 3. Initialize Pygame & Camera Feed
# -----------------------------------------------------------------------------
pygame.init()
pygame.display.set_caption("PhyzAI Remote Control - InsightFace Vision")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("Error: Could not open video device.")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

screen = pygame.display.set_mode((frame_width, frame_height))
clock = pygame.time.Clock()

print(f"\nVision loop active at {frame_width}x{frame_height}. Press ESC to exit.")

# -----------------------------------------------------------------------------
# 4. Main Processing Loop
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

        # Process frame with InsightFace (runs detection + embedding extraction)
        faces = app.get(frame)

        # Convert frame to RGB for Pygame display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
                    color = (0, 255, 0) # Green for match
                else:
                    label = f"Unknown ({best_score:.2f})"

            # Draw bounding box & label directly onto image buffer
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

        # Render image array to Pygame window
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