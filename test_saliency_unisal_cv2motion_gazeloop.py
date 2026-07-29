# conda activate saliency2_env

import sys
import os
import time
import cv2
import torch
from torchvision import transforms
import numpy as np


# Tell Python to look inside the 'unisal' subfolder for modules
repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unisal")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)


# Import UNISAL modules from the repository
from unisal.model import UNISAL
import unisal.utils as utils



# 1. Select device (CUDA GPU, Apple Silicon MPS, or CPU)
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Using compute device: {device}")



def extract_saliency_peaks_with_radius(sal_map, max_peaks=3, min_distance=30, threshold=0.35):
    """
    Finds local maxima and dynamically calculates the radial size (spread) of each hotspot peak.
    
    Returns:
      peaks: List of tuples [(x, y, score, dynamic_radius), ...]
    """
    sal_masked = sal_map.copy()
    sal_masked[sal_masked < threshold] = 0.0

    peaks = []
    sal_copy = sal_masked.copy()

    for _ in range(max_peaks):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(sal_copy)

        if max_val < threshold:
            break

        peak_x, peak_y = max_loc

        # ----------------------------------------------------
        # Measure Adaptive Radius
        # ----------------------------------------------------
        # Create a binary threshold around this peak (e.g., > 50% of peak height)
        peak_cutoff = max_val * 0.5
        _, binary_region = cv2.threshold(sal_copy, peak_cutoff, 1.0, cv2.THRESH_BINARY)
        binary_region = (binary_region * 255).astype(np.uint8)

        # Find the contour containing the peak
        contours, _ = cv2.findContours(binary_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        calculated_radius = min_distance  # Fallback radius
        for cnt in contours:
            # Check if this contour contains our peak coordinate
            if cv2.pointPolygonTest(cnt, (peak_x, peak_y), False) >= 0:
                # Get minimum enclosing circle for this specific peak blob
                _, r = cv2.minEnclosingCircle(cnt)
                calculated_radius = max(int(r), 15)  # Enforce minimum radius of 15px
                break

        peaks.append((peak_x, peak_y, float(max_val), calculated_radius))

        # Zero out the detected region on sal_copy to find the next independent peak
        cv2.circle(sal_copy, (peak_x, peak_y), max(calculated_radius, min_distance), 0, -1)

    return peaks



###################
# Gaze Functions
###################

import math

class PhyzyGazeManager:
    def __init__(self, dwell_time=2.0, ior_duration=4.0):
        self.current_gaze = None        # (x, y)
        self.current_radius = 40        # Dynamic radius of current target
        self.gaze_start_time = 0.0
        self.dwell_time = dwell_time
        
        # IOR list now stores tuples: (x, y, timestamp, radius)
        self.ior_list = []
        self.ior_duration = ior_duration

    def update(self, candidates, current_time):
        """
        candidates: List of (x, y, score, radius) tuples
        """
        # Clean expired entries
        self.ior_list = [entry for entry in self.ior_list 
                         if (current_time - entry[2]) < self.ior_duration]

        if not candidates:
            return self.current_gaze, self.current_radius

        # Find candidate matching current gaze (or take top candidate)
        best_x, best_y, best_score, best_radius = candidates[0]

        if self.current_gaze is None:
            self.current_gaze = (best_x, best_y)
            self.current_radius = best_radius
            self.gaze_start_time = current_time
        else:
            time_spent = current_time - self.gaze_start_time

            # When dwell timer expires, record IOR with its specific dynamic radius!
            if time_spent >= self.dwell_time:
                curr_x, curr_y = self.current_gaze
                self.ior_list.append((curr_x, curr_y, current_time, self.current_radius))
                
                # Lock onto new target
                self.current_gaze = (best_x, best_y)
                self.current_radius = best_radius
                self.gaze_start_time = current_time

        return self.current_gaze, self.current_radius



def apply_foveal_suppression(img_rgb, ior_list, current_time, ior_radius=50, ior_duration=4.0):
    """
    Suppresses recently visited gaze locations in the input image by zeroing them out,
    forcing the saliency model to find secondary visual targets in the rest of the room.
    """
    masked_img = img_rgb.copy()
    
    for ior_x, ior_y, timestamp in ior_list:
        time_elapsed = current_time - timestamp
        if time_elapsed < ior_duration:
            # Fade suppression strength over time as IOR expires
            # (Starts fully blacked out, gradually returns to normal)
            fade_factor = time_elapsed / ior_duration  # 0.0 (fresh) to 1.0 (expired)
            
            # Apply black mask circle over the visited face/object location
            mask_radius = int(ior_radius * (1.0 - 0.3 * fade_factor))
            
            # Draw black circle on the image passed to UNISAL
            cv2.circle(masked_img, (ior_x, ior_y), mask_radius, (0, 0, 0), -1)

    return masked_img


def suppress_heatmap_around_points(saliency_map, points, radius=180, suppression_factor=0.0):
    """
    Clears or dims the saliency heatmap values around specified points.
    
    Parameters:
      saliency_map: 2D numpy array [0.0, 1.0] or [0, 255]
      points: List of (x, y) tuples in the same coordinate space as saliency_map
      radius: Pixel radius to suppress
      suppression_factor: 0.0 = completely black out / remove heat, 0.2 = keep subtle heat
    """
    suppressed = saliency_map.copy()
    
    for pt in points:
        if pt is None:
            continue
        px, py = int(pt[0]), int(pt[1])
        
        # Create a circular mask for suppression
        mask = np.ones_like(suppressed, dtype=np.float32)
        cv2.circle(mask, (px, py), radius, suppression_factor, -1)
        
        # Apply smooth feathered edge around the suppression circle
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        suppressed = (suppressed * mask).astype(saliency_map.dtype)
        
    return suppressed



def draw_phyzy_eyes(img, gaze_pt, eye_radius=24, pupil_radius=10):
    """Draws a pair of stylized cartoon eyes (2x scaled) centered at gaze_pt (x, y)."""
    if gaze_pt is None:
        return

    gx, gy = gaze_pt
    spacing = eye_radius + 8  # Adjusted spacing to keep 2x eyes proportioned

    left_eye = (gx - spacing, gy)
    right_eye = (gx + spacing, gy)

    for eye_center in [left_eye, right_eye]:
        # White sclera
        cv2.circle(img, eye_center, eye_radius, (255, 255, 255), -1)
        # Black outline
        cv2.circle(img, eye_center, eye_radius, (0, 0, 0), 3)
        # Dark pupil
        cv2.circle(img, eye_center, pupil_radius, (20, 20, 20), -1)
        # Glint highlight (2x offset and size)
        cv2.circle(img, (eye_center[0] - 4, eye_center[1] - 5), 4, (255, 255, 255), -1)





#############################
# Main Loop
#############################

# 1. Initialize with only the matching sources
model = UNISAL(sources=['dhf1k', 'salicon']).to(device)
model.eval()

weights_path = "unisal/training_runs/pretrained_unisal/weights_best.pth"

if os.path.exists(weights_path):
    checkpoint = torch.load(weights_path, map_location=device)
    raw_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

    model_state = model.state_dict()
    cleaned_dict = {}
    domains = ['salicon', 'dhf1k']

    for key, value in raw_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[7:]
        if new_key.startswith("model."):
            new_key = new_key[6:]

        if new_key in model_state:
            cleaned_dict[new_key] = value
            continue

        for domain in domains:
            if domain.upper() in new_key:
                alt_key = new_key.replace(domain.upper(), domain.lower())
                if alt_key in model_state:
                    cleaned_dict[alt_key] = value
                    break
            elif domain.lower() in new_key:
                alt_key = new_key.replace(domain.lower(), domain.upper())
                if alt_key in model_state:
                    cleaned_dict[alt_key] = value
                    break

    missing, unexpected = model.load_state_dict(cleaned_dict, strict=False)
    print(f"Loaded {len(cleaned_dict)} weights into UNISAL (Missing: {len(missing)}).")







# 3. Open Webcam Stream
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam stream.")
    exit(1)

print("Running UNISAL. Press 'q' to quit.")

# Target spatial shape expected by UNISAL (384, 288) or standard ratio
input_height, input_width = 288, 384

# EMA temporal smoothing variables
smoothed_map = None
# Alpha controls weight of current frame:
# 0.2 = heavy smoothing / slow fade
# 0.5 = balanced responsiveness and stability
# 0.8 = fast response / minimal smoothing
alpha = 0.5 


# Initialize motion memory variables before the main loop
prev_gray = None
motion_memory = np.zeros((288, 384), dtype=np.float32)

# Parameters for tuning human-like motion retention:
# decay_rate: How fast attention fades after motion stops.
# 0.85 to 0.92 gives a natural 1.5 to 2.5 second lingering tail.
decay_rate = 0.9

motion_weight = 0.5  # Weight of motion relative to static saliency


gaze_manager = PhyzyGazeManager(dwell_time=2.0, ior_duration=4.0)


with torch.no_grad():
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w, _ = frame.shape

        # ----------------------------------------------------
        # 1. Calculate Real-Time Motion
        # ----------------------------------------------------
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray_frame, (384, 288))
        
        if prev_gray is None:
            prev_gray = gray_resized
            current_motion = np.zeros((288, 384), dtype=np.float32)
        else:
            frame_diff = cv2.absdiff(gray_resized, prev_gray)
            prev_gray = gray_resized

            # Noise floor threshold
            _, motion_thresh = cv2.threshold(frame_diff, 18, 255, cv2.THRESH_TOZERO)
            motion_blur = cv2.GaussianBlur(motion_thresh, (9, 9), 0).astype(np.float32)
            
            max_diff = motion_blur.max()
            if max_diff > 30.0:
                current_motion = motion_blur / max_diff
            else:
                current_motion = np.zeros((288, 384), dtype=np.float32)

        # ----------------------------------------------------
        # 2. Update Motion Memory (Decay & Accumulate)
        # ----------------------------------------------------
        # Decay previous memory frame-by-frame
        motion_memory = motion_memory * decay_rate
        
        # Merge new motion peaks into memory (taking maximum intensity)
        motion_memory = np.maximum(motion_memory, current_motion)



        current_time = time.time()


        # ----------------------------------------------------
        # 3. Run UNISAL Static Saliency (salicon)
        # ----------------------------------------------------
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



        # ----------------------------------------------------
        # 1. Apply Foveal Suppression to Input Image
        # ----------------------------------------------------
        # Black out active IOR locations on a copy of the RGB frame
        suppressed_rgb = apply_foveal_suppression(
            rgb_frame, 
            gaze_manager.ior_list, 
            current_time, 
            ior_radius=180, #60, 
            ior_duration=gaze_manager.ior_duration
        )

        # ----------------------------------------------------
        # 2. Run UNISAL on Suppressed Frame
        # ----------------------------------------------------
        resized = cv2.resize(suppressed_rgb, (384, 288))

        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)

        saliency_out = model(tensor, source='salicon')
        saliency_map_raw = torch.sigmoid(saliency_out).squeeze().cpu().numpy()





        # Normalize static map independently to [0.0, 1.0]
        static_sal = (saliency_map_raw - saliency_map_raw.min()) / (saliency_map_raw.max() - saliency_map_raw.min() + 1e-8)

        # ----------------------------------------------------
        # 4. Blend Decaying Motion Memory into Static Saliency
        # ----------------------------------------------------
        combined_saliency = static_sal + (motion_memory * motion_weight)
        combined_saliency = np.clip(combined_saliency, 0.0, 1.0)

        # ----------------------------------------------------
        # 5. Temporal EMA Smoothing
        # ----------------------------------------------------
        if smoothed_map is None:
            smoothed_map = combined_saliency
        else:
            smoothed_map = alpha * combined_saliency + (1 - alpha) * smoothed_map

        # Map to 0-255 uint8 for visualization
        saliency_map = (smoothed_map * 255).astype(np.uint8)

        # Resize and render
        saliency_map_resized = cv2.resize(saliency_map, (orig_w, orig_h))
        heatmap = cv2.applyColorMap(saliency_map_resized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)



        # ----------------------------------------------------
        # Extract Focus Coordinates for Phyzy
        # ----------------------------------------------------
        # Normalize combined saliency map to 0.0 - 1.0 for peak detection
        sal_norm = (smoothed_map - smoothed_map.min()) / (smoothed_map.max() - smoothed_map.min() + 1e-8)
        
        # Extract top 3 target peaks on low-res map, then scale to camera frame size
        raw_peaks = extract_saliency_peaks(sal_norm, max_peaks=3, min_distance=30, threshold=0.4)

        scale_x = orig_w / 384.0
        scale_y = orig_h / 288.0

        target_coords = []
        for i, (px, py, score) in enumerate(raw_peaks):
            # Scale coordinates to original frame dimensions
            frame_x = int(px * scale_x)
            frame_y = int(py * scale_y)
            target_coords.append((frame_x, frame_y, score))

            # Color coding: Target #1 (Primary) is Green, others are Cyan
            color = (0, 255, 0) if i == 0 else (255, 255, 0)
            
            # Draw crosshair target
            cv2.drawMarker(overlay, (frame_x, frame_y), color, 
                           markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.circle(overlay, (frame_x, frame_y), 12, color, 2)

            # Display Target ID and Confidence Score
            label = f"P{i+1}: ({frame_x},{frame_y}) [{score:.2f}]"
            cv2.putText(overlay, label, (frame_x + 15, frame_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Print top target coordinate to console for Phyzy's pan/tilt control
        if target_coords:
            primary_x, primary_y, score = target_coords[0]
            # print(f"Phyzy Look Target -> X: {primary_x}, Y: {primary_y} (Score: {score:.2f})")




        # Gaze

        #current_time = time.time()

        # Update Phyzy's decision engine
        phyzy_gaze = gaze_manager.update(target_coords, current_time)

        # Draw candidate targets (Cyan/Green crosshairs)
        for i, (frame_x, frame_y, score) in enumerate(target_coords):
            cv2.drawMarker(overlay, (frame_x, frame_y), (255, 255, 0), 
                           markerType=cv2.MARKER_CROSS, markerSize=14, thickness=1)

        # Draw Phyzy's active gaze (The Eyes!)
        if phyzy_gaze is not None:
            draw_phyzy_eyes(overlay, phyzy_gaze)
            #cv2.putText(overlay, "Phyzy Looking Here", (phyzy_gaze[0] - 50, phyzy_gaze[1] + 30),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)






        # Display side-by-side or overlay
        #combined = cv2.hconcat([frame, overlay])
        combined = overlay

        combined = cv2.resize(combined, (0,0), fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)

        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(combined, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("UNISAL Saliency", combined)
        cv2.moveWindow("UNISAL Saliency", 100, 100)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()