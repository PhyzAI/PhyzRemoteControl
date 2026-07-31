# conda activate saliency2_env

import sys
import os
import time
import cv2
import torch
from torchvision import transforms
import numpy as np


# FEATURE TOGGLES
ENABLE_MOTION_FLASH = False  # Set to False to disable the orange shockwave ring



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



###################
# Gaze Functions
###################

import random
import time
import math

class PhyzyGazeManager:
    def __init__(self, primary_dwell_range=(2.0, 3.5), secondary_dwell_range=(0.8, 1.5), 
                 reflex_dwell_range=(0.6, 1.0), ior_duration=4.0):
        self.current_gaze = None        # (x, y)
        self.current_radius = 40        
        self.gaze_start_time = 0.0
        self.current_dwell_time = 2.0   
        
        self.primary_dwell_range = primary_dwell_range      
        self.secondary_dwell_range = secondary_dwell_range  
        self.reflex_dwell_range = reflex_dwell_range        # Fast glance range (0.4s - 0.8s)
        
        self.is_reflex_lock = False     # Flag for active motion interrupt
        self.ior_list = []
        self.ior_duration = ior_duration
        self.last_reflex_event = None  # Stores {'x', 'y', 'radius', 'time'}

    def _sample_dwell_time(self, mode="primary"):
        if mode == "reflex":
            return random.uniform(*self.reflex_dwell_range)
        elif mode == "primary":
            return random.uniform(*self.primary_dwell_range)
        else:
            return random.uniform(*self.secondary_dwell_range)

    def trigger_motion_interrupt(self, motion_event, current_time, frame_scale=(1.0, 1.0)):
        mx = int(motion_event['x'] * frame_scale[0])
        my = int(motion_event['y'] * frame_scale[1])
        mrad = int(motion_event['radius'] * frame_scale[0])

        # Ignore if in active IOR region
        for ix, iy, timestamp, irad in self.ior_list:
            if (current_time - timestamp) < self.ior_duration:
                if math.hypot(mx - ix, my - iy) < (irad * 0.8):
                    return False

        # Override active gaze immediately
        if self.current_gaze is not None and not self.is_reflex_lock:
            curr_x, curr_y = self.current_gaze
            self.ior_list.append((curr_x, curr_y, current_time, self.current_radius))

        self.current_gaze = (mx, my)
        self.current_radius = mrad
        self.gaze_start_time = current_time
        self.current_dwell_time = self._sample_dwell_time(mode="reflex")
        self.is_reflex_lock = True
        
        # RECORD EVENT FOR VISUAL FLASH
        self.last_reflex_event = {
            'x': mx,
            'y': my,
            'base_radius': mrad,
            'time': current_time
        }
        return True

    def update(self, candidates, current_time):
        self.ior_list = [entry for entry in self.ior_list if (current_time - entry[2]) < self.ior_duration]

        if not candidates and self.current_gaze is None:
            return self.current_gaze, self.current_radius

        time_spent = current_time - self.gaze_start_time

        # Check if active dwell timer (reflex or normal) expired
        if time_spent >= self.current_dwell_time:
            if self.current_gaze is not None:
                curr_x, curr_y = self.current_gaze
                self.ior_list.append((curr_x, curr_y, current_time, self.current_radius))

            self.is_reflex_lock = False  # Clear reflex lock status

            if candidates:
                best_x, best_y, best_score, best_radius = candidates[0]
                self.current_gaze = (best_x, best_y)
                self.current_radius = best_radius
                self.gaze_start_time = current_time
                
                mode = "primary" if best_score >= 0.65 else "secondary"
                self.current_dwell_time = self._sample_dwell_time(mode=mode)
            else:
                self.current_gaze = None

        else:
            # Smooth pursuit tracking during active lock
            if candidates and self.current_gaze is not None:
                for cx, cy, cscore, cradius in candidates:
                    dist = math.hypot(cx - self.current_gaze[0], cy - self.current_gaze[1])
                    if dist < self.current_radius * 1.5:
                        self.current_gaze = (int(0.4 * cx + 0.6 * self.current_gaze[0]),
                                             int(0.4 * cy + 0.6 * self.current_gaze[1]))
                        self.current_radius = cradius
                        break

        return self.current_gaze, self.current_radius
    

def apply_foveal_suppression(rgb_image, ior_list, current_time, ior_duration=4.0, attenuation=0.5):
    """
    Softly dims past gaze regions on the input image using a smooth Gaussian vignette 
    whose blur radius scales proportionally to half of each region's target radius.
    """
    suppressed_img = rgb_image.copy().astype(np.float32)

    for ior_x, ior_y, timestamp, ior_radius in ior_list:
        if (current_time - timestamp) < ior_duration:
            # 1. Base mask (1.0 everywhere)
            mask = np.ones((suppressed_img.shape[0], suppressed_img.shape[1]), dtype=np.float32)
            
            # 2. Draw sharp attenuated inner region
            cv2.circle(mask, (int(ior_x), int(ior_y)), int(ior_radius), attenuation, -1)
            
            # 3. Dynamic blur kernel size scaled to ~half the radius
            # Kernel size must be an odd integer >= 3
            ksize = int(ior_radius * 0.7) # was 0.5
            if ksize % 2 == 0:
                ksize += 1
            ksize = max(3, ksize)
            
            # Sigma set to ~half the target radius
            sigma = ior_radius * 0.5
            
            # 4. Apply smooth Gaussian falloff
            mask = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
            
            # 5. Apply to frame channels
            suppressed_img *= mask[:, :, np.newaxis]

    return np.clip(suppressed_img, 0, 255).astype(np.uint8)


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

def draw_reflex_flash(frame, reflex_event, current_time, duration=0.35):
    """
    Renders an expanding, fading orange shockwave ring at the reflex trigger point.
    Halved in radius for a tighter, more subtle visual indicator.
    """
    # 1. Respect global feature flag toggle
    if not ENABLE_MOTION_FLASH or reflex_event is None:
        return frame

    elapsed = current_time - reflex_event['time']
    if elapsed >= duration:
        return frame  # Animation finished

    progress = elapsed / duration
    
    # 2. HALVED RADIUS: Start at 0.5x base radius and expand up to 0.8x
    cx, cy = reflex_event['x'], reflex_event['y']
    half_base_radius = int(reflex_event['base_radius'] * 0.5)
    radius = int(half_base_radius * (1.0 + 0.6 * progress))
    
    # Opacity fade from 0.8 down to 0.0
    alpha = max(0.0, 0.8 * (1.0 - progress))
    
    overlay = frame.copy()
    orange_color = (0, 140, 255)  # BGR format
    thickness = max(2, int(4 * (1.0 - progress)))
    
    # Draw ring and subtle inner glow
    cv2.circle(overlay, (cx, cy), max(1, radius), orange_color, thickness)
    cv2.circle(overlay, (cx, cy), max(1, radius - thickness), orange_color, -1)
    
    # Blend overlay onto target frame
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    return frame



def draw_phyzy_eyes(img, gaze_pt, eye_radius=2*24, pupil_radius=2*10):
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



###################
## New Functions ##
###################


def process_motion_memory(frame, prev_gray, motion_memory, 
                          decay_rate=0.85, 
                          spike_threshold=0.25,  # Lowered from 0.55 -> 0.25 (sensitive to subtle movement)
                          min_area=150):         # Lowered from 300 -> 150 (detects smaller moving regions)
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray_frame, (384, 288))

    if prev_gray is None:
        return np.zeros((288, 384), dtype=np.float32), gray_resized, np.zeros((288, 384), dtype=np.float32), None

    frame_diff = cv2.absdiff(gray_resized, prev_gray)
    
    # 1. Lower threshold for difference image (12 instead of 20) to capture smooth motion at 5 FPS
    _, motion_thresh = cv2.threshold(frame_diff, 12, 255, cv2.THRESH_BINARY)
    
    # 2. Apply morphological dilation to bridge sparse motion blobs caused by low FPS frame gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion_thresh = cv2.dilate(motion_thresh, kernel, iterations=1)

    motion_blur = cv2.GaussianBlur(frame_diff, (9, 9), 0).astype(np.float32)

    max_diff = motion_blur.max()
    current_motion = motion_blur / max_diff if max_diff > 15.0 else np.zeros((288, 384), dtype=np.float32)

    # Accumulate motion memory
    updated_memory = np.maximum(motion_memory * decay_rate, current_motion)

    # ------------------------------------------------------------------
    # Motion Spike Reflex Detection
    # ------------------------------------------------------------------
    motion_event = None
    if max_diff > (255.0 * spike_threshold):
        contours, _ = cv2.findContours(motion_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        largest_cnt = None
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area and area >= min_area:
                max_area = area
                largest_cnt = cnt

        if largest_cnt is not None:
            (cx, cy), r = cv2.minEnclosingCircle(largest_cnt)
            motion_event = {
                'x': int(cx),
                'y': int(cy),
                'radius': max(int(r), 25),
                'intensity': float(max_diff / 255.0)
            }

    return current_motion, gray_resized, updated_memory, motion_event


def run_unisal_model(model, frame, gaze_manager, current_time, device):
    """
    Applies foveal suppression to input image and runs UNISAL static saliency inference.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply dynamic foveal suppression using each target's stored dynamic radius
    suppressed_rgb = apply_foveal_suppression(
        rgb_frame,
        gaze_manager.ior_list,
        current_time,
        ior_duration=gaze_manager.ior_duration
    )

    resized = cv2.resize(suppressed_rgb, (384, 288))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)

    saliency_out = model(tensor, source='salicon')
    raw_map = torch.sigmoid(saliency_out).squeeze().cpu().numpy()

    # Normalize static saliency to [0.0, 1.0]
    return (raw_map - raw_map.min()) / (raw_map.max() - raw_map.min() + 1e-8)


def extract_saliency_peaks_with_radius(sal_map, max_peaks=3, min_distance=30, threshold=0.35):
    """
    Extracts peak locations and measures their adaptive spatial radiuses.
    """
    sal_copy = sal_map.copy()
    sal_copy[sal_copy < threshold] = 0.0

    peaks = []

    for _ in range(max_peaks):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(sal_copy)

        if max_val < threshold:
            break

        peak_x, peak_y = max_loc

        # Calculate dynamic radius based on surrounding blob contour
        peak_cutoff = max_val * 0.5
        _, binary_region = cv2.threshold(sal_copy, peak_cutoff, 1.0, cv2.THRESH_BINARY)
        binary_region = (binary_region * 255).astype(np.uint8)

        contours, _ = cv2.findContours(binary_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        calculated_radius = min_distance
        for cnt in contours:
            if cv2.pointPolygonTest(cnt, (peak_x, peak_y), False) >= 0:
                _, r = cv2.minEnclosingCircle(cnt)
                calculated_radius = max(int(r), 15)
                break

        peaks.append((peak_x, peak_y, float(max_val), calculated_radius))

        # Clear detected peak area to search for secondary targets
        cv2.circle(sal_copy, (peak_x, peak_y), max(calculated_radius, min_distance), 0, -1)

    return peaks


def apply_saliency_suppression_and_extract_targets(smoothed_map, gaze_manager, current_time, orig_w, orig_h, attenuation=0.80):
    """
    Attenuates saliency around past gaze points using proportional Gaussian blurring.
    """
    saliency_for_peaks = smoothed_map.copy()

    model_scale_x = 384.0 / orig_w
    model_scale_y = 288.0 / orig_h

    for ior_x, ior_y, timestamp, ior_rad in gaze_manager.ior_list:
        if (current_time - timestamp) < gaze_manager.ior_duration:
            ix = int(ior_x * model_scale_x)
            iy = int(ior_y * model_scale_y)
            rad = int(ior_rad * model_scale_x)
            
            mask = np.ones_like(saliency_for_peaks, dtype=np.float32)
            cv2.circle(mask, (ix, iy), rad, attenuation, -1)
            
            # Dynamic kernel size relative to the scaled saliency radius
            ksize = int(rad * 0.9) # was 0.5
            if ksize % 2 == 0:
                ksize += 1
            ksize = max(3, ksize)
            
            sigma = rad * 0.5
            
            mask = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
            saliency_for_peaks *= mask

    # Re-normalize remaining map
    if saliency_for_peaks.max() > 0.1:
        saliency_for_peaks = saliency_for_peaks / saliency_for_peaks.max()

    raw_peaks = extract_saliency_peaks_with_radius(saliency_for_peaks, max_peaks=3, min_distance=30, threshold=0.35)

    target_coords = []
    scale_x = orig_w / 384.0
    scale_y = orig_h / 288.0

    for px, py, score, rad in raw_peaks:
        frame_x = int(px * scale_x)
        frame_y = int(py * scale_y)
        frame_rad = int(rad * scale_x)
        target_coords.append((frame_x, frame_y, score, frame_rad))

    return target_coords, saliency_for_peaks


def draw_targets_and_eyes(overlay, target_coords, phyzy_gaze):
    """
    Renders candidate target markers and Phyzy's eye overlay on the display frame.
    """
    for i, (frame_x, frame_y, score, frame_rad) in enumerate(target_coords):
        color = (0, 255, 0) if i == 0 else (255, 255, 0)
        
        cv2.drawMarker(overlay, (frame_x, frame_y), color, 
                       markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
        cv2.circle(overlay, (frame_x, frame_y), 12, color, 2)

        label = f"P{i+1}: ({frame_x},{frame_y}) [{score:.2f}]"
        cv2.putText(overlay, label, (frame_x + 15, frame_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if phyzy_gaze is not None:
        draw_phyzy_eyes(overlay, phyzy_gaze)






#############################
# Main Loop
#############################

# 1. Initialize UNISAL Model
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

# 2. Open Webcam Stream
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam stream.")
    exit(1)

print("Running UNISAL with Adaptive Hotspot Suppression. Press 'q' to quit.")

# Pipeline State Variables
input_height, input_width = 288, 384
smoothed_map = None
alpha = 0.5 

prev_gray = None
motion_memory = np.zeros((288, 384), dtype=np.float32)
decay_rate = 0.9
motion_weight = 0.5 

gaze_manager = PhyzyGazeManager(
    primary_dwell_range=(1.8, 3.5),   # Deep focus (1.8s to 3.5s)
    secondary_dwell_range=(0.9, 1.8), # Quick glances (0.6s to 1.4s)
    ior_duration=4.0                  # Cooldown duration
)

with torch.no_grad():
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w, _ = frame.shape
        current_time = time.time()

        # 1. Process Motion Memory and Detect Spikes
        current_motion, prev_gray, motion_memory, motion_event = process_motion_memory(
            frame, prev_gray, motion_memory, decay_rate
        )

        # 2. Reflex Fast-Path: Trigger Immediate Glance on Motion Spike
        if motion_event is not None:
            scale_x = orig_w / 384.0
            scale_y = orig_h / 288.0
            
            # Attempt reflex override (returns True if lock was triggered)
            gaze_manager.trigger_motion_interrupt(motion_event, current_time, frame_scale=(scale_x, scale_y))

        # 3. Run UNISAL Static Saliency Model
        static_sal = run_unisal_model(model, frame, gaze_manager, current_time, device)

        # 4. Blend Motion Memory & Apply Temporal Smoothing
        combined_saliency = np.clip(static_sal + (motion_memory * motion_weight), 0.0, 1.0)
        smoothed_map = combined_saliency if smoothed_map is None else (alpha * combined_saliency + (1 - alpha) * smoothed_map)

        # 5. Extract Candidates & Update Gaze State Engine
        target_coords, suppressed_sal_map = apply_saliency_suppression_and_extract_targets(
            smoothed_map, gaze_manager, current_time, orig_w, orig_h
        )

        phyzy_gaze, current_radius = gaze_manager.update(target_coords, current_time)


        # 6. Render Visualization Overlay
        sal_visual = (cv2.resize(suppressed_sal_map, (orig_w, orig_h)) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(sal_visual, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        # Draw eyes and target coordinates
        draw_targets_and_eyes(overlay, target_coords, phyzy_gaze)

        # ---> ADD REFLEX FLASH HERE <---
        overlay = draw_reflex_flash(overlay, gaze_manager.last_reflex_event, current_time)

        # Resize for Display Output
        combined = cv2.resize(overlay, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)




        # Calculate & Show FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(combined, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("UNISAL Saliency", combined)
        cv2.moveWindow("UNISAL Saliency", 100, 100)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()