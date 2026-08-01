# PHYZAI New Vision Processing Pipeline Testing
# July 2026: initial version by TheRFengineer@gmail.com


# conda activate saliency2_env
# tested with python 3.11.15, pytorch 2.5.1, py-opencv 5.0.0, numpy 2.4.6
# Instructions for installing unisal are here:
# https://github.com/rdroste/unisal
# I'm not sure all the dependencies they list are correct.  I don't seem to need torchvision.



# TODO:
# X Add randomness to detected point ratingq


# NOTES:
# * When there are groups of people, may need to adjust parameters.  Less motion weight?
# * Secondary loop to find # of faces, and adjust saliency weight accordingly.
#   If many faces, then reduce motion weight, and increase secondary dwell time.



## FEATURE TOGGLES
ENABLE_MOTION_FLASH = True  # Set to False to disable the orange burst
FRAME_PROCESSING_SCALE = 0.5  # Scale frame to 50% for fast processing
SALIENCY_RADIUS_SCALE = 1.1 # Make radius slightly larger than the detected blob for better target coverage


## SALIENCY PARAMETERS
SECONDARY_TARGETS_THRESHOLD = 0.1  # Threshold for secondary target extraction (0.20 = 20% of max saliency)
MAX_PEAKS = 5  # Maximum number of secondary targets to extract per frame
MIN_DISTANCE = 60  # Minimum distance between extracted peaks to avoid clustering
# Note: other parameters are defined in the PhyzyGazeManager instantantiation


## GAZE PARAMETERS
IOR_DURATION = 6.0  # Inhibition of Return duration in seconds.  Was 4.0

# Stochastic noise level for gaze selection (0.0 = deterministic, 0.15 = 15% random variation)
# Adds slight randomness to candidate saliency scores to break deterministic loops and feel organic.
STOCHASTIC_GAZE_NOISE = 0.35


## MOTION TUNING PARAMETERS
# Rate at which past motion fades out per frame (0.0 to 1.0).
# A higher value (0.9) creates a longer temporal "motion trail" or memory, 
# helping keep Phyzy focused on recent movement even if a hand/object briefly stops moving.
MOTION_DECAY_RATE = 0.9

# Blending weight of motion energy vs. static visual saliency (UNISAL).
# At 0.7, dynamic movement (waving hands, moving faces) heavily dominates 
# static visual clutter, ensuring motion reliably triggers gaze shifts and reflex locks.
MOTION_WEIGHT = 0.75  # was 0.5



import sys
import os
import time
import cv2
import torch
import numpy as np
import threading
import random
import math


# Import UNISAL modules from the repository
repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unisal")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)
from unisal.model import UNISAL
import unisal.utils as utils


# Select device (CUDA GPU, Apple Silicon MPS, or CPU)
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

class PhyzyGazeManager:
    def __init__(self, primary_dwell_range=(1.8, 3.5), secondary_dwell_range=(1.0, 1.4), 
                 reflex_dwell_range=(0.6, 1.0), ior_duration=IOR_DURATION, motion_cooldown=1.0):
        """
        Gaze Engine Parameters:
        
        primary_dwell_range   : tuple (min, max) seconds.
                                Random dwell time range for high-confidence focus targets 
                                (e.g., faces or primary objects of interest).
                                
        secondary_dwell_range : tuple (min, max) seconds.
                                Random dwell time range for quick environment checks and 
                                background saliency points.
                                
        reflex_dwell_range    : tuple (min, max) seconds.
                                Dwell duration for motion interrupts (e.g., hand waves or 
                                sudden movement), ensuring glances feel fast and responsive.
                                
        ior_duration          : float, seconds.
                                Cooldown period for Inhibition of Return (IOR). Suppresses 
                                recently visited locations to encourage natural gaze shifts.
                                
        motion_cooldown       : float, seconds.
                                Refractory period after a motion glance completes, preventing 
                                Phyzy from ping-ponging between rapid motion triggers.
        """

        self.current_gaze = None        
        self.current_radius = 40        
        self.gaze_start_time = 0.0
        self.current_dwell_time = 2.0   
        
        # Gaze Anchor Stack
        self.anchor_gaze = None         # Stores (x, y, radius) of the primary target before a glance
        self.is_glance = False          # Flag indicating active target is a temporary glance
        
        self.primary_dwell_range = primary_dwell_range      
        self.secondary_dwell_range = secondary_dwell_range  
        self.reflex_dwell_range = reflex_dwell_range        
        
        self.is_reflex_lock = False     
        self.last_reflex_time = 0.0      
        self.motion_cooldown = motion_cooldown  
        
        self.ior_list = []
        self.ior_duration = ior_duration
        self.last_reflex_event = None  

    def trigger_motion_interrupt(self, motion_event, current_time, frame_scale=(1.0, 1.0)):
        mx = int(motion_event['x'] * frame_scale[0])
        my = int(motion_event['y'] * frame_scale[1])
        mrad = int(motion_event['radius'] * frame_scale[0])

        # 1. Continuity check: If already looking at motion nearby, track smoothly
        if self.is_reflex_lock and self.current_gaze is not None:
            dist_to_current = math.hypot(mx - self.current_gaze[0], my - self.current_gaze[1])
            if dist_to_current < (self.current_radius * 2.0):
                self.current_gaze = (int(0.3 * mx + 0.7 * self.current_gaze[0]),
                                     int(0.3 * my + 0.7 * self.current_gaze[1]))
                self.current_radius = mrad
                return False  

        # 2. Refractory Check
        if (current_time - self.last_reflex_time) < self.motion_cooldown:
            return False

        # 3. IOR Check
        for ix, iy, timestamp, irad in self.ior_list:
            if (current_time - timestamp) < self.ior_duration:
                if math.hypot(mx - ix, my - iy) < (irad * 0.8):
                    return False

        # 4. SAVE ANCHOR: Save current focus target before taking the reflex jump!
        if self.current_gaze is not None and not self.is_glance and not self.is_reflex_lock:
            self.anchor_gaze = (self.current_gaze[0], self.current_gaze[1], self.current_radius)

        # Execute reflex interrupt
        self.current_gaze = (mx, my)
        self.current_radius = mrad
        self.gaze_start_time = current_time
        self.current_dwell_time = random.uniform(*self.reflex_dwell_range)
        self.is_reflex_lock = True
        self.is_glance = True  # Mark as temporary glance
        
        self.last_reflex_event = {
            'x': mx, 'y': my, 'base_radius': mrad, 'time': current_time
        }
        return True

    def update(self, candidates, current_time):
        self.ior_list = [entry for entry in self.ior_list if (current_time - entry[2]) < self.ior_duration]

        if not candidates and self.current_gaze is None:
            return self.current_gaze, self.current_radius

        time_spent = current_time - self.gaze_start_time

        # Active Dwell Expired -> Decide Next Look Location
        if time_spent >= self.current_dwell_time:
            if self.current_gaze is not None:
                curr_x, curr_y = self.current_gaze
                # Suppress the secondary/glance target so we don't immediately look right back at it
                self.ior_list.append((curr_x, curr_y, current_time, self.current_radius))

            if self.is_reflex_lock:
                self.last_reflex_time = current_time
                self.is_reflex_lock = False

            # ------------------------------------------------------------------
            # RETURN TO ANCHOR LOGIC
            # ------------------------------------------------------------------
            returned_to_anchor = False
            if self.is_glance and self.anchor_gaze is not None:
                ax, ay, arad = self.anchor_gaze
                
                # Verify if the anchor is still a valid candidate (or close to one)
                anchor_valid = False
                for cx, cy, cscore, cradius in candidates:
                    if math.hypot(cx - ax, cy - ay) < (arad * 2.0):
                        # Anchor is still present! Snap back directly to candidate
                        self.current_gaze = (cx, cy)
                        self.current_radius = cradius
                        anchor_valid = True
                        break
                
                if anchor_valid:
                    self.gaze_start_time = current_time
                    self.current_dwell_time = random.uniform(*self.primary_dwell_range)
                    self.is_glance = False
                    self.anchor_gaze = None
                    returned_to_anchor = True

            # Standard selection if no anchor or anchor no longer exists
            if not returned_to_anchor:
                self.is_glance = False
                self.anchor_gaze = None
                
                # Filter candidates to find non-suppressed candidates OUTSIDE IOR zones
                valid_candidates = []
                for cx, cy, cscore, cradius in candidates:
                    in_ior = False
                    for ix, iy, itime, irad in self.ior_list:
                        # Check distance against active IOR entries
                        if math.hypot(cx - ix, cy - iy) < (irad * 1.5):
                            in_ior = True
                            break
                    if not in_ior:
                        # Add Gaussian stochastic noise to candidate score for natural choice variation
                        stochastic_score = cscore + random.gauss(0, STOCHASTIC_GAZE_NOISE)
                        valid_candidates.append((cx, cy, cscore, cradius, stochastic_score))

                if valid_candidates:
                    # Sort candidates by noisy score to introduce spontaneous gaze choices
                    valid_candidates.sort(key=lambda item: item[4], reverse=True)
                    best_x, best_y, best_score, best_radius, _ = valid_candidates[0]
                elif candidates:
                    # SAFEGUARD: If all peaks were in IOR, fallback to top raw candidate
                    # to keep gaze continuous and avoid vanishing eyes
                    best_x, best_y, best_score, best_radius = candidates[0]
                else:
                    best_x, best_y, best_score, best_radius = None, None, None, None

                if best_x is not None:
                    # Store current face location as anchor if branching to a secondary glance
                    if self.current_gaze is not None and not self.is_glance:
                        self.anchor_gaze = (self.current_gaze[0], self.current_gaze[1], self.current_radius)

                    self.current_gaze = (best_x, best_y)
                    self.current_radius = best_radius
                    self.gaze_start_time = current_time
                    
                    is_primary = (best_score >= 0.85)
                    self.is_glance = not is_primary
                    
                    dwell_range = self.secondary_dwell_range if self.is_glance else self.primary_dwell_range
                    self.current_dwell_time = random.uniform(*dwell_range)
                else:
                    # Only clear gaze if candidates list was completely empty from the extraction step
                    self.current_gaze = None


        else:
            # Smooth pursuit tracking during active dwell
            if candidates and self.current_gaze is not None:
                # Find the candidate closest to our current gaze location
                best_candidate = None
                min_dist = float('inf')

                for cx, cy, cscore, cradius in candidates:
                    dist = math.hypot(cx - self.current_gaze[0], cy - self.current_gaze[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_candidate = (cx, cy, cradius)

                # If the candidate is within reasonable tracking range, smoothly move towards it
                if best_candidate and min_dist < (self.current_radius * 2.5):
                    cx, cy, cradius = best_candidate
                    self.current_gaze = (int(0.4 * cx + 0.6 * self.current_gaze[0]),
                                         int(0.4 * cy + 0.6 * self.current_gaze[1]))
                    self.current_radius = cradius

        return self.current_gaze, self.current_radius




######################
# Webcam
######################

class ThreadedWebcam:
    """
    Grabs video frames asynchronously in a background thread to prevent 
    neural network inference latency from throttling camera sampling rate.
    """
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        
        # Start background capture thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()




##################
# Drawing
##################


def draw_reflex_flash(frame, reflex_event, current_time, duration=0.35):
    """
    Renders an expanding, fading orange shockwave ring at the reflex trigger point.
    """
    if not ENABLE_MOTION_FLASH or reflex_event is None:
        return frame

    elapsed = current_time - reflex_event['time']
    if elapsed >= duration:
        return frame  # Animation finished

    progress = elapsed / duration
    
    cx, cy = reflex_event['x'], reflex_event['y']
    half_base_radius = int(reflex_event['base_radius'] * 0.5)
    radius = int(half_base_radius * (1.0 + 0.6 * progress))
    
    alpha = max(0.0, 0.8 * (1.0 - progress))
    
    overlay = frame.copy()
    orange_color = (0, 140, 255)  # BGR format
    thickness = max(2, int(4 * (1.0 - progress)))
    
    cv2.circle(overlay, (cx, cy), max(1, radius), orange_color, thickness)
    cv2.circle(overlay, (cx, cy), max(1, radius - thickness), orange_color, -1)
    
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    return frame


def draw_phyzy_eyes(img, gaze_pt, eye_radius=2*24, pupil_radius=2*10):
    """Draws a pair of stylized cartoon eyes (2x scaled) centered at gaze_pt (x, y)."""
    if gaze_pt is None:
        return

    gx, gy = gaze_pt
    spacing = eye_radius + 8

    left_eye = (gx - spacing, gy)
    right_eye = (gx + spacing, gy)

    for eye_center in [left_eye, right_eye]:
        cv2.circle(img, eye_center, eye_radius, (255, 255, 255), -1)
        cv2.circle(img, eye_center, eye_radius, (0, 0, 0), 3)
        cv2.circle(img, eye_center, pupil_radius, (20, 20, 20), -1)
        cv2.circle(img, (eye_center[0] - 4, eye_center[1] - 5), 4, (255, 255, 255), -1)


def draw_targets_and_eyes(overlay, target_coords, phyzy_gaze):
    """Renders candidate target markers and Phyzy's eye overlay on the display frame."""
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





###################
## Saliency Core ##
###################

def process_motion_memory(frame, prev_gray, motion_memory, 
                          MOTION_DECAY_RATE=0.85, 
                          spike_threshold=0.25,  
                          min_area=150):         
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray_frame, (384, 288))

    if prev_gray is None:
        return np.zeros((288, 384), dtype=np.float32), gray_resized, np.zeros((288, 384), dtype=np.float32), None

    frame_diff = cv2.absdiff(gray_resized, prev_gray)
    
    _, motion_thresh = cv2.threshold(frame_diff, 12, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion_thresh = cv2.dilate(motion_thresh, kernel, iterations=1)

    motion_blur = cv2.GaussianBlur(frame_diff, (9, 9), 0).astype(np.float32)

    max_diff = motion_blur.max()
    current_motion = motion_blur / max_diff if max_diff > 15.0 else np.zeros((288, 384), dtype=np.float32)

    updated_memory = np.maximum(motion_memory * MOTION_DECAY_RATE, current_motion)

    # Motion Spike Reflex Detection
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


def run_unisal_model(model, frame, device):
    """
    Pure UNISAL static saliency inference pass without pre-darkening artifacts.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_frame, (384, 288))
    
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
                calculated_radius = max(int(r * SALIENCY_RADIUS_SCALE), 15)
                break

        peaks.append((peak_x, peak_y, float(max_val), calculated_radius))

        # Soft clear around detected candidate to find distinct secondary targets
        cv2.circle(sal_copy, (peak_x, peak_y), max(calculated_radius, min_distance), 0, -1)

    return peaks


def apply_saliency_suppression_and_extract_targets(smoothed_map, gaze_manager, current_time, orig_w, orig_h):
    """
    Applies Gaussian attenuation to historical IOR points, gracefully scales 
    secondary peaks, and ensures target candidates remain stable.
    """
    map_h, map_w = smoothed_map.shape
    attenuation_mask = np.ones((map_h, map_w), dtype=np.float32)
    ys, xs = np.ogrid[:map_h, :map_w]

    model_scale_x = float(map_w) / float(orig_w)
    model_scale_y = float(map_h) / float(orig_h)

    suppression_points = []

    # 1. Gather historical IOR points (past dwells)
    for ior_x, ior_y, timestamp, ior_rad in gaze_manager.ior_list:
        elapsed = current_time - timestamp
        if elapsed < gaze_manager.ior_duration:
            decay_factor = 1.0 - (elapsed / gaze_manager.ior_duration)
            ix = ior_x * model_scale_x
            iy = ior_y * model_scale_y
            rad = ior_rad * model_scale_x
            # 0.90 Notch depth (deep suppression without total blackout)
            suppression_points.append((ix, iy, rad * 1.2, decay_factor * 0.90))

    # Apply continuous Gaussian notches
    for cx, cy, radius, weight in suppression_points:
        sigma = max(radius * 0.85, 8.0)
        dist_sq = (xs - cx)**2 + (ys - cy)**2
        gaussian_notch = 1.0 - (weight * np.exp(-dist_sq / (2.0 * sigma**2)))
        attenuation_mask *= gaussian_notch

    # 2. Attenuate the smoothed saliency map
    saliency_attenuated = smoothed_map * np.clip(attenuation_mask, 0.05, 1.0)

    # ------------------------------------------------------------------
    # 3. STABLE RE-NORMALIZATION: Scale by max remaining peak
    # ------------------------------------------------------------------
    map_max = saliency_attenuated.max()
    
    # If a valid secondary feature exists (max > 0.08), scale it up to 1.0
    if map_max > 0.08:
        saliency_for_peaks = saliency_attenuated / map_max
    else:
        saliency_for_peaks = saliency_attenuated

    # 4. Extract candidates (threshold at 0.10 for secondary targets)
    raw_peaks = extract_saliency_peaks_with_radius(
        saliency_for_peaks, max_peaks=MAX_PEAKS, min_distance=MIN_DISTANCE, threshold=SECONDARY_TARGETS_THRESHOLD
    )

    # 5. SAFEGUARD: If threshold filtered out everything, grab the single highest peak
    if not raw_peaks and map_max > 0.02:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(saliency_for_peaks)
        raw_peaks = [(max_loc[0], max_loc[1], float(max_val), 30)]

    target_coords = []
    scale_x = float(orig_w) / float(map_w)
    scale_y = float(orig_h) / float(map_h)

    for px, py, score, rad in raw_peaks:
        frame_x = int(px * scale_x)
        frame_y = int(py * scale_y)
        frame_rad = int(rad * scale_x)
        target_coords.append((frame_x, frame_y, score, frame_rad))

    return target_coords, saliency_for_peaks





#############################
# Main Loop
#############################

# Initialize UNISAL Model
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


# Open Webcam Stream
cap = ThreadedWebcam(src=0)
time.sleep(0.5)

print("Running UNISAL with Adaptive Hotspot Suppression. Press 'q' to quit.")

# Pipeline State Variables
input_height, input_width = 288, 384
smoothed_map = None
alpha = 0.5 

prev_gray = None
motion_memory = np.zeros((288, 384), dtype=np.float32)


gaze_manager = PhyzyGazeManager(
    primary_dwell_range=(1.8, 3.5),   # Deep focus (1.8s to 3.5s)
    secondary_dwell_range=(0.9, 1.8), # Quick glances (0.9s to 1.8s)
    ior_duration=4.0                  # Cooldown duration
)


with torch.no_grad():
    while True:
        start_time = time.time()
        ret, raw_frame = cap.read()
        if not ret or raw_frame is None:
            continue

        if FRAME_PROCESSING_SCALE != 1.0:
            frame = cv2.resize(raw_frame, (0, 0), 
                               fx=FRAME_PROCESSING_SCALE, 
                               fy=FRAME_PROCESSING_SCALE, 
                               interpolation=cv2.INTER_AREA)
        else:
            frame = raw_frame

        orig_h, orig_w, _ = frame.shape
        current_time = time.time()

        # 1. Process Motion Memory and Detect Spikes
        current_motion, prev_gray, motion_memory, motion_event = process_motion_memory(
            frame, prev_gray, motion_memory, MOTION_DECAY_RATE
        )

        # 2. Reflex Fast-Path: Trigger Immediate Glance on Motion Spike
        if motion_event is not None:
            scale_x = orig_w / 384.0
            scale_y = orig_h / 288.0
            gaze_manager.trigger_motion_interrupt(motion_event, current_time, frame_scale=(scale_x, scale_y))

        # 3. Run UNISAL Static Saliency Model (Pure inference, no image dark spots)
        static_sal = run_unisal_model(model, frame, device)

        # 4. Blend Motion Memory & Apply Temporal Smoothing
        combined_saliency = np.clip(static_sal + (motion_memory * MOTION_WEIGHT), 0.0, 1.0)
        smoothed_map = combined_saliency if smoothed_map is None else (alpha * combined_saliency + (1 - alpha) * smoothed_map)

        # 5. Single Pass: Unified Smooth Attenuation & Candidate Extraction
        target_coords, suppressed_sal_map = apply_saliency_suppression_and_extract_targets(
            smoothed_map, gaze_manager, current_time, orig_w, orig_h
        )

        phyzy_gaze, current_radius = gaze_manager.update(target_coords, current_time)


        # 6. Render Visualization Overlay
        # Render suppressed_sal_map directly (already scaled in Step 5)
        sal_visual = (cv2.resize(suppressed_sal_map, (orig_w, orig_h)) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(sal_visual, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        
        draw_targets_and_eyes(overlay, target_coords, phyzy_gaze)
        overlay = draw_reflex_flash(overlay, gaze_manager.last_reflex_event, current_time)

        combined = overlay

        # Calculate & Show FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(combined, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("UNISAL Saliency", combined)
        cv2.moveWindow("UNISAL Saliency", 100, 100)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up threaded camera resource
cap.release()
cv2.destroyAllWindows()