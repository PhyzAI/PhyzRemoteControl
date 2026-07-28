
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

with torch.no_grad():
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w, _ = frame.shape


        # 1. Convert OpenCV BGR -> 3-channel RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. Resize to expected spatial dimensions (width=384, height=288)
        resized = cv2.resize(rgb_frame, (384, 288))

        # 3. Shape transformation:
        # (288, 384, 3) -> (3, 288, 384) -> (1, 1, 3, 288, 384)
        # Dimensions: [Batch Size (B=1), Sequence Length (T=1), Channels (C=3), Height (H=288), Width (W=384)]
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

        # ImageNet normalization across channels
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        # Add Batch (B) and Time (T) dimensions -> [1, 1, 3, 288, 384]
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)

        # 4. Model Forward Pass
        saliency_out = model(tensor, source='salicon')

        # Extract output and squeeze back to 2D image matrix
        # saliency_out typically returns shape [1, 1, 1, 288, 384] or [1, 1, 288, 384]
        saliency_map = torch.sigmoid(saliency_out).squeeze().cpu().numpy()



        # Normalize saliency map to range [0, 255]
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        saliency_map = (saliency_map * 255).astype(np.uint8)

        # Resize back to original webcam frame dimensions
        saliency_map_resized = cv2.resize(saliency_map, (orig_w, orig_h))

        # Apply jet colormap for visual heatmap rendering
        heatmap = cv2.applyColorMap(saliency_map_resized, cv2.COLORMAP_JET)

        # Blend heatmap with original frame
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        # Display side-by-side or overlay
        combined = cv2.hconcat([frame, overlay])

        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(combined, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("UNISAL Saliency - Original (Left) vs PyTorch Saliency Overlay (Right)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

