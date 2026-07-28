
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

# 2. Initialize and load pretrained UNISAL model
model = UNISAL(sources=['dhf1k', 'salicon']).to(device)
model.eval()


# Change "weights.pt" to "weights_best.pth"
weights_path = os.path.join("training_runs", "pretrained_unisal", "weights_best.pth")

if not os.path.exists(weights_path):
    # Fallback check in case the script is executed inside the inner directory
    weights_path = os.path.join("unisal", "training_runs", "pretrained_unisal", "weights_best.pth")



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



else:
    print(f"Warning: Could not find weights at {weights_path}.")






transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0)

# EMA Smoothing variable
smoothed_sal = None
alpha = 0.6  # Balance between responsiveness and stability

print("Running UNISAL Sharp Video Saliency. Press 'q' to quit.")

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (384, 224))
        
        # 1. Input MUST be 5D: [Batch=1, Time=1, Channels=3, Height=224, Width=384]
        input_tensor = transform(img_resized).to(device).unsqueeze(0).unsqueeze(0)

        # 2. Forward pass with salicon
        saliency_out = model(input_tensor, source='salicon')
        sal_map = saliency_out.squeeze().cpu().numpy()

        # 3. Dynamic Range Normalization
        sal_map = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8)

        # 4. Temporal EMA filter
        if smoothed_sal is None:
            smoothed_sal = sal_map
        else:
            smoothed_sal = alpha * sal_map + (1 - alpha) * smoothed_sal

        # 5. CUTOFF THRESHOLD: Zero out low background heat (<35%) for sharp regions
        sharp_sal = smoothed_sal.copy()
        sharp_sal[sharp_sal < 0.35] = 0.0

        # Re-normalize the active spots
        if sharp_sal.max() > 0:
            sharp_sal = sharp_sal / sharp_sal.max()

        sal_visual = (sharp_sal * 255).astype(np.uint8)

        # 6. Overlay on original frame
        heatmap = cv2.resize(sal_visual, (frame.shape[1], frame.shape[0]))
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Suppress overlay where saliency is 0
        mask = (heatmap > 0).astype(np.float32)
        mask = cv2.merge([mask, mask, mask])
        
        overlay = (frame * (1 - mask * 0.5) + heatmap_color * (mask * 0.5)).astype(np.uint8)

        cv2.imshow("UNISAL Sharp Saliency", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()