import os
import sys

# Tell Python to look inside the 'unisal' subfolder for modules
repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unisal")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)


# Import UNISAL modules from the repository
from unisal.model import UNISAL
import unisal.utils as utils

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = UNISAL(sources=['dhf1k', 'salicon']).to(device)

weights_path = "unisal/training_runs/pretrained_unisal/weights_best.pth"

if not os.path.exists(weights_path):
    print(f"Error: File not found at {weights_path}")
    sys.exit(1)

checkpoint = torch.load(weights_path, map_location=device)

# 1. Unpack state_dict if wrapped
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    raw_dict = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    raw_dict = checkpoint["state_dict"]
elif isinstance(checkpoint, dict) and "model" in checkpoint:
    raw_dict = checkpoint["model"]
else:
    raw_dict = checkpoint

# 2. Print sample keys to see exact structure
print("\n=== CHECKPOINT KEYS (First 5) ===")
for k in list(raw_dict.keys())[:5]:
    print(" Checkpoint key:", k)

print("\n=== MODEL EXPECTED KEYS (First 5) ===")
for k in list(model.state_dict().keys())[:5]:
    print(" Model expects:", k)

# 3. Test simple remapping


model_state = model.state_dict()
cleaned_dict = {}

# Suffixes that need case-insensitivity mapping
domains = ['salicon', 'dhf1k']

for key, value in raw_dict.items():
    new_key = key
    
    # Strip common wrapper prefixes if present
    if new_key.startswith("module."):
        new_key = new_key[7:]
    if new_key.startswith("model."):
        new_key = new_key[6:]
        
    # Direct match first
    if new_key in model_state:
        cleaned_dict[new_key] = value
        continue

    # Case-insensitive domain matching (e.g., _SALICON -> _salicon)
    matched = False
    for domain in domains:
        if domain.upper() in new_key:
            alt_key = new_key.replace(domain.upper(), domain.lower())
            if alt_key in model_state:
                cleaned_dict[alt_key] = value
                matched = True
                break
        elif domain.lower() in new_key:
            alt_key = new_key.replace(domain.lower(), domain.upper())
            if alt_key in model_state:
                cleaned_dict[alt_key] = value
                matched = True
                break

missing, unexpected = model.load_state_dict(cleaned_dict, strict=False)

print("\n=== FINAL REMAPPING RESULTS ===")
print(f"Total keys in checkpoint: {len(raw_dict)}")
print(f"Successfully mapped keys: {len(cleaned_dict)}")
print(f"Missing keys remaining:   {len(missing)}")
print(f"Unexpected keys ignored:  {len(unexpected)}")

print("\n=== SAMPLE MISSING KEYS (First 10) ===")
for k in missing[:10]:
    print(" Missing:", k)