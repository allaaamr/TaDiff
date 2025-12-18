import torch
import pytorch_lightning as pl

# ---- CHANGE THIS to your actual checkpoint path ----
ckpt_path = "ckpt/best.ckpt"
fixed_path = "ckpt/best_fixed.ckpt"
# -----------------------------------------------------

# Load checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu")

# Print existing keys just for visibility
print("Checkpoint keys before fix:", ckpt.keys())

# # Add the missing version key if not present
# if "pytorch-lightning_version" not in ckpt:
ckpt["pytorch-lightning_version"] = pl.__version__
ckpt["state_dict"] = ckpt["model_state_dict"] 
#     print(f"Added pytorch-lightning_version = {pl.__version__}")
# else:
#     print("Checkpoint already has pytorch-lightning_version")

# # Save the fixed checkpoint
torch.save(ckpt, fixed_path)
print("Saved fixed checkpoint to:", fixed_path)
