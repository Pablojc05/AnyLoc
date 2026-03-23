# %% Set the './../' from the script folder
import os
import sys
from pathlib import Path
dir_name = None
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
    print(f'Current script directory: {dir_name}')
except NameError:
    print('WARN: __file__ not found, trying local')
    dir_name = os.path.abspath('')
lib_path = os.path.realpath(f'{Path(dir_name).parent}')
print(f'Library path: {lib_path}')
# Add to path
if lib_path not in sys.path:
    print(f'Adding library path: {lib_path} to PYTHONPATH')
    sys.path.append(lib_path)
else:
    print(f'Library path {lib_path} already in PYTHONPATH')


# %%
import torch
import numpy as np
from PIL import Image
from utilities import VLAD, get_top_k_recall
from utilities import DinoV2ExtractFeatures
from torchvision import transforms as tvf

# Setup parameters
c_centers_file = os.path.join(lib_path, "demo/cache/vocabulary/dinov2_vitl14/l20_value_c16/aerial/c_centers.pt")
query_img_path = "/media/pjimenez/ExtremeSSD/Backup_portatil_pjimenez/Tarea_Localizacion/AnyLoc2023-Public-Data/Public/Datasets-All/VPAir/queries/00074.png"
db_vlads_descs_file = os.path.join(lib_path, "demo/cache/vocabulary/dinov2_vitl14/l20_value_c16/aerial/descs/db-VPAir.pt")

base_tf = tvf.Compose([
    tvf.ToTensor(),
    tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
])

# Load VLAD with pre-trained cluster centers
vlad = VLAD(num_clusters=16, desc_dim=None, cache_dir=os.path.dirname(c_centers_file))
# Check for cluster centers
if vlad.can_use_cache_vlad():
        print("Using cached VLAD cluster centers")
        vlad.fit(None)
else:    
    print("Cluster centers not found in cache, cannot proceed!")
    raise FileNotFoundError(f"Cluster centers file not found: {c_centers_file}")
# c_centers = torch.load("demo/cache/vocabulary/dinov2_vitl14/l20_value_c16/aerial/c_centers.pt")
# vlad.fit(None)  # Load cluster centers

# Extract DINOv2 features and generate VLAD for query
extractor = DinoV2ExtractFeatures("dinov2_vitl14", layer=20, facet="value", device="cuda")
query_img = Image.open(query_img_path).convert("RGB")
query_img_tensor = base_tf(query_img).to("cuda").unsqueeze(0)  # Transform to tensor
print(f"Original query image tensor shape: {query_img_tensor.shape}")
b, c, h, w = query_img_tensor.shape
h_new, w_new = (h // 14) * 14, (w // 14) * 14
print(f"Cropping query image tensor to: ({h_new}, {w_new}) for compatibility with DINOv2")
query_img_tensor = tvf.CenterCrop((h_new, w_new))(query_img_tensor)
query_features = extractor(query_img_tensor)
print(f"Devices used: query image tensor: {query_img_tensor.device}, query features: {query_features.device}, VLAD cluster centers: {vlad.c_centers.device}")
query_vlad = vlad.generate(query_features.cpu().squeeze()) 

# Generate VLAD for database (batch)
# db_features = [extractor(img).squeeze() for img in db_images]
db_vlads = torch.load(db_vlads_descs_file)

# Find most similar images
distances, indices, _ = get_top_k_recall(
    top_k=[1, 5, 10],
    db=db_vlads,
    qu=query_vlad,
    gt_pos=np.array([[]], dtype=object),  # Empty list for no ground truth
    method="cosine"
)

# Top match
best_match_idx = indices[0][0]
best_match_distance = distances[0][0]

print(f"Shape of indices: {indices.shape}, distances: {distances.shape}")
print(f"Best match index: {best_match_idx}, distance: {best_match_distance}")
print(f"Indices: {indices}, Distances: {distances}")