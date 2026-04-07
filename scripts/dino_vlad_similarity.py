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
import numpy as np
import torch
from torch.nn import functional as F
import einops as ein
from PIL import Image
import matplotlib.pyplot as plt
from dino_extractor import ViTExtractor
from utilities import VLAD, get_top_k_similarities, seed_everything
from sklearn.cluster import DBSCAN
from torchvision import transforms as tvf
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Union, Literal
from tqdm.auto import tqdm
from configs import ProgArgs, prog_args, BaseDatasetArgs, \
        base_dataset_args, device
import time
import joblib
import tyro
import traceback
from dvgl_benchmark.datasets_ws import BaseDataset
from custom_datasets.catec_dataloader import Catec

# Setup parameters
cache_dir = os.path.join(lib_path, "cache_foundloc")
datasets_dir = "/media/upia/c752a2d6-42ac-4005-b6ed-6a16c25ba66b/pjimenez/aerial_datasets"
dataset_name = "fuentes_andalucia"
query_img_path = os.path.join(datasets_dir, "fuentes_andalucia/queries_samples/muestra_vuelo_11mar_5.png")
vlad_cache_dir = os.path.join(cache_dir, "vocabulary/dino_vits8/l9_key_c128/aerial")
vlad_save_dir = os.path.join(vlad_cache_dir, "descs")


base_tf = tvf.Compose([
    tvf.ToTensor(),
    tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
])

@dataclass
class LocalArgs:
    # Program arguments (dataset directories and wandb only)
    prog: ProgArgs = ProgArgs(cache_dir=cache_dir, data_vg_dir=datasets_dir, vg_dataset_name=dataset_name,
                             wandb_proj="Dino-v1-Descs", wandb_entity="catec", wandb_group="VLAD-Descs", wandb_run_name="catec_demo")
    # BaseDataset arguments
    bd_args: BaseDatasetArgs = base_dataset_args
    # Experiment identifier (None = don't use) [won't be used for caching]
    exp_id: Union[str, None] = "ablations/dino_vlad_similarity"
    # VLAD Caching directory (None = don't cache)
    vlad_cache_dir: Path = vlad_cache_dir
    # VLAD Caching for the database and query
    vlad_cache_db_qu: bool = True
    """
        If the `vlad_cache_dir` is not None (then VLAD caching is 
        turned on), this flag controls whether the database and query
        images are cached. If False, then only the cluster centers are
        cached. If True, then database and query image VLADs are also
        cached. This is controlled by the ID for caching (it's made
        None in case of no caching).
    """
    # Resize the image (doesn't work, always 320, 320)
    resize: Tuple[int, int] = field(default_factory=lambda: (320, 320))
    # Number of clusters for VLAD
    num_clusters: int = 128
    # Dataset split for VPR (BaseDataset)
    data_split: Literal["train", "test", "val"] = "test"
    # Dino parameters
    # Model type
    model_type: Literal["dino_vits8", "dino_vits16", "dino_vitb8", 
            "dino_vitb16", "vit_small_patch8_224", 
            "vit_small_patch16_224", "vit_base_patch8_224", 
            "vit_base_patch16_224"] = "dino_vits8"
    """
        Model for Dino-v2 to use as the base model.
    """
    # Down-scaling H, W resolution for images (before giving to Dino)
    down_scale_res: Tuple[int, int] = (224, 298)
    # Layer for extracting Dino feature (descriptors)
    desc_layer: int = 9
    # Facet for extracting descriptors
    desc_facet: Literal["query", "key", "value", "token"] = "key"
    # Apply log binning to the descriptor
    desc_bin: bool = False
    # Sub-sample query images (RAM or VRAM constraints) (1 = off)
    sub_sample_qu: int = 1
    # Sub-sample database images (RAM or VRAM constraints) (1 = off)
    sub_sample_db: int = 1
    # Sub-sample database images for VLAD clustering only
    sub_sample_db_vlad: int = 1
    """
        Use sub-sampling for creating the VLAD cluster centers. Use
        this to reduce the RAM usage during the clustering process.
        Unlike `sub_sample_qu` and `sub_sample_db`, this is only used
        for clustering and not for the actual VLAD computation.
    """
    # Values for top-k (for monitoring)
    top_k_vals: List[int] = field(default_factory=lambda:\
                                list(range(1, 11, 1)))
    # Show a matplotlib plot for recalls
    show_plot: bool = False
    # Use hard or soft descriptor assignment for VLAD
    vlad_assignment: Literal["hard", "soft"] = "hard"
    # Softmax temperature for VLAD (soft assignment only)
    vlad_soft_temp: float = 1.0
    # Stride for ViT (extractor)
    vit_stride: int = 4
    # Save Database and Query VLAD descriptors (final)
    save_vlad_descs: Optional[Path] = vlad_save_dir
    """
        Internal use only (don't set normally). Save the database and
        query VLAD descriptors to this folder. The file name is
        `db-<prog.vg_dataset_name>.pt` (for database VLADs) and 
        `qu-<prog.vg_dataset_name>.pt` (for query VLADs).
        This is independent of the caching mechanisms.
    """
    # Use stored VLAD descriptors for the database and query (if available)
    use_cache_vlad_descs: bool = True

@torch.no_grad()
def build_vlads(largs: LocalArgs, ds: BaseDataset):
    """Build VLAD descriptors for the database."""
    cache_dir = largs.vlad_cache_dir
    if cache_dir is not None:
        print(f"Using cache directory: {cache_dir}")
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
            print(f"Directory created: {cache_dir}")
        else:
            print(f"Directory already exists: {cache_dir}")

    vlad = VLAD(largs.num_clusters, None, 
            vlad_mode=largs.vlad_assignment, 
            soft_temp=largs.vlad_soft_temp, cache_dir=cache_dir)
    print("VLAD instance created")
    
    # Load Dino feature extractor model
    extractor = ViTExtractor(largs.model_type, largs.vit_stride,
                device=device)
    print("Dino model loaded")

    def extract_vlad_descriptors(indices):
        descs = []
        for idx in tqdm(indices, desc="Extracting VLAD descriptors for database"):
            img = ds[idx][0]
            img = ein.rearrange(img, "c h w -> 1 c h w").to(device)
            img = F.interpolate(img, largs.down_scale_res)
            ret = extractor.extract_descriptors(img, 
                    layer=largs.desc_layer, facet=largs.desc_facet) # [1, 1, num_descs, d_dim]
            descs.append(ret.cpu()[0])
        descs = torch.cat(descs, dim=0)  # [num_images, num_descs, d_dim]
        return descs
    
    # Check for cluster centers
    if vlad.can_use_cache_vlad():
            print("Using cached VLAD cluster centers")
            vlad.fit(None)
    else:    
        print("Cluster centers not found in cache, cannot proceed!")
        raise FileNotFoundError(f"Cluster centers file not found: {cache_dir}/c_centers.pt")


    # Database VLADs
    c_dbq = largs.vlad_cache_db_qu
    num_db = ds.database_num
    db_indices = np.arange(0, num_db, largs.sub_sample_db)
    db_img_names = ds.get_image_relpaths(db_indices)
    print("Building VLADs for database...")
    # Load vlad descriptors from cache files if they exist
    if largs.use_cache_vlad_descs and largs.save_vlad_descs is not None:
        db_file = f"{largs.save_vlad_descs}/db-{largs.prog.vg_dataset_name}.pt"
        if os.path.exists(db_file):
            print(f"Loading database VLADs from {db_file}")
            db_vlads = torch.load(db_file)
            print(f"Loaded database VLADs shape: {db_vlads.shape}")
        else:
            print(f"Saved database VLADs not found at {db_file}, computing from scratch")
    elif c_dbq and vlad.can_use_cache_ids(db_img_names):
        print("Using cached ids VLADs for database images")
        db_vlads = vlad.generate_multi([None] * len(db_img_names), 
                db_img_names)
    else:
        print("Valid cache not found, doing forward pass")
        full_db = extract_vlad_descriptors(db_indices)
        if not c_dbq:
            db_img_names = [None] * len(db_img_names)
        db_vlads: torch.Tensor = vlad.generate_multi(full_db, 
                db_img_names)
        del full_db
        print(f"Database VLADs shape: {db_vlads.shape}")

    # Query VLAD
    query_img = Image.open(query_img_path).convert("RGB")
    # query_img_tensor = base_tf(query_img).to("cuda").unsqueeze(0)  # Transform to tensor
    query_img = ein.rearrange(base_tf(query_img), "c h w -> 1 c h w").to(device)
    query_img = F.interpolate(query_img, largs.down_scale_res)
    print(f"Original query image tensor shape: {query_img.shape}")
    # b, c, h, w = query_img.shape
    # h_new, w_new = (h // 14) * 14, (w // 14) * 14
    # print(f"Cropping query image tensor to: ({h_new}, {w_new}) for compatibility with DINOv2")
    # query_img_tensor = tvf.CenterCrop((h_new, w_new))(query_img_tensor)
    query_features = extractor.extract_descriptors(query_img, 
                    layer=largs.desc_layer, facet=largs.desc_facet) # [1, 1, num_descs, d_dim]
    print(f"Extracted query features shape: {query_features.shape}")
    query_desc = query_features.cpu()[0][0]  # [num_descs, d_dim]
    query_vlad = vlad.generate(query_desc, None)  # [vlad_dim]
    print(f"Query VLAD shape: {query_vlad.shape}")

    return db_vlads, query_vlad

@torch.no_grad()
def main(largs: LocalArgs):
    print(f"Arguments: {largs}")
    seed_everything(42)

    print("\n--------- Loading datasets ---------")
    ds_dir = largs.prog.data_vg_dir
    ds_split = largs.data_split
    ds_name = largs.prog.vg_dataset_name
    print(f"Dataset name {ds_name} from directory {ds_dir} with split: {ds_split}")

    # Load dataset
    if ds_name=="fuentes_andalucia":
        ds = Catec(largs.bd_args, ds_dir, ds_name, ds_split)
    else:
        print(f"Dataset {ds_name} not supported!")
        raise NotImplementedError(f"Dataset {ds_name} not supported yet")
        
    print("--------- Building VLAD descriptors ---------")
    db_vlads, query_vlad = build_vlads(largs, ds)

    # If saving (for internal debugging only)
    if largs.save_vlad_descs is not None:
        print("\n------ Saving VLAD descriptors ------")
        print(f"DB VLAD shape: {db_vlads.shape}")
        print(f"QU VLAD shape: {query_vlad.shape}")
        save_dir = largs.save_vlad_descs
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")
        else:
            print(f"Save directory already exists: {save_dir}")
        
        # Save files if they don't already exist
        db_file = f"{save_dir}/db-{ds_name}.pt"
        qu_file = f"{save_dir}/qu-{ds_name}.pt"
        if os.path.exists(db_file):
            print(f"Database VLADs file already exists at {db_file}, skipping save")
        else:
            torch.save(db_vlads.cpu(), db_file)
            print(f"Saved database VLADs to {db_file}")
        if os.path.exists(qu_file):
            print(f"Query VLAD file already exists at {qu_file}, skipping save")
        else:
            torch.save(query_vlad.cpu(), qu_file)
            print(f"Saved query VLAD to {qu_file}")

    print("\n--------- Finding similar images ---------")

    # Find most similar images
    distances, indices = get_top_k_similarities(
        largs.top_k_vals,
        db=db_vlads,
        qu=query_vlad,
        method="cosine"
    )
    print("------------ Similarities calculated ------------")

    # Top match
    best_match_idx = indices[0][0]
    best_match_distance = distances[0][0]

    print(f"Best match index: {best_match_idx}, distance: {best_match_distance}")
    print(f"Indices: {indices}, Distances: {distances}")

    # Show the query image and the top 4 matches
    print("\n------------ Displaying query and top 4 matches -----------")
    # Extract image paths for the top 4 matches
    top_k = min(4, len(indices[0]))
    top_indices = [int(idx) for idx in indices[0][:top_k]]
    top_distances = distances[0][:top_k]
    top_img_paths = [ds.get_image_paths()[idx] for idx in top_indices]
    top_match_imgs = [Image.open(path).convert("RGB") for path in top_img_paths]
    query_img = Image.open(query_img_path).convert("RGB")

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1], wspace=0.12, hspace=0.25)
    ax_query = fig.add_subplot(gs[:, 0])
    ax_query.imshow(query_img)
    ax_query.set_title("Query Image", fontsize=16)
    ax_query.axis("off")

    match_axes = [fig.add_subplot(gs[i // 2, 1 + (i % 2)]) for i in range(top_k)]
    for i, ax in enumerate(match_axes):
        ax.imshow(top_match_imgs[i])
        ax.set_title(f"Rank {i+1} (Idx: {top_indices[i]}, Dist: {top_distances[i]:.4f})", fontsize=12)
        ax.axis("off")

    # Hide any unused axes if fewer than 4 matches are available
    for ax in match_axes[top_k:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show(block=False)

    print("\n------------  Top-K matches Coordinates ------------")
    # Get coordinates for top-K matches
    coords = ds.get_coordinates(indices[0])
    print(f"Coordinates for top-K matches: {coords}")

    print("\n------------  Clustering coordinates of top-K matches ------------")
    # Cluster the coordinates of the top-K matches (if available)
    valid_coords = [c for c in coords if c is not None]
    if len(valid_coords) > 0:
        # The samples are placed with 60m distance, so we can use a DBSCAN eps of around 62 to cluster them
        clusters = DBSCAN(eps=62, min_samples=3).fit(valid_coords)
        cluster_labels = clusters.labels_
        print(f"DBSCAN cluster labels for top-K matches: {cluster_labels}")
        # Identify core samples in the clusters
        core_samples_mask = np.zeros_like(cluster_labels, dtype=bool)
        core_samples_mask[clusters.core_sample_indices_] = True
        # Extract bigger cluster
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        print(f"Unique cluster labels: {unique_labels}, counts: {counts}")
        if len(unique_labels) > 0:
            biggest_cluster_label = unique_labels[np.argmax(counts)]
            print(f"Biggest cluster label: {biggest_cluster_label}")
            biggest_cluster_coords = [valid_coords[i] for i in range(len(valid_coords)) if cluster_labels[i] == biggest_cluster_label]
            print(f"Coordinates in the biggest cluster: {biggest_cluster_coords}")

            # Plot the clustered coordinates showing the index of the top-K match in the plot
            fig2, ax = plt.subplots(figsize=(6, 6))
            marker_size = [120 if core_samples_mask[i] else 30 for i in range(len(valid_coords))]
            scatter = ax.scatter(*zip(*valid_coords), c=cluster_labels, cmap='tab10', s=marker_size, label='Top-K Matches')
            for i, (x, y) in enumerate(valid_coords):
                ax.text(x, y, str(int(indices[0][i])), fontsize=11, ha='right')
            ax.set_title("DBSCAN Clustering of Top-K Matches Coordinates")
            ax.set_xlabel("Easting")
            ax.set_ylabel("Northing")
            ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.grid()
            plt.show()
            plt.close()
        else:
            print("No clusters found in DBSCAN")
    else:
        print("No valid coordinates available for clustering")

if __name__ == "__main__" and ("ipykernel" not in sys.argv[0]):
    largs = tyro.cli(LocalArgs, description=__doc__)
    _start = time.time()
    try:
        main(largs)
    except:
        print("Unhandled exception")
        traceback.print_exc()
    finally:
        print(f"Program ended in {time.time()-_start:.3f} seconds")
        exit(0)