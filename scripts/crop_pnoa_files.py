"""
Script to crop PNOA TIFF files into smaller tiles based on a defined ROI and sampling grid.
The script reads a PNOA TIFF file, extracts a specified ROI, and divides it into smaller tiles of a given size, with a specified step size between them. 
The tiles are saved as PNG files, and a CSV file is generated containing the tile names and their corresponding center coordinates in geographic space.
"""

import os
import glob
import argparse
import csv
import numpy as np

import rasterio
from rasterio.plot import show
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from matplotlib import image, pyplot as plt
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Crop PNOA files to ROIs")
    parser.add_argument("--pnoa_file", type=str, required=True, help="Path to the PNOA TIFF file")
    parser.add_argument("--output_dir", type=str, default="tiles", help="Directory to save the cropped tiles")
    return parser.parse_args()

def main():
    args = parse_args()
    pnoa_dir = os.path.dirname(args.pnoa_file)
    img_file = os.path.basename(args.pnoa_file)
    

    if not os.path.isdir(pnoa_dir):
        raise FileNotFoundError(f"Directory not found: {pnoa_dir}")
    
    # If output_dir is default, create it inside the PNOA directory
    if args.output_dir == "tiles":
        img_file_without_ext = os.path.splitext(img_file)[0]
        output_path = os.path.join(pnoa_dir, f"{img_file_without_ext}_tiles")
    else:
        output_path = args.output_dir

    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, "tiles_locations.csv")

    # --- Open PNOA file ---
    img_src = rasterio.open(os.path.join(pnoa_dir, img_file))

    # --- Read all bands and plot original image ---
    bands = img_src.read()
    print("------ PNOA Image Information -----")
    print(f"Bands shape: {bands.shape}")  # Should be (bands, height, width)
    print(f"Data type: {bands.dtype}")

    nb,h,w = bands.shape
    orig_img = np.zeros([h, w, 3],dtype=bands.dtype)
    orig_img[:,:,0] = np.ones([h, w])*bands[0]
    orig_img[:,:,1] = np.ones([h, w])*bands[1]
    orig_img[:,:,2] = np.ones([h, w])*bands[2]


    fig1 = plt.figure(figsize=(9,7)) # figsize=(9,7)
    plt.imshow(orig_img)
    plt.title("Plotting original img from " + img_file)
    plt.margins(0,0)

    # --- Extract metadata ---
    # print(f"Image CRS: {img_src.crs}")
    # print(f"Image bounds: {img_src.bounds}")
    width = img_src.width
    height = img_src.height
    print(f"Image width: {width}, height: {height}")
    # print(f"Image count (bands): {img_src.count}")
    # print(f"Metadata: {img_src.meta}")
    src_transform = img_src.transform
    print(f"Image transform: {src_transform}")
    res_x = src_transform.a
    res_y = -src_transform.e
    print(f"Image resolution: {img_src.res}")

    # --- Manage tile and step sizes variables ---
    # Desired output tile size
    SAMPLE_WIDTH = 420 # [pixels]
    SAMPLE_HEIGHT = 420 # [pixels]
    half_x = (SAMPLE_WIDTH // 2)
    half_y = (SAMPLE_HEIGHT // 2)

    # Desired step size between sampled tiles
    STEP_M = 60 # [meters]
    step_px_x = int(round(STEP_M / res_x)) # [pixels]
    step_px_y = int(round(STEP_M / res_y)) # [pixels]

    # FOV and overlap calculation
    fov_m = SAMPLE_WIDTH * res_x  # Field of view in meters for the tile width
    overlap_perc = ((fov_m - STEP_M) / fov_m) * 100  # Overlap percentage between tiles

    print("\n------ Tile Parameters ------")
    print(f"Tile size in pixels: ({SAMPLE_WIDTH}, {SAMPLE_HEIGHT})")
    print(f"Step size in pixels: ({step_px_x}, {step_px_y})")
    print(f"Overlap percentage: {overlap_perc:.2f}%")

    # ---- Define ROI ----
    ROI_COL_OFF = 3000
    ROI_ROW_OFF = 14500
    ROI_WIDTH = 2500
    ROI_HEIGHT = 3500

    print("\n------ ROI Parameters ------")
    print(f"ROI offset (pixels): ({ROI_COL_OFF}, {ROI_ROW_OFF})")
    print(f"ROI size (pixels): ({ROI_WIDTH}, {ROI_HEIGHT})")

    # Create a window for the ROI
    roi_window = Window(ROI_COL_OFF, ROI_ROW_OFF, ROI_WIDTH, ROI_HEIGHT)

    # Read the ROI from the image using the window
    roi = img_src.read(window=roi_window)
    
    # Update transform for the ROI
    roi_transform = window_transform(roi_window, img_src.transform)
    print(f"ROI transform: {roi_transform}")

    # Compute center coordinates of the ROI
    center_col = width / 2
    center_row = height / 2
    center_x, center_y = roi_transform * (center_col, center_row)
    print(f"ROI Center (geographic): ({center_x}, {center_y})")

    # ---- RGB visualization of ROI ----
    roi_nb, roi_h, roi_w = roi.shape
    ROI_rgb = np.zeros([roi_h, roi_w, 3], dtype=roi.dtype)
    ROI_rgb[:, :, 0] = np.ones([roi_h, roi_w])*roi[0]
    ROI_rgb[:, :, 1] = np.ones([roi_h, roi_w])*roi[1]
    ROI_rgb[:, :, 2] = np.ones([roi_h, roi_w])*roi[2]

    # Save ROI
    roi_img_pil = Image.fromarray(ROI_rgb)
    roi_img_path = os.path.join(output_path, "ROI.png")
    roi_img_pil.save(roi_img_path)

    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.imshow(ROI_rgb)
    ax2.set_title("Cropped ROI with sampling grid")

    # Compute grid center pixels for the tiles, taking into account the tile size, step size and border conditions
    centers = []
    for row in range(half_y, roi_h - half_y + 1, step_px_y):
        for col in range(half_x, roi_w - half_x + 1, step_px_x):
            centers.append((col, row))

    # Plot ALL centers (red)
    center_cols = [c[0] for c in centers]
    center_rows = [c[1] for c in centers]
    scatter_all = ax2.scatter(center_cols, center_rows, c='red', s=10)

    # Current point (green) - will be updated in the loop
    scatter_current = ax2.scatter([], [], c='green', s=40)

    plt.ion()  # interactive mode

    # --- Divide the ROI in tiles ---
    fig3, ax3 = plt.subplots(figsize=(4, 4))
    print("\n------ Starting tile cropping and saving -----")
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['tile_name', 'center_x', 'center_y'])  # Write header

        tile_count = 0
        for row in range(half_y, roi_h - half_y + 1, step_px_y):
            for col in range(half_x, roi_w - half_x + 1, step_px_x):
                # Update current marker on the grid
                scatter_current.set_offsets([[col, row]])

                tile_window = Window(col-half_x, row-half_y, SAMPLE_WIDTH, SAMPLE_HEIGHT)
                tile_transform = window_transform(tile_window, roi_transform)

                # Compute center coordinates of the tile
                tile_center_x, tile_center_y = tile_transform * (col, row)

                # Read the tile data
                tile_data = roi[:, row-half_y:row+half_y, col-half_x:col+half_x]

                nb,h,w = tile_data.shape
                tile_img = np.zeros([h, w, 3],dtype=tile_data.dtype)
                tile_img[:,:,0] = np.ones([h, w])*tile_data[0]
                tile_img[:,:,1] = np.ones([h, w])*tile_data[1]
                tile_img[:,:,2] = np.ones([h, w])*tile_data[2]

                # Visualize the tile
                ax3.clear()
                ax3.imshow(tile_img)
                ax3.set_title(f"Tile {tile_count} - Center: ({tile_center_x:.2f}, {tile_center_y:.2f})")
                ax3.axis('off')

                # Refresh plots
                fig2.canvas.draw()
                fig3.canvas.draw()
                plt.pause(0.2)  # Pause to visualize the tile

                # Save the tile as a png file
                tile_filename = f"{tile_count:04d}.png"
                tile_path = os.path.join(output_path, tile_filename)
                tile_img_pil = Image.fromarray(tile_img)
                tile_img_pil.save(tile_path)

                # Write the tile name and center coordinates to the CSV file
                csv_writer.writerow([tile_filename, tile_center_x, tile_center_y])
                
                print(f"Saved {tile_filename} with center coordinates ({tile_center_x}, {tile_center_y})")
                tile_count += 1

        print(f"\nTotal tiles created: {tile_count}")

    plt.close()

    # Save log with selected parameters
    log_path = os.path.join(output_path, "settings.txt")
    with open(log_path, mode='w') as log_file:
        log_file.write(f"PNOA file: {args.pnoa_file}\n")
        log_file.write(f"Image resolution (m/pix): ({res_x}, {res_y})\n")
        log_file.write(f"ROI pixel coordinates: (col_off: {ROI_COL_OFF}, row_off: {ROI_ROW_OFF}, width: {ROI_WIDTH}, height: {ROI_HEIGHT})\n")
        log_file.write(f"ROI center coordinates (geographic): ({center_x}, {center_y})\n")
        log_file.write(f"Tile size (pixels): ({SAMPLE_WIDTH}, {SAMPLE_HEIGHT})\n")
        log_file.write(f"Sampling distance (meters): {STEP_M}\n")
        log_file.write(f"Sampling distance (pixels): ({step_px_x}, {step_px_y})\n")
        log_file.write(f"Field of view (meters): {fov_m:.2f}\n")
        log_file.write(f"Overlap percentage: {overlap_perc:.2f}%\n")
        log_file.write(f"Total tiles created: {tile_count}\n")
    log_file.close()
    print(f"Saved settings log at: {log_path}")
        
if __name__ == "__main__":
    main()