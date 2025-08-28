import os
import csv
import tifffile as tiff
import numpy as np
from pathlib import Path

def extract_sparse_cube(image, center_z, center_y, center_x, size=128):
    half = size // 2
    y1, y2 = center_y - half, center_y + half
    x1, x2 = center_x - half, center_x + half

    # Check Y/X bounds
    if y1 < 0 or y2 > image.shape[1] or x1 < 0 or x2 > image.shape[2]:
        return None  # out of bounds in Y/X

    # Compute 11 sparse Z indices spaced by 3
    slice_indices = sorted(set([center_z + i*3 for i in range(-5, 6)]))

    # Check Z bounds
    if slice_indices[0] < 0 or slice_indices[-1] >= image.shape[0]:
        return None  # out of bounds in Z

    # Extract 11 slices
    slices = [image[z, y1:y2, x1:x2] for z in slice_indices]

    # Stack into (11, 128, 128)
    cube = np.stack(slices, axis=0)
    return cube

def process_cube(cube_dir, central_output_dir):
    cube_dir = Path(cube_dir)
    cube_id = cube_dir.name
    
    crop_dir = cube_dir / "CropForHTP"
    tif_files = list(crop_dir.glob(f"*crop.tif"))
    if not tif_files:
        print(f"⚠️ No TIF file found for {cube_id}")
        return

    tif_path = tif_files[0]
    image = tiff.imread(tif_path)
    assert image.ndim == 3, "Expected 3D TIF image"

    csv_path = crop_dir / f"{cube_dir.name}_crop_labels.csv"
    if not csv_path.exists():
        print(f"⚠️ CSV not found for {cube_id}: {csv_path}")
        return

    annotations = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x, y, z = int(row['x']), int(row['y']), int(row['z'])
            label = row['label']
            annotations.append((x, y, z, label))

    predictions_folder = cube_dir / "Predictions"
    patch_folder = predictions_folder / f"{cube_id}_crop_patches"
    patch_folder.mkdir(parents=True, exist_ok=True)
    central_output_dir.mkdir(exist_ok=True)

    saved_count = 0
    for (x, y, z, label) in annotations:
        cube = extract_sparse_cube(image, z, y, x)
        if cube is None:
            print(f"⚠️ Skipped annotation at ({x}, {y}, {z}) in {cube_id} — out of bounds")
            continue
        
        filename = f"{cube_id}_{label}.tif"
        local_path = patch_folder / filename
        central_path = central_output_dir / filename
        
        tiff.imwrite(local_path, cube.astype(image.dtype))
        tiff.imwrite(central_path, cube.astype(image.dtype))
        saved_count += 1

    print(f"✅ {cube_id}: Saved {saved_count} valid patches to: {patch_folder} and {central_output_dir}")

def main(cubes_root_dir):
    cubes_root_dir = Path(cubes_root_dir)
    assert cubes_root_dir.exists(), f"Root directory does not exist: {cubes_root_dir}"
    
    cube_dirs = sorted([d for d in cubes_root_dir.iterdir() if d.is_dir() and d.name.startswith("Cube_")])
    if not cube_dirs:
        print(f"⚠️ No cube directories found in: {cubes_root_dir}")
        return

    central_output_dir = cubes_root_dir / "All_Crops"

    print(f"Found {len(cube_dirs)} cube directories to process")
    
    for cube_dir in cube_dirs:
        try:
            process_cube(cube_dir, central_output_dir)
        except Exception as e:
            print(f"❌ Error processing {cube_dir.name}: {str(e)}")

if __name__ == "__main__":
    cubes_root_dir = "/home/hmohamed/Documents/CUBES_segmentation/patch_ext_test"
    main(cubes_root_dir)
