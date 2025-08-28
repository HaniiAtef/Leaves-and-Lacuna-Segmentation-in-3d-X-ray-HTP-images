import os
import csv
import tifffile as tiff
import numpy as np
from pathlib import Path

def extract_cube(image, center_z, center_y, center_x, size=128):
    half = size // 2
    z1, z2 = center_z - half, center_z + half
    y1, y2 = center_y - half, center_y + half
    x1, x2 = center_x - half, center_x + half

    pad_z = max(0, -z1), max(0, z2 - image.shape[0])
    pad_y = max(0, -y1), max(0, y2 - image.shape[1])
    pad_x = max(0, -x1), max(0, x2 - image.shape[2])

    z1 = max(0, z1)
    z2 = min(image.shape[0], z2)
    y1 = max(0, y1)
    y2 = min(image.shape[1], y2)
    x1 = max(0, x1)
    x2 = min(image.shape[2], x2)

    crop = image[z1:z2, y1:y2, x1:x2]
    if any(pad_z + pad_y + pad_x):
        crop = np.pad(crop,
                      (pad_z, pad_y, pad_x),
                      mode='constant', constant_values=0)
    return crop

def process_cube(cube_dir, central_output_dir):
    cube_dir = Path(cube_dir)
    # cube_id = cube_dir.name.replace("_", "")  # e.g. Cube_01 → Cube01
    cube_id = cube_dir.name  # e.g. Cube_01 → Cube01
    
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

    for (x, y, z, label) in annotations:
        cube = extract_cube(image, z, y, x)
        
        filename = f"{cube_id}_{label}.tif"
        local_path = patch_folder / filename
        central_path = central_output_dir / filename
        
        tiff.imwrite(local_path, cube.astype(image.dtype))
        tiff.imwrite(central_path, cube.astype(image.dtype))

    print(f"✅ {cube_id}: Saved {len(annotations)} patches to: {patch_folder} and {central_output_dir}")

def main(cubes_root_dir):
    cubes_root_dir = Path(cubes_root_dir)
    assert cubes_root_dir.exists(), f"Root directory does not exist: {cubes_root_dir}"
    
    cube_dirs = sorted([d for d in cubes_root_dir.iterdir() if d.is_dir() and d.name.startswith("Cube_")])
    if not cube_dirs:
        print(f"⚠️ No cube directories found in: {cubes_root_dir}")
        return

    # Central folder to collect all patches
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
