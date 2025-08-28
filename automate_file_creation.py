import os
import shutil
from pathlib import Path

def create_cube_structure(source_dir):
    source_dir = Path(source_dir)
    cubes_segmentations = source_dir.parent / "Cubes_segmentations"
    
    # Create main directory if it doesn't exist
    cubes_segmentations.mkdir(exist_ok=True)
    
    # Find all cube TIF files in the source directory
    tif_files = list(source_dir.glob("Cube_*_crop.tif"))
    
    if not tif_files:
        print(f"⚠️ No cube TIF files found in: {source_dir}")
        return
    
    print(f"Found {len(tif_files)} cube TIF files to process")
    
    for tif_path in tif_files:
        try:
            # Extract cube number (e.g., "01" from "Cube_01_crop.tif")
            cube_num = tif_path.stem.split('_')[1]
            cube_name = f"Cube_{cube_num}"
            
            # Create cube directory structure
            cube_dir = cubes_segmentations / cube_name
            crop_dir = cube_dir / "CropForHTP"
            crop_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy the TIF file
            dest_path = crop_dir / tif_path.name
            shutil.copy2(tif_path, dest_path)
            
            print(f"✅ Created {cube_name} structure with copied TIF file")
            
        except Exception as e:
            print(f"❌ Error processing {tif_path.name}: {str(e)}")

if __name__ == "__main__":
    crop_for_htp_path = "/home/hmohamed/Documents/CUBES_segmentation/CropForHTP"
    create_cube_structure(crop_for_htp_path)