import ast
import concurrent.futures
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import tqdm


def process_row(row):
    # Set up directory parameters
    VolumeName = row["VolumeName"]
    dir1 = VolumeName.rsplit("_", 1)[0]
    dir2 = VolumeName.rsplit("_", 2)[0]
    filepath = os.path.join(data_root, f"{split}", dir2, dir1, VolumeName)

    dirpath = os.path.dirname(filepath)
    dirpath = dirpath.replace(f"/{split}/", f"/{split}_fixed/")

    if os.path.exists(os.path.join(dirpath, os.path.basename(filepath))):
        return

    # Read Image
    image = sitk.ReadImage(filepath)

    # Set Spacing
    (x, y), z = map(float, ast.literal_eval(row["XYSpacing"])), row["ZSpacing"]
    image.SetSpacing((x, y, z))

    # Set Origin
    image.SetOrigin(ast.literal_eval(row["ImagePositionPatient"]))

    # Set Direction
    orientation = ast.literal_eval(row["ImageOrientationPatient"])
    row_cosine, col_cosine = orientation[:3], orientation[3:6]
    z_cosine = np.cross(row_cosine, col_cosine).tolist()
    image.SetDirection(row_cosine + col_cosine + z_cosine)

    # Fix Rescale
    RescaleIntercept = row["RescaleIntercept"]
    RescaleSlope = row["RescaleSlope"]
    adjusted_hu = image * RescaleSlope + RescaleIntercept

    # Convert the image to int16
    adjusted_hu = sitk.Cast(adjusted_hu, sitk.sitkInt16)

    # Write Image
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(adjusted_hu, os.path.join(dirpath, os.path.basename(filepath)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", required=False, default='train', type='str')
    args = parser.parse_args()
    
    split = args.split
    d = "validation" if split == "valid" else "train"
    
    data_root = Path("")
    metadata = pd.read_csv(os.path.join(data_root, f"metadata/{d}_metadata.csv"))
    rows = [row[1] for row in metadata.iterrows()]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm.tqdm(executor.map(process_row, rows), total=len(rows)))
