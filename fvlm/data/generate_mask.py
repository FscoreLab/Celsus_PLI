import os
from pathlib import Path

import nibabel as nib
import pandas as pd
from totalsegmentator.python_api import totalsegmentator


def generate_mask(row):
    VolumeName = row["VolumeName"]
    dir1 = VolumeName.rsplit("_", 1)[0]
    dir2 = VolumeName.rsplit("_", 2)[0]
    filepath = os.path.join(data_root, f"{split}_fixed", dir2, dir1, VolumeName)

    dirpath = os.path.dirname(filepath)
    dirpath = dirpath.replace(f"/{split}_fixed/", f"/{split}_mask/")
    
    input_img = nib.load(filepath)
    output_img = totalsegmentator(input_img, quiet=True)

    Path(dirpath).mkdir(parents=True, exist_ok=True)
    nib.save(output_img, os.path.join(dirpath, os.path.basename(filepath)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--split", required=False, default='train', type=str)
    args = parser.parse_args()
    
    split = args.split
    d = "validation" if split == "valid" else "train"
    data_root = Path("/scratch/railabs/ld258/dataset/PUBLIC/CT_RATE/dataset/")
    metadata = pd.read_csv(os.path.join(data_root, f"metadata/{d}_metadata.csv"))
    rows = [row[1] for row in metadata.iterrows()]
    for row in rows:
        generate_mask(row)
