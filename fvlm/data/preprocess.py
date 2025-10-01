import torch
from monai import transforms
from pathlib import Path
import numpy as np
import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_image(loader, mask_path):
    try:
        phase = 'train' if 'train' in mask_path else 'valid'

        mask_path = mask_path.replace(f"{phase}_mask", f"resized_{phase}_masks")
        img_path = mask_path.replace("masks", "images")

        if (
            Path(
                img_path.replace(f"resized_{phase}_images", f"processed_{phase}_images")
            ).exists()
            and Path(
                mask_path.replace(f"resized_{phase}_masks", f"processed_{phase}_masks")
            ).exists()
        ):
            return None

        data = loader({"image": img_path, "label": mask_path})

        image = data["image"]
        label = data["label"]
        
        old_unique_organ_ids = label.unique()

        roi_coords = np.nonzero(label[0])
        min_dhw = torch.from_numpy(np.min(roi_coords, axis=1))
        max_dhw = torch.from_numpy(np.max(roi_coords, axis=1))

        extend_d = 5
        extend_hw = 20

        min_dhw = torch.maximum(
            min_dhw - torch.tensor([extend_d, extend_hw, extend_hw]),
            torch.tensor([0, 0, 0]),
        )

        max_dhw = torch.minimum(
            max_dhw + torch.tensor([extend_d, extend_hw, extend_hw]),
            torch.tensor([image.shape[1], image.shape[2], image.shape[3]]),
        )

        data["image"] = image[
            :, min_dhw[0] : max_dhw[0], min_dhw[1] : max_dhw[1], min_dhw[2] : max_dhw[2]
        ]
        data["label"] = label[
            :, min_dhw[0] : max_dhw[0], min_dhw[1] : max_dhw[1], min_dhw[2] : max_dhw[2]
        ]

        new_unique_organ_ids = data["label"].unique()

        assert torch.all(old_unique_organ_ids == new_unique_organ_ids)

        saver = transforms.Compose(
            [
                transforms.SpatialPadd(
                    keys=["image"],
                    spatial_size=(112, 256, 352),
                    mode="constant",
                    constant_values=0
                ),
                transforms.SpatialPadd(
                    keys=["label"],
                    spatial_size=(112, 256, 352),
                    mode="constant",
                    constant_values=0
                ),
                transforms.SaveImaged(
                    output_dir=str(
                        Path(
                            img_path.replace(
                                f"resized_{phase}_images", f"processed_{phase}_images")
                        ).parent
                    ),
                    keys=["image"],
                    output_postfix="",
                    separate_folder=False,
                    resample=False,
                ),
                transforms.SaveImaged(
                    output_dir=str(
                        Path(
                            mask_path.replace(
                                f"resized_{phase}_masks", f"processed_{phase}_masks")
                        ).parent
                    ),
                    keys=["label"],
                    output_postfix="",
                    separate_folder=False,
                    resample=False,
                ),
            ]
        )

        saver(data)
    
    except Exception as e:
        print(e, img_path)
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", required=False, default='train', type='str')
    args = parser.parse_args()
    
    split = args.split

    image_root = f"{split}_fix"
    mask_root = f"{split}_mask"

    mask_paths = []
    for root, _, files in os.walk(mask_root):
        for file in files:
            mask_paths.append(os.path.join(root, file))

    loader = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], image_only=True, ensure_channel_first=True),
            transforms.Transposed(keys=["image", "label"], indices=(0, 3, 2, 1)),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-1150, a_max=350,
                b_min=0.0, b_max=1.0, clip=True
            )
        ])

    max_workers = 64
    func = partial(process_image, loader)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in tqdm(executor.map(func, mask_paths), total=len(mask_paths)):
            pass
    