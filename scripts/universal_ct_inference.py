#!/usr/bin/env python3
"""
Universal CT-CLIP Inference Function

This script provides a unified interface for CT-CLIP inference on both:
- ZIP archives containing DICOM files
- Individual NIFTI (.nii.gz) files

Returns predictions as JSON format.
"""

import json
import os
import shutil
import sys
import tempfile
import zipfile
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))


def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

import dicom2nifti

from CT_CLIP.ct_clip.ct_clip import CTCLIP

# Import CT-CLIP components
from transformer_maskgit.transformer_maskgit.ctvit import CTViT


def apply_softmax(array):
    """
    Applies softmax function to a torch array.

    Args:
        array (torch.Tensor): Input tensor array.

    Returns:
        torch.Tensor: Tensor array after applying softmax.
    """
    softmax = torch.nn.Softmax(dim=0)
    softmax_array = softmax(array)
    return softmax_array


class UniversalCTInference:
    """Universal CT-CLIP inference for DICOM archives and NIFTI files."""

    def __init__(self, model=None, model_path=None, device=None, verbose=False):
        """Initialize the inference pipeline."""
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.verbose = verbose

        # Initialize model components
        self.tokenizer = None
        self.model = model  # Can be pre-loaded model or None

        # DICOM processing components (lazy loaded)
        self.dicom_extractor = None
        self.dicom_converter = None

        # Default pathologies from CT-CLIP paper
        self.pathologies = [
            "Medical material",
            "Arterial wall calcification",
            "Cardiomegaly",
            "Pericardial effusion",
            "Coronary artery wall calcification",
            "Hiatal hernia",
            "Lymphadenopathy",
            "Emphysema",
            "Atelectasis",
            "Lung nodule",
            "Lung opacity",
            "Pulmonary fibrotic sequela",
            "Pleural effusion",
            "Mosaic attenuation pattern",
            "Peribronchial thickening",
            "Consolidation",
            "Bronchiectasis",
            "Interlobular septal thickening",
        ]

        if self.verbose:
            print(f"Initialized Universal CT Inference on device: {self.device}")

    def load_model(self):
        """Load CT-CLIP model components."""
        if self.model is not None and self.tokenizer is not None:
            return  # Already loaded

        if self.verbose:
            print("Loading CT-CLIP model...")

        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True)

        # If model is already provided, just move to device and set eval mode
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
            if self.verbose:
                print("Using pre-loaded model!")
            return

        # Otherwise load from file
        if self.model_path is None:
            raise ValueError("Either model or model_path must be provided")

        # Initialize text encoder
        text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
        text_encoder.resize_token_embeddings(len(self.tokenizer))

        # Initialize image encoder
        image_encoder = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=20,
            temporal_patch_size=10,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8,
        )

        # Initialize CT-CLIP model
        self.model = CTCLIP(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            dim_image=294912,
            dim_text=768,
            dim_latent=512,
            extra_latent_projection=False,
            use_mlm=False,
            downsample_image_embeds=False,
            use_all_token_embeds=False,
        )

        # Load pretrained weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

        self.model.to(self.device)
        self.model.eval()

        if self.verbose:
            print("Model loaded successfully!")

    def _init_dicom_components(self):
        """Lazy initialization of DICOM processing components."""
        # DICOM converter no longer needed - using dicom2nifti library
        pass

    def nii_to_tensor(self, nii_file, spacing=None, min_slices=20):
        """Convert NIFTI file to tensor in CT-CLIP format following original pipeline.
        
        Args:
            nii_file: Path to NIFTI file
            spacing: Optional spacing override (will use real spacing from header if None)
            min_slices: Minimum number of slices required
        """
        if self.verbose:
            print(f"Processing NIFTI file: {nii_file}")
            
        # Load NIFTI file
        img = nib.load(nii_file)
        img_data = img.get_fdata()
        
        # Read real spacing from NIFTI header
        # header.get_zooms() returns (z, x, y), but we need (z, xy, xy) for processing
        zooms = img.header.get_zooms()
        real_spacing = (zooms[0], zooms[1], zooms[2])  # (z, x, y)
        
        if self.verbose:
            print(f"   Loaded shape: {img_data.shape}")
            print(f"   Raw range: {img_data.min():.1f} to {img_data.max():.1f}")
            print(f"   Real spacing from header: {real_spacing}")

        # STEP 1: NO RescaleSlope/Intercept - NIFTI already contains HU values!
        # (Original code incorrectly applied them twice)
        
        # STEP 2: Apply HU windowing (like original)
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)
        
        if self.verbose:
            print(f"   After HU windowing: {img_data.min():.1f} to {img_data.max():.1f}")
        
        # STEP 3: Transpose to match original format [H,W,D] -> [D,H,W]
        img_data = img_data.transpose(2, 0, 1)
        
        if self.verbose:
            print(f"   After transpose: {img_data.shape}")
        
        # STEP 4: Convert to tensor and apply spacing-based resize (like original)
        tensor = torch.tensor(img_data, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims: [1, 1, D, H, W]
        
        # ВАЖНО: После transpose данные стали [D,H,W], поэтому spacing тоже нужно transposed!
        # real_spacing = (H_sp, W_sp, D_sp), после transpose нужен (D_sp, H_sp, W_sp)
        if spacing is not None:
            current_spacing = spacing
        else:
            # Transpose spacing to match transposed data: (H,W,D) -> (D,H,W)
            current_spacing = (real_spacing[2], real_spacing[0], real_spacing[1])
        
        target_spacing = (1.5, 0.75, 0.75)  # Original target spacing (D, H, W)
        
        if self.verbose:
            if spacing is not None:
                print(f"   Using override spacing: {current_spacing}")
            else:
                print(f"   Using real header spacing: {current_spacing}")
            print(f"   Target spacing: {target_spacing}")
        
        # Apply spacing-based resize using the original function
        img_data = resize_array(tensor, current_spacing, target_spacing)
        img_data = img_data[0][0]  # Remove batch and channel dims
        img_data = np.transpose(img_data, (1, 2, 0))  # [H, W, D]
        
        if self.verbose:
            print(f"   After spacing resize: {img_data.shape}")
        
        # Check minimum slices
        if img_data.shape[-1] < min_slices:
            raise ValueError(f"Image has {img_data.shape[-1]} slices, minimum required: {min_slices}")
        
        # STEP 5: Apply original normalization (divide by 1000, not z-score)
        img_data = (img_data / 1000).astype(np.float32)
        
        if self.verbose:
            print(f"   After normalization: {img_data.min():.4f} to {img_data.max():.4f}")
        
        # STEP 6: Convert back to tensor and apply original crop/pad logic
        tensor = torch.tensor(img_data)
        
        # Get the dimensions of the input tensor
        target_final_shape = (480, 480, 240)  # H, W, D
        h, w, d = tensor.shape
        
        # Calculate cropping/padding values for height, width, and depth (like original)
        dh, dw, dd = target_final_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)
        
        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]
        
        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before
        
        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before
        
        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before
        
        # Apply padding with value=-1 (like original)
        tensor = torch.nn.functional.pad(tensor, 
                                       (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), 
                                       value=-1)
        
        # STEP 7: Final permutation to match original format [H, W, D] -> [D, H, W] -> [1, D, H, W]
        tensor = tensor.permute(2, 0, 1)  # [H, W, D] -> [D, H, W]
        final_tensor = tensor.unsqueeze(0)  # Add batch dim: [1, D, H, W]
        
        if self.verbose:
            print(f"   Final tensor shape: {final_tensor.shape}")
            print(f"   Final range: {final_tensor.min():.4f} to {final_tensor.max():.4f}")
        
        return final_tensor

    def process_dicom_archive(self, zip_path):
        """Process DICOM ZIP archive and return tensor using dicom2nifti."""
        if self.verbose:
            print(f"Processing DICOM archive: {zip_path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directories
            extract_dir = os.path.join(temp_dir, "dicom_extract")
            nifti_dir = os.path.join(temp_dir, "nifti")
            os.makedirs(extract_dir, exist_ok=True)
            os.makedirs(nifti_dir, exist_ok=True)

            # Extract ZIP archive
            if self.verbose:
                print("Extracting DICOM archive...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            if self.verbose:
                print("Converting DICOM to NIFTI using dicom2nifti library...")

            try:
                # Use dicom2nifti with reorient=False for perfect match with original NIFTI
                dicom2nifti.convert_directory(
                    extract_dir,
                    nifti_dir,
                    compression=True,
                    reorient=False,  # CRITICAL: This ensures perfect match with original NIFTI
                )

                # Find created NIFTI file
                created_files = [f for f in os.listdir(nifti_dir) if f.endswith(".nii.gz")]

                if not created_files:
                    raise ValueError("No NIFTI files created by dicom2nifti")

                # Use first created file
                nifti_file = os.path.join(nifti_dir, created_files[0])
                if self.verbose:
                    print(f"✅ Successfully converted to: {nifti_file}")

            except Exception as e:
                if self.verbose:
                    print(f"❌ dicom2nifti conversion failed: {e}")
                    print("Trying alternative method: dicom_series_to_nifti...")

                # Fallback to series conversion
                study_id = Path(zip_path).stem
                nifti_file = os.path.join(nifti_dir, f"{study_id}.nii.gz")

                dicom2nifti.dicom_series_to_nifti(
                    extract_dir, nifti_file, reorient_nifti=False  # CRITICAL: This ensures perfect match
                )
                if self.verbose:
                    print(f"✅ Alternative conversion successful: {nifti_file}")

            # Load the NIFTI file immediately and return the tensor
            # since temporary files will be cleaned up
            tensor = self.nii_to_tensor(nifti_file)
            return tensor

    def predict_pathologies(self, tensor, custom_pathologies=None):
        """
        Run pathology predictions on tensor.
        
        Args:
            tensor: Input tensor
            custom_pathologies: Custom list of pathologies to test
        """
        predictions = {}
        
        # Use custom pathologies if provided, otherwise use default
        pathologies_to_use = custom_pathologies if custom_pathologies is not None else self.pathologies

        with torch.no_grad():
            for pathology in pathologies_to_use:
                # Create text prompts
                texts = [f"{pathology} is present.", f"{pathology} is not present."]
                text_tokens = self.tokenizer(
                    texts, return_tensors="pt", padding="max_length", truncation=True, max_length=512
                ).to(self.device)

                # Add batch dimension and move to device
                input_tensor = tensor.unsqueeze(0).to(self.device)  # [1, 1, D, H, W]

                # Run inference
                output = self.model(text_tokens, input_tensor, device=self.device)

                # Apply softmax to get probabilities
                probs = apply_softmax(output.cpu())

                present_prob = float(probs[0])

                    
                predictions[pathology] = present_prob

        return predictions

    def infer(self, input_path, custom_pathologies=None):
        """
        Main inference function.

        Args:
            input_path (str): Path to ZIP archive (DICOM) or .nii.gz file
            custom_pathologies (list, optional): Custom list of pathologies to test

        Returns:
            dict: JSON-serializable dictionary with predictions
        """
        # Load model if not already loaded
        self.load_model()

        input_path = Path(input_path)

        # Determine input type and process accordingly
        if input_path.suffix.lower() == ".zip":
            if self.verbose:
                print(f"Processing DICOM archive: {input_path}")
            # Process DICOM archive (returns tensor directly)
            tensor = self.process_dicom_archive(str(input_path))
            source_type = "dicom_archive"

        elif input_path.suffix.lower() == ".gz" and input_path.name.endswith(".nii.gz"):
            if self.verbose:
                print(f"Processing NIFTI file: {input_path}")
            # Process NIFTI file directly
            tensor = self.nii_to_tensor(str(input_path))
            source_type = "nifti"

        else:
            raise ValueError(
                f"Unsupported file format: {input_path.suffix}. "
                "Supported formats: .zip (DICOM archive), .nii.gz (NIFTI)"
            )

        # Run predictions
        if self.verbose:
            print("Running pathology predictions...")
        predictions = self.predict_pathologies(tensor, custom_pathologies)

        # Prepare result
        result = {
            "input_file": str(input_path),
            "source_type": source_type,
            "tensor_shape": list(tensor.shape),
            "pathology_predictions": predictions,
            "metadata": {
                "model_path": self.model_path,
                "device": str(self.device),
                "num_pathologies": len(self.pathologies),
            },
        }

        if self.verbose:
            print(f"Inference completed! Found {len(predictions)} pathology predictions.")
        return result


def ct_clip_inference(input_path, model_path, device=None, custom_pathologies=None):
    """
    Convenient wrapper function for CT-CLIP inference.

    Args:
        input_path (str): Path to ZIP archive (DICOM) or .nii.gz file
        model_path (str): Path to CT-CLIP model checkpoint
        device (str, optional): Device to use ('cuda', 'cpu', or None for auto)
        custom_pathologies (list, optional): Custom list of pathologies to test

    Returns:
        dict: JSON-serializable dictionary with predictions
    """
    if device is not None:
        device = torch.device(device)

    inferencer = UniversalCTInference(model_path, device)
    return inferencer.infer(input_path, custom_pathologies)


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Universal CT-CLIP Inference")
    parser.add_argument("input_path", help="Path to ZIP archive (DICOM) or .nii.gz file")
    parser.add_argument("--model", required=True, help="Path to CT-CLIP model checkpoint")
    parser.add_argument("--output", help="Output JSON file (default: print to stdout)")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device to use")

    args = parser.parse_args()

    try:
        result = ct_clip_inference(args.input_path, args.model, args.device)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to: {args.output}")
        else:
            print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
