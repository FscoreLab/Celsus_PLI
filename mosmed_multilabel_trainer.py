"""
MosMed Multi-Label CT-CLIP Fine-tuning Trainer
Based on CTCLIPTrainer.py with improvements for sparse multi-label data
"""

import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import tqdm

# Accelerate for distributed training
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs
from sklearn.metrics import roc_auc_score
from transformers import BertModel, BertTokenizer

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

# Force transformers offline mode after first download
import os

# Project imports
from clearml import Task
from clearml.storage import StorageManager

from CT_CLIP.ct_clip.ct_clip import CTCLIP
from mosmed_dataset import create_data_loaders
from training_losses import MaskedWeightedBCE, make_class_weights
from transformer_maskgit.transformer_maskgit.ctvit import CTViT

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


class CosineWithWarmup:
    """Cosine annealing scheduler with linear warmup and minimum LR."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_scale=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_scale = min_lr_scale
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_count = 0

        # Start with lr=0
        for group in optimizer.param_groups:
            group["lr"] = 0.0

    def step(self):
        if self.step_count < self.warmup_steps:
            # Linear warmup
            scale = (self.step_count + 1) / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            scale = self.min_lr_scale + (1 - self.min_lr_scale) * 0.5 * (1 + np.cos(np.pi * progress))

        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * scale

        self.step_count += 1


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = base_lr * (step + 1) / warmup_length
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            param_group["lr"] = lr

    return _lr_adjuster


class SimpleCTCLIPClassifier(nn.Module):
    """Simple classification head for CT-CLIP fine-tuning."""

    def __init__(
        self,
        ct_clip_model,
        num_classes: int = 15,
        dropout: float = 0.3,
        latent_norm: str = "none",
        reinit_head: bool = False,
        head_init_std: float = 0.01,
    ):
        super().__init__()
        self.ct_clip = ct_clip_model
        self.num_classes = num_classes

        # Optional normalization before head
        self.latent_norm = (latent_norm or "none").lower()
        if self.latent_norm == "layernorm":
            self.pre_head_norm = nn.LayerNorm(512)
        else:
            self.pre_head_norm = nn.Identity()

        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))

        # Optional small reinitialization for stable logits
        if reinit_head:
            for m in self.classifier.modules():
                if isinstance(m, nn.Linear):
                    try:
                        nn.init.trunc_normal_(m.weight, std=head_init_std)
                    except Exception:
                        nn.init.normal_(m.weight, std=head_init_std)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Will be replaced in setup_data() with MaskedWeightedBCE; keep safe default
        self.loss_fn = nn.BCEWithLogitsLoss()
        # Tokenizer will be available after setup_model() via self.ct_clip.tokenizer
        self.tokenizer = None

    def freeze_encoder(self):
        """Freeze CT-CLIP encoder."""
        for param in self.ct_clip.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self, layers_to_unfreeze: Optional[List[str]] = None):
        """Unfreeze specific parts of CT-CLIP encoder (except text encoder)."""
        if layers_to_unfreeze is None:
            # Unfreeze all except text encoder
            for param in self.ct_clip.parameters():
                param.requires_grad = True
            for param in self.ct_clip.text_transformer.parameters():
                param.requires_grad = False
        else:
            # Unfreeze specific layers (for future use)
            pass

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Forward pass."""
        device = images.device

        # Get image embeddings from CT-CLIP
        # Create dummy text tokens (not used for classification)
        dummy_text = [""] * images.size(0)

        text_tokens = self.tokenizer(
            dummy_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512
        ).to(device)

        _, image_latents, _ = self.ct_clip(text_tokens, images, device=device, return_latents=True)

        # Apply optional normalization before head
        if self.latent_norm == "l2":
            image_latents = torch.nn.functional.normalize(image_latents, dim=1)
        else:
            image_latents = self.pre_head_norm(image_latents)

        logits = self.classifier(image_latents)

        result = {"logits": logits}

        if targets is not None:
            # Check if loss function expects sample_weights
            if hasattr(self.loss_fn, "forward") and "sample_weights" in self.loss_fn.forward.__code__.co_varnames:
                # SampleWeightedBCE case
                if sample_weights is not None:
                    loss = self.loss_fn(logits, targets, sample_weights)
                else:
                    # Fallback: equal weights for all samples
                    equal_weights = torch.ones_like(targets)
                    loss = self.loss_fn(logits, targets, equal_weights)
            else:
                # MaskedWeightedBCE case (original behavior)
                loss = self.loss_fn(logits, targets)
            result["loss"] = loss

        return result


class MosmedMultiLabelTrainer:
    """Main trainer class for MosMed multi-label CT-CLIP fine-tuning."""

    def __init__(
        self,
        model_path: str,
        csv_path: str,
        nifti_dir: str,
        target_pathologies: List[str],
        batch_size: int = 4,
        num_epochs: int = 25,
        lr_head: float = 1e-3,
        lr_encoder: float = 1e-5,
        weight_decay: float = 0.05,
        warmup_ratio: float = 0.1,
        clip_grad_norm: float = 5.0,
        results_folder: str = "./results",
        project_name: str = "Celsus_PLI",
        task_name: str = "CT-CLIP-MultiLabel",
        max_samples: Optional[int] = None,
        resume_from_checkpoint: Optional[str] = None,
        restart_from_weights: Optional[str] = None,
        enable_augmentations: bool = False,
        aug_flip_prob: float = 0.25,
        use_pseudo_labels: bool = False,
        pseudo_label_weight: float = 0.3,
        prefer_soft_pseudo: bool = True,
        latent_norm: str = "none",
        reinit_head: bool = False,
        head_init_std: float = 0.01,
    ):

        # Initialize accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, init_kwargs])

        # Store config
        self.model_path = model_path
        self.csv_path = csv_path
        self.nifti_dir = nifti_dir
        self.target_pathologies = target_pathologies
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr_head = lr_head
        self.lr_encoder = lr_encoder
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.clip_grad_norm = clip_grad_norm
        self.results_folder = Path(results_folder)
        self.max_samples = max_samples
        self.resume_from_checkpoint = resume_from_checkpoint  # NEW
        self.restart_from_weights = restart_from_weights  # NEW
        self.enable_augmentations = enable_augmentations
        self.aug_flip_prob = aug_flip_prob
        self.use_pseudo_labels = use_pseudo_labels
        self.pseudo_label_weight = pseudo_label_weight
        self.prefer_soft_pseudo = prefer_soft_pseudo
        self.latent_norm = latent_norm
        self.reinit_head = reinit_head
        self.head_init_std = head_init_std
        self._restart_head_only_done = False

        # Resume state variables
        self.start_epoch = 0
        self.best_auc = 0.0

        # Create results folder
        self.results_folder.mkdir(parents=True, exist_ok=True)

        # Initialize ClearML
        self.task = None
        self.logger = None
        if self.accelerator.is_main_process:
            self.task = Task.init(project_name=project_name, task_name=task_name)
            self.logger = self.task.get_logger()

        # Initialize model
        self.setup_model()

        # Initialize data
        self.setup_data()

        # Training stages
        self.stage = 0  # 0=head-only, 1=top2, 2=full

        # Initialize optimizer
        self.setup_optimizer()

        # Prepare with accelerator
        self.prepare_training()

        # Load checkpoint if resuming or restart from weights
        if self.resume_from_checkpoint:
            self.load_checkpoint()
        elif self.restart_from_weights:
            self.restart_from_weights_only()

        print(f"Trainer initialized on device: {self.accelerator.device}")
        print(f"Number of target pathologies: {len(target_pathologies)}")

    def _load_model_checkpoint(self, model_path: str) -> dict:
        """Load model checkpoint from local path or S3."""
        if model_path.startswith("s3://"):
            print(f"Downloading model from S3: {model_path}")
            local_path = StorageManager.get_local_copy(model_path, extract_archive=False, force_download=True)
            print(f"Model downloaded to: {local_path}")
        else:
            local_path = model_path

        print(f"Loading checkpoint from: {local_path}")
        return torch.load(local_path, map_location="cpu", weights_only=False)

    def load_checkpoint(self):
        """Load checkpoint for resuming training (simplified version)."""
        if not self.resume_from_checkpoint:
            return

        print(f"Loading checkpoint for resume: {self.resume_from_checkpoint}")
        checkpoint = self._load_model_checkpoint(self.resume_from_checkpoint)

        # 1. Load model state
        if "model_state_dict" in checkpoint:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print("✓ Model state loaded")

        # 2. Determine stage from epoch
        if "stage" in checkpoint:
            self.stage = checkpoint["stage"]
            print(f"✓ Training stage loaded: {self.stage}")
        else:
            # Backward compatibility: determine stage based on epoch
            if "epoch" in checkpoint:
                epoch = checkpoint["epoch"]
                if epoch > 20:  # Extended: Stage 2 starts after epoch 20
                    self.stage = 2  # Stage 2: full LLRD
                elif epoch >= 3:
                    self.stage = 1  # Stage 1: top-2 blocks (extended until epoch 20)
                else:
                    self.stage = 0  # Stage 0: head only
                print(f"✓ Training stage determined from epoch: {self.stage}")

        # 3. Recreate optimizer for correct stage (fresh state, correct LRs)
        print(f"🔄 Recreating optimizer for stage {self.stage} (fresh state)...")
        self._build_optimizer_for_stage()

        # Prepare optimizer with accelerator
        self.optimizer = self.accelerator.prepare(self.optimizer)
        print("✓ Optimizer prepared with accelerator")

        # 4. Recreate scheduler
        total_steps = len(self.train_loader) * self.num_epochs
        self._build_scheduler(total_steps)

        # 5. Restore scheduler position and apply LRs
        if "scheduler_state" in checkpoint:
            scheduler_state = checkpoint["scheduler_state"]
            self.scheduler.step_count = scheduler_state.get("step_count", 0)
            print(f"✓ Scheduler position restored: {self.scheduler.step_count}")
        else:
            # Backward compatibility: calculate scheduler step based on epoch
            if "epoch" in checkpoint:
                epoch = checkpoint["epoch"]
                steps_per_epoch = len(self.train_loader)
                estimated_steps = epoch * steps_per_epoch
                self.scheduler.step_count = estimated_steps
                print(f"✓ Scheduler position estimated: {estimated_steps}")

        # Apply scheduler to set correct LRs based on step_count
        if self.scheduler.step_count > 0:
            # Calculate what LRs should be at this step
            if self.scheduler.step_count < self.scheduler.warmup_steps:
                # Linear warmup
                scale = (self.scheduler.step_count + 1) / self.scheduler.warmup_steps
            else:
                # Cosine annealing
                progress = (self.scheduler.step_count - self.scheduler.warmup_steps) / (
                    self.scheduler.total_steps - self.scheduler.warmup_steps
                )
                scale = self.scheduler.min_lr_scale + (1 - self.scheduler.min_lr_scale) * 0.5 * (
                    1 + np.cos(np.pi * progress)
                )

            # Apply calculated LRs
            for group, base_lr in zip(self.optimizer.param_groups, self.scheduler.base_lrs):
                group["lr"] = base_lr * scale

            print(f"✓ LRs applied for step {self.scheduler.step_count} (scale={scale:.6f})")

        # 6. Load training progress
        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1  # Start from next epoch
            print(f"✓ Resuming from epoch {self.start_epoch + 1}")

        # 7. Load best metrics
        if "metrics" in checkpoint and "macro_roc_auc" in checkpoint["metrics"]:
            self.best_auc = checkpoint["metrics"]["macro_roc_auc"]
            print(f"✓ Best AUC so far: {self.best_auc:.4f}")

        print(f"✅ Checkpoint loaded successfully! Fresh optimizer with correct LRs for stage {self.stage}")
        print("   📝 Note: Optimizer momentum reset (this is OK for resume)")

        # Show final optimizer state
        print("📊 Final optimizer state:")
        for i, group in enumerate(self.optimizer.param_groups):
            print(f"  Group {i}: LR={group['lr']:.2e}, WD={group['weight_decay']:.3f}, params={len(group['params'])}")

    def restart_from_weights_only(self):
        """Restart training with only model weights from checkpoint (fresh optimizer/scheduler/epochs)."""
        print(f"🔄 Restarting training with weights from: {self.restart_from_weights}")
        checkpoint = self._load_model_checkpoint(self.restart_from_weights)

        # Load ONLY model weights
        if "model_state_dict" in checkpoint:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print("✅ Model weights loaded")

        # Show what we loaded
        if "epoch" in checkpoint:
            print(f"📈 Loaded weights from epoch {checkpoint['epoch']}")
        if "metrics" in checkpoint and "macro_roc_auc" in checkpoint["metrics"]:
            auc = checkpoint["metrics"]["macro_roc_auc"]
            print(f"📊 Original model had AUC: {auc:.4f}")

        # Our restart policy: 1 epoch head-only, then switch to full finetune
        self.stage = 0
        self._restart_head_only_done = False
        print("🚀 Starting fresh training cycle with loaded weights:")
        print("   - Epoch: 0 (head-only)")
        print("   - After first epoch → Stage 2 (full encoder with LLRD)")
        print("   - Optimizer: fresh state")
        print("   - Scheduler: fresh state")

    def setup_model(self):
        """Initialize CT-CLIP model and classification head."""
        print("Loading CT-CLIP model...")

        # Initialize BERT tokenizer and text encoder (cached to avoid repeated downloads)
        if not hasattr(MosmedMultiLabelTrainer, "_cached_bert_model"):
            print("🔥 DOWNLOADING BERT MODEL FOR THE FIRST TIME (210MB)...")
            tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True)
            text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
            text_encoder.resize_token_embeddings(len(tokenizer))
            MosmedMultiLabelTrainer._cached_bert_model = (tokenizer, text_encoder)
            print("✅ BERT model cached successfully!")
        else:
            print("✅ Using cached BERT model (no download)...")
            tokenizer, text_encoder = MosmedMultiLabelTrainer._cached_bert_model

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

        # Initialize CT-CLIP
        ct_clip = CTCLIP(
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

        # Load pretrained weights (supports S3 paths)
        checkpoint = self._load_model_checkpoint(self.model_path)
        if "model_state_dict" in checkpoint:
            ct_clip.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            ct_clip.load_state_dict(checkpoint, strict=False)

        # Create classifier
        self.model = SimpleCTCLIPClassifier(
            ct_clip,
            num_classes=len(self.target_pathologies),
            latent_norm=self.latent_norm,
            reinit_head=self.reinit_head,
            head_init_std=self.head_init_std,
        )
        # Set tokenizer from CTCLIP (avoid duplicate loading)
        self.model.tokenizer = ct_clip.tokenizer

        # Start with frozen encoder
        self.model.freeze_encoder()

        # Always keep text encoder frozen (we only use visual features)
        for param in self.model.ct_clip.text_transformer.parameters():
            param.requires_grad = False
        print("Encoder frozen for initial training (text encoder permanently frozen)")

    def setup_data(self):
        """Initialize datasets and dataloaders."""
        print("Setting up datasets...")

        # Create data loaders using the dedicated module
        self.train_loader, self.val_loader, y_all = create_data_loaders(
            csv_path=self.csv_path,
            nifti_dir=self.nifti_dir,
            target_pathologies=self.target_pathologies,
            batch_size=self.batch_size,
            val_split=0.15,
            cache_dir="/training-data/celsus_pli/ct_clip_cache",  # Cache preprocessed data
            max_samples=self.max_samples,  # For debugging
            num_workers=8,
            random_seed=42,
            enable_augmentations=self.enable_augmentations,
            aug_flip_prob=self.aug_flip_prob,
            use_pseudo_labels=self.use_pseudo_labels,
            pseudo_label_weight=self.pseudo_label_weight,
            prefer_soft_pseudo=self.prefer_soft_pseudo,
        )

        # Print class balance info (ASL doesn't need pos_weights)
        known = ~torch.isnan(y_all)
        pos = ((y_all == 1) & known).sum(dim=0).float()
        neg = ((y_all == 0) & known).sum(dim=0).float()

        print("Class balance (for info, ASL doesn't use pos_weights):")
        for i, pathology in enumerate(self.target_pathologies):
            p, n = pos[i], neg[i]
            total = p + n
            pos_ratio = p / total if total > 0 else 0
            print(f"  {pathology}: pos={p:.0f}, neg={n:.0f}, pos_ratio={pos_ratio:.3f}")

        print("Data loaders created successfully")

        # Setup loss function based on pseudo-label usage
        if self.use_pseudo_labels:
            # For pseudo-labels: use SampleWeightedBCE (no class/pos weights)
            from training_losses import SampleWeightedBCE

            self.model.loss_fn = SampleWeightedBCE()
            print("🔄 Using SampleWeightedBCE for pseudo-labels (no class/pos weights)")
        else:
            # For real labels only: use MaskedWeightedBCE with class weights
            known = ~torch.isnan(y_all)
            pos = ((y_all == 1) & known).sum(dim=0).float()
            neg = ((y_all == 0) & known).sum(dim=0).float()
            class_w, pos_w = make_class_weights(pos, neg, beta=0.9999, min_pos_w=1.2, max_pos_w=8.0)
            self.model.loss_fn = MaskedWeightedBCE(class_weight=class_w, pos_weight=pos_w)
            print("⚖️ Using MaskedWeightedBCE with class weights (real labels only)")

    def _split_params(self, modules, lr, wd, betas=(0.9, 0.95)):
        """Split parameters into weight decay and no weight decay groups."""
        nd_keys = (
            "bias",
            "layernorm",
            "ln",
            "norm",
            "pos_embed",
            "relative_position",
            "logit_scale",
            "temperature",
            "token_embed",
            "cls_token",
        )
        decay, no_decay = [], []

        for m in modules:
            for n, p in m.named_parameters(recurse=True):
                if not p.requires_grad:
                    continue
                if n.endswith(".bias") or any(k in n.lower() for k in nd_keys):
                    no_decay.append(p)
                else:
                    decay.append(p)

        groups = []
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": wd, "betas": betas})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0, "betas": betas})
        return groups

    def _build_optimizer_for_stage(self):
        """Build optimizer for current stage."""
        params = []

        # LR constants (parameterized)
        lr_head = float(self.lr_head)
        base_encoder_lr = float(self.lr_encoder)

        # Stage 1: top-2 blocks with LLRD from base encoder LR
        lr_stage1_L1 = 1.0 * base_encoder_lr
        lr_stage1_L2 = 0.6 * base_encoder_lr

        # Stage 2: full LLRD from base encoder LR
        lr_stage2_early = 0.2 * base_encoder_lr
        lr_stage2_mid = 0.6 * base_encoder_lr
        lr_stage2_late = 1.0 * base_encoder_lr
        lr_patch_embed = 0.2 * base_encoder_lr

        # Betas
        betas_encoder = (0.9, 0.95)
        betas_head = (0.9, 0.999)

        # Head always trains with head betas
        params += self._split_params([self.model.classifier], lr=lr_head, wd=self.weight_decay, betas=betas_head)

        if self.stage == 0:
            # Stage 0: freeze encoder completely
            for p in self.model.ct_clip.parameters():
                p.requires_grad = False
            print("Stage 0: Head-only training")

        elif self.stage == 1:
            # Stage 1: unfreeze top-2 blocks with LLRD
            for p in self.model.ct_clip.parameters():
                p.requires_grad = False

            # Find transformer layers (try different possible paths)
            vit = self.model.ct_clip.visual_transformer
            if hasattr(vit, "blocks"):
                layers = vit.blocks
            elif hasattr(vit, "layers"):
                layers = vit.layers
            elif hasattr(vit, "transformer") and hasattr(vit.transformer, "layers"):
                layers = vit.transformer.layers
            else:
                # Fallback: find any module with "block" or "layer" in name
                layers = []
                for name, module in vit.named_modules():
                    if "block" in name.lower() or "layer" in name.lower():
                        layers.append(module)

            if len(layers) >= 2:
                top_blocks = [layers[-1], layers[-2]]  # Last 2 blocks
                for blk in top_blocks:
                    for p in blk.parameters():
                        p.requires_grad = True

                # LLRD: different LR for L-1 and L-2 with encoder betas
                params += self._split_params(
                    [top_blocks[0]], lr=lr_stage1_L1, wd=self.weight_decay, betas=betas_encoder
                )  # L-1
                params += self._split_params(
                    [top_blocks[1]], lr=lr_stage1_L2, wd=self.weight_decay, betas=betas_encoder
                )  # L-2
                print(f"Stage 1: Unfroze top-2 blocks ({len(layers)} total blocks)")
            else:
                print(f"Warning: Could not find transformer blocks in ViT, found {len(layers)} layers")

        elif self.stage == 2:
            # Stage 2: full unfreeze with LLRD (except text encoder)
            for p in self.model.ct_clip.parameters():
                p.requires_grad = True

            # Keep text encoder frozen
            for param in self.model.ct_clip.text_transformer.parameters():
                param.requires_grad = False

            # Find transformer layers (support both ViT-style and CTViT)
            vit = self.model.ct_clip.visual_transformer

            layers = []
            if hasattr(vit, "blocks"):
                layers = list(vit.blocks)
            elif hasattr(vit, "layers"):
                layers = list(vit.layers)
            elif hasattr(vit, "transformer") and hasattr(vit.transformer, "layers"):
                layers = list(vit.transformer.layers)

            # CTViT support: combine spatial and temporal encoder layers if present
            if hasattr(vit, "enc_spatial_transformer") and hasattr(vit.enc_spatial_transformer, "layers"):
                layers += list(vit.enc_spatial_transformer.layers)
            if hasattr(vit, "enc_temporal_transformer") and hasattr(vit.enc_temporal_transformer, "layers"):
                layers += list(vit.enc_temporal_transformer.layers)

            if len(layers) > 0:
                depth = len(layers)
                if depth > 6:
                    early = layers[: depth // 3]
                    mid = layers[depth // 3 : 2 * depth // 3]
                    late = layers[2 * depth // 3 :]

                    params += self._split_params(
                        list(early), lr=lr_stage2_early, wd=self.weight_decay, betas=betas_encoder
                    )
                    params += self._split_params(list(mid), lr=lr_stage2_mid, wd=self.weight_decay, betas=betas_encoder)
                    params += self._split_params(
                        list(late), lr=lr_stage2_late, wd=self.weight_decay, betas=betas_encoder
                    )
                    print(
                        f"Stage 2: LLRD over encoder layers depth={depth} -> early/mid/late: {len(early)}/{len(mid)}/{len(late)}"
                    )
                else:
                    mid_point = depth // 2
                    early = layers[:mid_point]
                    late = layers[mid_point:]
                    params += self._split_params(
                        list(early), lr=lr_stage2_early, wd=self.weight_decay, betas=betas_encoder
                    )
                    params += self._split_params(
                        list(late), lr=lr_stage2_late, wd=self.weight_decay, betas=betas_encoder
                    )
                    print(f"Stage 2: 2-way LLRD over encoder layers: early={len(early)}, late={len(late)}")
            else:
                print("Warning: No encoder layers found to unfreeze; please verify encoder structure")

            # Patch/embed modules in CTViT
            patch_like_modules = []
            if hasattr(vit, "patch_embed"):
                patch_like_modules.append(vit.patch_embed)
            if hasattr(vit, "to_patch_emb"):
                patch_like_modules.append(vit.to_patch_emb)
            if hasattr(vit, "to_patch_emb_first_frame"):
                patch_like_modules.append(vit.to_patch_emb_first_frame)
            if patch_like_modules:
                params += self._split_params(
                    patch_like_modules, lr=lr_patch_embed, wd=self.weight_decay, betas=betas_encoder
                )

        self.optimizer = torch.optim.AdamW(params, eps=1e-8)

        # Print parameter groups
        print(f"Optimizer created with {len(self.optimizer.param_groups)} parameter groups:")
        for i, group in enumerate(self.optimizer.param_groups):
            lr = group["lr"]
            wd = group["weight_decay"]
            num_params = len(group["params"])
            print(f"  Group {i}: LR={lr:.2e}, WD={wd:.3f}, params={num_params}")

    def _build_scheduler(self, total_steps):
        """Build scheduler for current optimizer."""
        warmup_steps = int(total_steps * float(self.warmup_ratio))
        self.scheduler = CosineWithWarmup(
            optimizer=self.optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_scale=0.1
        )
        print(f"Scheduler: {total_steps} total steps, {warmup_steps} warmup steps, min_lr_scale=0.1")

    def setup_optimizer(self):
        """Initialize optimizer and scheduler for stage 0."""
        self._build_optimizer_for_stage()
        total_steps = len(self.train_loader) * self.num_epochs
        self._build_scheduler(total_steps)

    def prepare_training(self):
        """Prepare training with accelerator."""
        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Gradient tracking
        total_grad_norm = 0.0
        total_clipped_grad_norm = 0.0
        clipping_events = 0
        optimizer_steps = 0

        pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            step = epoch * len(self.train_loader) + batch_idx

            # Unpack batch - dataset ALWAYS returns 4 elements: (images, targets, sample_weights, fnames)
            # When pseudo-labels are disabled, sample_weights are all 1.0 for real labels
            if len(batch) != 4:
                raise ValueError(f"Expected 4 elements in batch, got {len(batch)}")
            images, targets, sample_weights, fnames = batch

            # Forward pass
            outputs = self.model(images, targets, sample_weights)
            loss = outputs["loss"]

            # Backward pass
            self.accelerator.backward(loss)

            # Calculate gradient norm before clipping
            max_norm = float(self.clip_grad_norm)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

            # Track gradient statistics
            total_grad_norm += grad_norm.item()
            clipped_grad_norm = min(grad_norm.item(), max_norm)
            total_clipped_grad_norm += clipped_grad_norm

            if grad_norm.item() > max_norm:
                clipping_events += 1

            optimizer_steps += 1

            # Update with correct order
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log to ClearML
            if self.logger and batch_idx % 10 == 0:
                self.logger.report_scalar("Loss", "train_step", loss.item(), iteration=step)

                # Log learning rates for all groups
                for i, group in enumerate(self.optimizer.param_groups):
                    group_name = f"group_{i}" if i > 0 else "head"
                    self.logger.report_scalar("Learning Rate", group_name, group["lr"], iteration=step)

                # Log gradient statistics
                if optimizer_steps > 0:
                    self.logger.report_scalar(
                        "Gradients", "grad_norm", total_grad_norm / optimizer_steps, iteration=step
                    )
                    self.logger.report_scalar(
                        "Gradients", "grad_norm_clipped", total_clipped_grad_norm / optimizer_steps, iteration=step
                    )
                    self.logger.report_scalar(
                        "Gradients", "clip_fraction", clipping_events / optimizer_steps, iteration=step
                    )

        avg_loss = total_loss / num_batches

        # Log epoch-level gradient statistics
        if self.logger and optimizer_steps > 0:
            self.logger.report_scalar(
                "Gradients", "epoch_grad_norm", total_grad_norm / optimizer_steps, iteration=epoch
            )
            self.logger.report_scalar(
                "Gradients", "epoch_clip_fraction", clipping_events / optimizer_steps, iteration=epoch
            )

        return {"train_loss": avg_loss}

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_targets = []
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm.tqdm(self.val_loader, desc="Validation"):
                # Validation NEVER uses pseudo-labels - always real labels only
                # Batch format is always: (images, targets, sample_weights, fnames)
                # where sample_weights are all 1.0 for real labels
                if len(batch) != 4:
                    raise ValueError(f"Expected 4 elements in validation batch, got {len(batch)}")
                images, targets, sample_weights, fnames = batch

                outputs = self.model(images, targets, sample_weights)
                loss = outputs["loss"]

                total_loss += loss.item()
                num_batches += 1

                # Collect predictions for metrics
                logits = outputs["logits"].detach().cpu()
                all_logits.append(logits)
                all_targets.append(targets.cpu())

        # Calculate metrics
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Convert to numpy
        logits_np = torch.sigmoid(all_logits).numpy()
        targets_np = all_targets.numpy()

        # Calculate ROC-AUC per class (only for classes with both pos and neg samples)
        roc_aucs = []
        per_class_aucs = {}

        for i, pathology in enumerate(self.target_pathologies):
            # Get known labels for this class
            known_mask = ~np.isnan(targets_np[:, i])
            if known_mask.sum() > 0:
                y_true = targets_np[known_mask, i]
                y_pred = logits_np[known_mask, i]

                # Check if we have both classes
                if len(np.unique(y_true)) > 1:
                    auc = roc_auc_score(y_true, y_pred)
                    roc_aucs.append(auc)
                    per_class_aucs[pathology] = auc

        avg_loss = total_loss / num_batches
        macro_auc = np.mean(roc_aucs) if roc_aucs else 0.0

        metrics = {"val_loss": avg_loss, "macro_roc_auc": macro_auc, "num_valid_classes": len(roc_aucs)}

        # Add per-class AUC scores to metrics
        for pathology, auc in per_class_aucs.items():
            metrics[f"auc_{pathology}"] = auc

        # Log to ClearML
        if self.logger:
            self.logger.report_scalar("Loss", "val", avg_loss, iteration=epoch)
            self.logger.report_scalar("ROC-AUC", "macro", macro_auc, iteration=epoch)

            # Log per-class AUC scores
            for pathology, auc in per_class_aucs.items():
                self.logger.report_scalar("ROC-AUC", pathology, auc, iteration=epoch)

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        if not self.accelerator.is_main_process:
            return

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.accelerator.get_state_dict(self.model),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state": {
                "step_count": self.scheduler.step_count,
                "warmup_steps": self.scheduler.warmup_steps,
                "total_steps": self.scheduler.total_steps,
                "min_lr_scale": self.scheduler.min_lr_scale,
                "base_lrs": self.scheduler.base_lrs,
            },
            "stage": self.stage,  # Save current training stage
            "metrics": metrics,
            "target_pathologies": self.target_pathologies,
        }

        checkpoint_path = self.results_folder / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def train(self):
        """Main training loop."""
        print(f"Starting training from epoch {self.start_epoch + 1}...")

        for epoch in range(self.start_epoch, self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")

            # Stage transitions
            stage_changed = False

            # Default schedule: Stage 0→1→2
            if (
                self.restart_from_weights
                and not self._restart_head_only_done
                and self.stage == 0
                and epoch > self.start_epoch
            ):
                # After the first epoch of head-only, go directly to full finetune
                if self.accelerator.is_main_process:
                    print("→ Stage 2: unfreeze all encoder (LLRD) [RESTART POLICY]")
                self.stage = 2
                self._restart_head_only_done = True
                stage_changed = True
            elif epoch == 1 and self.stage == 0:
                if self.accelerator.is_main_process:
                    print("→ Stage 1: unfreeze top-2 blocks")
                self.stage = 1
                stage_changed = True
            elif epoch > 5 and self.stage == 1:  # Extended: Stage 1 until epoch 20
                if self.accelerator.is_main_process:
                    print("→ Stage 2: unfreeze all encoder (LLRD)")
                self.stage = 2
                stage_changed = True

            # Rebuild optimizer/scheduler only if stage actually changed
            if stage_changed:
                print(f"Rebuilding optimizer and scheduler for stage {self.stage}...")
                self._build_optimizer_for_stage()
                total_steps = len(self.train_loader) * self.num_epochs
                self._build_scheduler(total_steps)
                # Only prepare optimizer (model and loaders are already prepared)
                self.optimizer = self.accelerator.prepare(self.optimizer)

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate_epoch(epoch)

            # Print metrics
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Macro ROC-AUC: {val_metrics['macro_roc_auc']:.4f}")
            print(f"Valid classes: {val_metrics['num_valid_classes']}")

            # Save checkpoint
            all_metrics = {**train_metrics, **val_metrics}
            self.save_checkpoint(epoch, all_metrics)

            # Save best model
            if val_metrics["macro_roc_auc"] > self.best_auc:
                self.best_auc = val_metrics["macro_roc_auc"]
                if self.accelerator.is_main_process:
                    best_path = self.results_folder / "best_model.pt"
                    torch.save(self.accelerator.get_state_dict(self.model), best_path)
                    print(f"New best model saved: {best_path}")

        print("Training completed!")
        print(f"Best Macro ROC-AUC: {self.best_auc:.4f}")


def main():
    """Main function."""
    # Configuration
    target_pathologies = [
        "cancer",
        "emphysema",
        "pleural_effusion",
        "coronary_calcium",
        "fibrosis",
        "airiness_decrease",
        "covid",
        "infiltrate",
        "atelectasis",
        "paracardial_fat_pathology",
        "pulmonary_trunk_pathology",
        "osteo_fracture",
        "aorta_pathology",  # Excluding problematic ones
    ]

    config = {
        "model_path": "/media/crazyfrogspb/Repos/CT-CLIP/CT_VocabFine_v2.pt",
        "csv_path": "/astral-data/Minio/kate/data/raw/mosmed_hackathon/train_val_v1.csv",
        "nifti_dir": "/astral-data/Minio/kate/data/raw/mosmed_hackathon/NIFTI/volumes",
        "target_pathologies": target_pathologies,
        "batch_size": 16,
        "num_epochs": 30,
        "lr_head": 1e-3,
        "lr_encoder": 1e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "results_folder": "./mosmed_results",
        "project_name": "Celsus_PLI",
        "task_name": "CT-CLIP-MultiLabel-Simple",
    }

    # Initialize trainer
    trainer = MosmedMultiLabelTrainer(**config)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
