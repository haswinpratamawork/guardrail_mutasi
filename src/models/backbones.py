"""Backbone loaders for the two supported architectures.

A backbone alone only produces features — see linear_probe.py and
finetune_head.py for the two classifier heads built on top of it.

Two modes per backbone:
  - "feature extractor": pooled embedding, backbone frozen, for the linear
    probe (logistic regression / XGBoost head).
  - "finetune model": backbone + attached Linear(features -> num_classes)
    head, all weights trainable, for end-to-end fine-tuning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import timm
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

DIT_CHECKPOINT = "microsoft/dit-base"
TIMM_NAMES = {
    "resnet50": "resnet50.a1_in1k",
    "efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "mobilenetv3": "mobilenetv3_large_100.ra_in1k",
    # Trained specifically for strong frozen-linear-probe transfer (unlike the
    # supervised-classification backbones above) — see conversation.
    "dinov2": "vit_base_patch14_dinov2.lvd142m",
    "clip_vit_b16": "vit_base_patch16_clip_224.laion2b",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
}


@dataclass
class FeatureExtractor:
    model: nn.Module
    preprocess: Callable
    embedding_dim: int

    @torch.no_grad()
    def embed_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Embed already-preprocessed tensors.

        Separate from `embed` so callers can preprocess one image at a time
        and release each full-resolution PIL before opening the next — some
        rendered pages decode to ~144MB, so holding a whole batch of them at
        once is what OOM-kills this on a small VM.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        return self._forward(batch.to(device))

    @torch.no_grad()
    def embed(self, pil_images: list) -> torch.Tensor:
        return self.embed_batch(torch.stack([self.preprocess(img) for img in pil_images]))

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _DitFeatureExtractor(FeatureExtractor):
    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=batch)
        return out.last_hidden_state[:, 0]  # [CLS] token


class _TimmFeatureExtractor(FeatureExtractor):
    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)  # num_classes=0 -> pooled features


def make_letterbox_transform(size: int, mean: tuple, std: tuple) -> Callable:
    """Resize the whole page to fit within `size`x`size` (aspect preserved,
    padded to square) instead of the ImageNet-style Resize+CenterCrop.

    CenterCrop assumes a roughly-square photo with the subject centered —
    for a tall document page, it crops off the top/bottom edges, which is
    exactly where truncation/quality defects (data_terpotong, kualitas_citra)
    tend to live. Letterboxing keeps the entire page, just downsampled.
    """

    def _transform(img: Image.Image) -> torch.Tensor:
        w, h = img.size
        scale = size / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
        tensor = TF.to_tensor(canvas)
        return TF.normalize(tensor, mean=list(mean), std=list(std))

    return _transform


def build_feature_extractor(
    backbone: str,
    image_size: int | None = None,
    preserve_aspect: bool = True,
    pretrained: bool = True,
) -> FeatureExtractor:
    """`image_size`: override the backbone's default input resolution
    (timm backbones only — fully convolutional + global-average-pooled, so
    they accept any input size with no architecture change; DiT's fixed
    positional embeddings mean this override isn't supported for it).

    `preserve_aspect`: True (default) = letterbox resize, no crop, keeps the
    whole page. False = classic Resize+CenterCrop at `image_size` — crops
    page edges, but avoids the blank-padding dilution letterboxing causes
    on a GAP-pooled backbone (see conversation: letterbox measurably hurt
    accuracy relative to crop at the same resolution).

    `pretrained`: False builds the architecture with random weights and skips
    the hub download, for callers that immediately load their own state_dict
    (the serving container, which ships weights in its model bundle and must
    not depend on network access at startup). timm still attaches the
    pretrained_cfg, so the preprocessing transform resolves identically."""
    if backbone == "dit":
        if image_size is not None:
            raise ValueError("image_size override isn't supported for dit (fixed positional embeddings)")
        if not pretrained:
            raise ValueError("pretrained=False isn't supported for dit (loads from the HF hub by name)")
        # Imported per-branch, not at module scope: dit is the only consumer of
        # transformers, and the serving image (timm backbones only) shouldn't
        # have to install it just to import this module.
        from transformers import AutoImageProcessor, AutoModel

        model = AutoModel.from_pretrained(DIT_CHECKPOINT)
        processor = AutoImageProcessor.from_pretrained(DIT_CHECKPOINT)
        preprocess = lambda img: torch.tensor(
            processor(img, return_tensors="np")["pixel_values"][0]
        )
        return _DitFeatureExtractor(model, preprocess, model.config.hidden_size)

    if backbone in TIMM_NAMES:
        timm_name = TIMM_NAMES[backbone]
        model = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        data_cfg = timm.data.resolve_data_config({}, model=model)
        if image_size is not None and preserve_aspect:
            preprocess = make_letterbox_transform(image_size, data_cfg["mean"], data_cfg["std"])
        elif image_size is not None:
            data_cfg = dict(data_cfg, input_size=(3, image_size, image_size), crop_pct=1.0)
            preprocess = timm.data.create_transform(**data_cfg, is_training=False)
        else:
            preprocess = timm.data.create_transform(**data_cfg, is_training=False)
        return _TimmFeatureExtractor(model, preprocess, model.num_features)

    raise ValueError(f"Unknown backbone: {backbone}")


def build_finetune_model(backbone: str, num_classes: int = 2):
    """Returns (model, preprocess_train, preprocess_eval) with a
    Linear(features -> num_classes) head attached, all weights trainable."""
    if backbone == "dit":
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        model = AutoModelForImageClassification.from_pretrained(
            DIT_CHECKPOINT, num_labels=num_classes, ignore_mismatched_sizes=True
        )
        processor = AutoImageProcessor.from_pretrained(DIT_CHECKPOINT)
        preprocess = lambda img: torch.tensor(
            processor(img, return_tensors="np")["pixel_values"][0]
        )
        return model, preprocess, preprocess

    if backbone in TIMM_NAMES:
        timm_name = TIMM_NAMES[backbone]
        model = timm.create_model(timm_name, pretrained=True, num_classes=num_classes)
        data_cfg = timm.data.resolve_data_config({}, model=model)
        preprocess_eval = timm.data.create_transform(**data_cfg, is_training=False)
        preprocess_train = timm.data.create_transform(**data_cfg, is_training=True)
        return model, preprocess_train, preprocess_eval

    raise ValueError(f"Unknown backbone: {backbone}")


def dit_logits(model, batch: torch.Tensor) -> torch.Tensor:
    return model(pixel_values=batch).logits
