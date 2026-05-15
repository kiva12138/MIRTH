from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Protocol, Tuple, Union

import timm
import torch
import torch.nn as nn
from PIL import Image
from PIL.Image import Image
from timm.models.vision_transformer import VisionTransformer
from torchvision.transforms import Compose, Resize

from config.config_vlm import Prism_7B_DINOSigLIP_224px

# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# === Interface for an Image Transform ===
class ImageTransform(Protocol):
    def __call__(self, img: Image, **kwargs: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: ...


@dataclass
class DinoSigLIPImageTransform:
    dino_image_transform: ImageTransform
    siglip_image_transform: ImageTransform

    def __call__(self, img: Image, **kwargs: str) -> Dict[str, torch.Tensor]:
        if img.width != img.height:
            print("Warning: Input image is not square. Consider cropping instead for better results.")
        return {"dino": self.dino_image_transform(img, **kwargs), "siglip": self.siglip_image_transform(img, **kwargs)}


class DinoSigLIPViTBackbone(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.default_image_size: int = config.default_image_size
        self.dino_timm_url = config.dino_url
        self.siglip_timm_url = config.siglip_url

        self.dino_featurizer: VisionTransformer = timm.create_model(self.dino_timm_url, pretrained=True, num_classes=0, img_size=self.default_image_size)
        self.siglip_featurizer: VisionTransformer = timm.create_model(self.siglip_timm_url, pretrained=True, num_classes=0, img_size=self.default_image_size)
        self.dino_featurizer.eval()
        self.siglip_featurizer.eval()

        # Return the SECOND-TO-LAST layer patches!
        self.dino_featurizer.forward = unpack_tuple(partial(self.dino_featurizer.get_intermediate_layers, n={len(self.dino_featurizer.blocks) - 2}))
        self.siglip_featurizer.forward = unpack_tuple(partial(self.siglip_featurizer.get_intermediate_layers, n={len(self.siglip_featurizer.blocks) - 2}))

        # Override default image size for larger resolution models
        self.dino_data_cfg = timm.data.resolve_model_data_config(self.dino_featurizer)
        self.siglip_data_cfg = timm.data.resolve_model_data_config(self.siglip_featurizer)
        self.dino_data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)
        self.siglip_data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        # Initialize both Transforms
        default_dino_transform = timm.data.create_transform(**self.dino_data_cfg, is_training=False)
        default_siglip_transform = timm.data.create_transform(**self.siglip_data_cfg, is_training=False)

        assert isinstance(default_dino_transform, Compose) and isinstance(default_siglip_transform, Compose)
        assert isinstance(default_dino_transform.transforms[0], Resize) and isinstance(default_siglip_transform.transforms[0], Resize)

        target_size = (self.default_image_size, self.default_image_size)
        dino_transform = Compose([Resize(target_size, interpolation=default_dino_transform.transforms[0].interpolation), *default_dino_transform.transforms[1:]])
        siglip_transform = Compose([Resize(target_size, interpolation=default_siglip_transform.transforms[0].interpolation), *default_siglip_transform.transforms[1:]])

        self.image_transform = DinoSigLIPImageTransform(dino_transform, siglip_transform)

    def get_image_transform(self) -> DinoSigLIPImageTransform:
        return self.image_transform

    def forward(self, pixel_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Runs the transformed image/pixel tensors through each vision backbone, returning concatenated patches."""
        dino_patches = self.dino_featurizer(pixel_values["dino"])[0]
        siglip_patches = self.siglip_featurizer(pixel_values["siglip"])[0]
        
        return torch.cat([dino_patches, siglip_patches], dim=2)

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.dino_data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        return self.dino_featurizer.embed_dim + self.siglip_featurizer.embed_dim

    @property
    def num_patches(self) -> int:
        assert self.dino_featurizer.patch_embed.num_patches == self.siglip_featurizer.patch_embed.num_patches
        return self.dino_featurizer.patch_embed.num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16


def get_Prism_7B_DINOSigLIP_224px_backbone_and_transform() -> Tuple[DinoSigLIPViTBackbone, DinoSigLIPImageTransform]:
    config = Prism_7B_DINOSigLIP_224px()
    vision_backbone: DinoSigLIPViTBackbone = DinoSigLIPViTBackbone(config)
    image_transform = vision_backbone.get_image_transform()
    return vision_backbone, image_transform
