from functools import partial
from typing import Callable, Dict, Sequence, Tuple, Union

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, Block
from models.vision_encoders import DinoSigLIPImageTransform, DinoSigLIPViTBackbone, unpack_tuple
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from utils.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)

class InfusedVisionTransformerBlock(nn.Module):
    def __init__(
        self,
        block,
        vision_dim: int,
    ):
        super().__init__()
        self.block = block
        self.infusion_scale_fc = nn.Linear(vision_dim, vision_dim, bias=False)
        self.infusion_shift_fc = nn.Linear(vision_dim, vision_dim, bias=False)
        
        nn.init.zeros_(self.infusion_scale_fc.weight)
        nn.init.zeros_(self.infusion_shift_fc.weight)

    def _replace_input_embeddings_old(
        self,
        input_embeddings: torch.Tensor,
        mask: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Replace positions indicated by mask with provided features in a differentiable way.

        Args:
            input_embeddings: (B, S, D)
            mask: (B, S) boolean or 0/1 mask indicating positions to replace
            features: (B, K, D) feature vectors to insert at the masked positions (per batch)

        Returns:
            Tensor of shape (B, S, D) with masked positions replaced by features, preserving autograd.
        """
        if features.dtype != input_embeddings.dtype:
            features = features.to(input_embeddings.dtype)

        new_input_embeddings = input_embeddings.clone() # (B, S, D)
        mask_bool = mask.to(dtype=torch.bool)
        mask_expanded = mask_bool.unsqueeze(-1) # (B, S, 1)

        repositioned_noisy_action_features = torch.zeros_like(new_input_embeddings) # (B, S, D)
        batch_indices = torch.arange(new_input_embeddings.shape[0], device=new_input_embeddings.device) # (B,)
        batch_indices = batch_indices.unsqueeze(1).expand(-1, features.shape[1]) # (B, K)

        masked_indices = torch.stack([torch.where(mask)[0] for mask in mask_bool]) # (B, K)
        repositioned_noisy_action_features[batch_indices, masked_indices] = features # (B, S, D)

        new_input_embeddings = torch.where(mask_expanded, repositioned_noisy_action_features, new_input_embeddings) # (B, S, D)

        return new_input_embeddings

    def forward(self, x, memory_embedding):
        if memory_embedding is None:
            return self.block(x)
        
        scale_embedding = self.infusion_scale_fc(memory_embedding)  # (batch_size, num_patches, vision_dim)
        beta_embedding = self.infusion_shift_fc(memory_embedding)  # (batch_size, num_patches, vision_dim)

        x = x + self.block.drop_path1(self.block.ls1(self.block.attn(self.block.norm1(x))))

        if memory_embedding.shape[1] != x.shape[1]:
            assert x.shape[1] == 261, "Expected x to have 261 tokens (1 cls + 256 patches + 4 register tokens)."
            
            new_patch = x[:, 1:257] + x[:, 1:257] * scale_embedding + beta_embedding
            mask = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
            mask[:, 1:257] = True
            x = self._replace_input_embeddings_old(x, mask, new_patch)
        else:
            x = x + x * scale_embedding + beta_embedding

        # Sanitize non-finite values in x (NaN/Inf -> 0) while preserving autograd graph
        # Use a conditional to avoid unnecessary work when everything is finite
        if not torch.isfinite(x).all():
            overwatch.warning("Nan/Inf detected in infused vision embedding.")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        x = x + self.block.drop_path2(self.block.ls2(self.block.mlp(self.block.norm2(x))))

        return x


class NullVisionTransformerBlockWrapper(nn.Module):
    """
    Null wrapper for ViT blocks that doesn't do anything; just calls the original block's forward function.
    Useful if you want to use a block wrapper every X blocks instead of every block (e.g., to reduce the number of new
    parameters introduced by a new wrapper).
    """

    def __init__(
        self,
        block,
    ):
        super().__init__()
        self.block = block

    def forward(self, x, memory_embedding):
        return self.block(x)


class InfusedVisionTransformer(VisionTransformer):

    def _intermediate_layers(
        self,
        x: torch.Tensor,
        fused_embeddings: torch.Tensor,
        n: Union[int, Sequence] = 1,
    ):
        """
        Copy of timm.models.vision_transformer.VisionTransformer._intermediate_layers() with modifications
        to take in language embeddings as additional input.
        """
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x, fused_embeddings)  # Modified to receive language_embeddings
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        fused_embeddings: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """
        Copy of timm.models.vision_transformer.VisionTransformer.get_intermediate_layers() with modifications
        to allow language embeddings as additional input.
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, fused_embeddings, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0 : self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens :] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)


class InfusedDinoSigLIPViTBackbone(nn.Module):
    def __init__(self, vision_backbone: DinoSigLIPViTBackbone, infuse=True, infuse_layers_ratio=0.25) -> None:
        super().__init__()
        self.vision_backbone = vision_backbone
        self.num_images_in_input = None
        self.infuse = infuse
        self.infuse_layers_ratio = infuse_layers_ratio

        # Wrap vision transformers
        self._wrap_vit(self.vision_backbone.dino_featurizer)  # DINOv2
        self._wrap_vit(self.vision_backbone.siglip_featurizer)  # SigLIP

    def _wrap_vit(self, vit: VisionTransformer) -> None:
        # Wrap vision transformer blocks
        block_wrappers = []
        num_blocks = len(vit.blocks)
        num_infuse_blocks = int(num_blocks * self.infuse_layers_ratio)
        if num_infuse_blocks == 0:
            num_infuse_blocks = 1  # Ensure at least one block is infused
        overwatch.info(f"Infusing last {num_infuse_blocks}/{num_blocks} ViT blocks.", ctx_level=2)
        infuse_index_set = set(range(num_blocks-num_infuse_blocks-1, num_blocks-1))
        
        for i, block in enumerate(vit.blocks):
            if self.infuse and i in infuse_index_set: 
                block_wrappers.append(InfusedVisionTransformerBlock(block=block, vision_dim=vit.embed_dim))
            else:
                block_wrappers.append(NullVisionTransformerBlockWrapper(block=block))
        vit.blocks = nn.Sequential(*block_wrappers)

        # Wrap vision transformer with new class that overrides functions used for forward pass
        vit.__class__ = InfusedVisionTransformer
        vit.forward = unpack_tuple(partial(vit.get_intermediate_layers, n={len(vit.blocks) - 2}))

    def get_num_patches(self) -> int:
        """Returns the number of vision patches output by the vision backbone."""
        return self.vision_backbone.num_patches()

    def forward(self, pixel_values: Dict[str, torch.Tensor], fused_memory: torch.Tensor=None) -> torch.Tensor:
        """
        Args:
            pixel_values (Dict[str, torch.Tensor]): Pixels for input image(s), (B, num_cameras, C, H, W).
            fused_memory (torch.Tensor): Fused memory tensor, (B, seq_len, dim).
        """
        assert pixel_values.keys() == {'dino', 'siglip'}, "Expected pixel_values to contain both 'dino' and 'siglip' keys."
        assert len(pixel_values['dino'].shape) == 5, "Expected pixel_values['dino'] to have shape (B, num_cameras, C, H, W)."
        
        B, num_cameras, C, H, W = pixel_values['dino'].shape
        if fused_memory is not None:
            fused_memory = fused_memory.view(B, num_cameras, self.num_patches, self.embed_dim)  # (B, num_cameras, seq_len, dim)
            fused_memory = fused_memory.flatten(0, 1)  # (B * num_cameras, seq_len, dim)
            dino_fused_memory = fused_memory[:, :, :self.vision_backbone.dino_featurizer.embed_dim]       # (B * num_cameras, seq_len, dino_dim)
            siglip_fused_memory = fused_memory[:, :, self.vision_backbone.dino_featurizer.embed_dim:]    # (B * num_cameras, seq_len, siglip_dim
        else:
            dino_fused_memory = None
            siglip_fused_memory = None

        dino_patches = pixel_values['dino'].flatten(0, 1)  # (B * num_cameras, C, H, W) 
        siglip_patches = pixel_values['siglip'].flatten(0, 1)  # (B * num_cameras, C, H, W)
        dino_patches = self.vision_backbone.dino_featurizer(dino_patches, dino_fused_memory) # (B * num_cameras, num_patches, dino_dim)
        siglip_patches = self.vision_backbone.siglip_featurizer(siglip_patches, siglip_fused_memory) # (B * num_cameras, num_patches, siglip_dim)
        dino_patches = dino_patches.unflatten(0, (B, num_cameras)) # (B, num_cameras, num_patches, dino_dim)
        siglip_patches = siglip_patches.unflatten(0, (B, num_cameras)) # (B, num_cameras, num_patches, siglip_dim)

        num_patch = dino_patches.shape[2]
        dino_dim, siglip_dim = dino_patches.shape[3], siglip_patches.shape[3]
        dino_patches = dino_patches.reshape(B, num_cameras * num_patch, dino_dim)       # (B, num_cameras * num_patches, dino_dim)
        siglip_patches = siglip_patches.reshape(B, num_cameras * num_patch, siglip_dim) # (B, num_cameras * num_patches, siglip_dim)

        combined_patches = torch.cat([dino_patches, siglip_patches], dim=2)
        
        return combined_patches  # (B, num_cameras * num_patches, dino_dim + siglip_dim)

    def get_image_transform(self) -> DinoSigLIPImageTransform:
        return self.vision_backbone.image_transform

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps each ViT block and then both of the _entire_ featurizers."""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        film_block_policy = partial(transformer_auto_wrap_policy, module_classes={InfusedVisionTransformerBlock})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy, film_block_policy])

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.vision_backbone.dino_data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        return self.vision_backbone.dino_featurizer.embed_dim + self.vision_backbone.siglip_featurizer.embed_dim

    @property
    def num_patches(self) -> int:
        # assert self.vision_backbone.dino_featurizer.patch_embed.num_patches == self.vision_backbone.siglip_featurizer.patch_embed.num_patches
        return self.vision_backbone.dino_featurizer.patch_embed.num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16

