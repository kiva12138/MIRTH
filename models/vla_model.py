from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from config.config_vla import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    NormalizationType,
)
from models.action_heads import L1RegressionActionHead, SinusoidalPositionalEncoding
from models.action_reason_token import PromptEncoder
from models.llm_llama2 import get_Prism_7B_DINOSigLIP_224px_backbone_llama2_and_tokenizer
from models.projectors import FusedMLPProjector, ProprioProjector
from models.vision_encoders import get_Prism_7B_DINOSigLIP_224px_backbone_and_transform
from models.memory_hub import ProprioMemoryHubForTraining, VisionMemoryHubForTraining, ActionMemoryHubForTraining
from models.infuse_wrapper import InfusedDinoSigLIPViTBackbone
from utils.overwatch import initialize_overwatch
from utils.train_utiils import get_action_tokens_mask, get_reasoning_tokens_mask

overwatch = initialize_overwatch(__name__)

@dataclass
class MIRTHConfig():
    # Path to pretrained VLA checkpoint
    pretrained_vla_path: Optional[Union[str, Path]] = None
    
    # Parameters required for initialization
    num_images_in_input: int = 1
    use_proprio: bool = False
    hf_token: Optional[str] = None
    action_token_type: str = "one_for_action_step" # options: "one_for_action_chunk", "one_for_action_step", "one_for_action_dim"
    use_timestamp: bool = False
    action_biattnn: bool = False
    
    # Parameters for Memory hub
    use_vision_memory_hub: bool = True
    use_proprio_memory_hub: bool = True
    use_action_memory_hub: bool = False
    vision_infusion_ratio: float = -1 # As the infusion machanism's performance is not stable, we set it to -1 by default to disable it. If > 0, it will not take effect and a warning is issued.
    mb_prefix_type : str = "union" # options: "union", "separate"
    long_memory_scale_number: int = 4
    short_memory_length: int = 4
    tau: float = 8.0
    beta_min: float = 0.01
    beta_max: float = 0.3
    gamma: float = 0.2
    lmbd: float = 0.2
    bias: float = 1.0
    
    # Parameters for action reason token
    use_reason_token: bool = True
    num_reason_token: int = 8
    reason_hidden: int = 512
    reason_p_drop: float = 0.0
    reason_out_scale: float = 1.0
    use_contrastive_loss: bool = False
    contrastive_tau_ra: float = -1
    contrastive_tau_rx: float = -1
    lambda_contrastive_ra: float = -1.0
    lambda_contrastive_rx: float = -1.0
    
    
@dataclass
class MIRTHOutput(CausalLMOutputWithPast):
    metrics: Optional[Dict[str, float]] = None
    actions: Optional[torch.Tensor] = None
    
    x_embeddings: Optional[torch.Tensor] = None
    r_embeddings: Optional[torch.Tensor] = None
    a_embeddings: Optional[torch.Tensor] = None


class MIRTH(nn.Module):
    def __init__(self, config: MIRTHConfig) -> None:
        super().__init__()
        self.config = config
        
        overwatch.info(f"Loading Vision Backbone...")
        self.vision_backbone, self.image_transform = get_Prism_7B_DINOSigLIP_224px_backbone_and_transform()
        overwatch.info(f"Loading Pretrained LLM...")
        # It would be much faster with use_flash_attention_2=True. But this requires to compile flash attention on your own machine.
        self.llm_backbone, self.tokenizer = get_Prism_7B_DINOSigLIP_224px_backbone_llama2_and_tokenizer(hf_token=self.config.hf_token, use_flash_attention_2=True, load_pretrained=True)
        self.projector = FusedMLPProjector(self.vision_backbone.embed_dim, self.llm_backbone.embed_dim)
        self.vlm_config = self.llm_backbone.config
        
        overwatch.info(f"Loading Pretrained VLA Weights...")
        model_state_dict = torch.load(self.config.pretrained_vla_path, map_location="cpu")["model"]
        assert len(model_state_dict) == 3 and all(k in model_state_dict for k in ["vision_backbone", "llm_backbone", "projector"]), "Invalid checkpoint !"

        overwatch.info(f"Loadding llm_backbone weights from checkpoint!")
        self.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
        overwatch.info(f"Loadding vision_backbone weights from checkpoint!")
        self.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])
        overwatch.info(f"Loadding projector weights from checkpoint!")
        self.projector.load_state_dict(model_state_dict["projector"])

        self.action_head = L1RegressionActionHead(
            input_dim=self.llm_backbone.embed_dim,
            hidden_dim=self.llm_backbone.embed_dim,
            action_token_type=self.config.action_token_type,
        )
        self.action_prediction_loss_fn = nn.L1Loss()
        overwatch.info("Initialized L1 Action Head and L1 Loss.")
        
        if self.config.vision_infusion_ratio > 0:
            overwatch.warn("Vision infusion is currently deprecated, skipping.")
        self.vision_backbone = InfusedDinoSigLIPViTBackbone(
            vision_backbone=self.vision_backbone, 
            infuse=False, 
            infuse_layers_ratio=0
        )
        overwatch.info(f"Initialized InfusedDinoSigLIPViTBackbone.")
        
        if self.config.use_proprio:
            self.proprio_projector = ProprioProjector(llm_dim=self.llm_backbone.embed_dim, proprio_dim=PROPRIO_DIM)
            overwatch.info("Initialized ProprioProjector.")
            
            if self.config.use_proprio_memory_hub:
                self.proprio_memory_hub = ProprioMemoryHubForTraining(
                    hidden_dim=PROPRIO_DIM,
                    long_memory_scale_number=self.config.long_memory_scale_number,
                    short_memory_length=self.config.short_memory_length,
                    tau=self.config.tau,
                    beta_min=self.config.beta_min,
                    beta_max=self.config.beta_max,
                    gamma=self.config.gamma,
                    lmbd=self.config.lmbd,
                    bias=self.config.bias,
                    prefix_type=self.config.mb_prefix_type,
                )
                # Whether the long / short memory are separated or unioned will affect the design of the prefix projection layers
                if self.config.mb_prefix_type == "separate":
                    self.proprio_memory_fc_long = nn.Linear(PROPRIO_DIM, self.llm_backbone.embed_dim)
                    self.proprio_memory_fc_short = nn.Linear(PROPRIO_DIM, self.llm_backbone.embed_dim)
                elif self.config.mb_prefix_type == "union":
                    self.proprio_memory_fc = nn.Linear(PROPRIO_DIM, self.llm_backbone.embed_dim)
                overwatch.info(f"Initialized Proprio Memory Hub with prefix type {self.config.mb_prefix_type}.")

        if self.config.use_vision_memory_hub:
            self.vision_memory_hub = VisionMemoryHubForTraining(
                hidden_dim=self.vision_backbone.embed_dim,
                long_memory_scale_number=self.config.long_memory_scale_number,
                short_memory_length=self.config.short_memory_length,
                tau=self.config.tau,
                beta_min=self.config.beta_min,
                beta_max=self.config.beta_max,
                gamma=self.config.gamma,
                lmbd=self.config.lmbd,
                bias=self.config.bias,
                prefix_type=self.config.mb_prefix_type,
            )
            if self.config.mb_prefix_type == "separate":
                self.vision_memory_fc_long = nn.Linear(self.vision_backbone.embed_dim, self.llm_backbone.embed_dim)
                self.vision_memory_fc_short = nn.Linear(self.vision_backbone.embed_dim, self.llm_backbone.embed_dim)
            elif self.config.mb_prefix_type == "union":
                self.vision_memory_fc = nn.Linear(self.vision_backbone.embed_dim, self.llm_backbone.embed_dim)
            overwatch.info(f"Initialized Vision Memory Hub with prefix type {self.config.mb_prefix_type}.")
            
        if self.config.use_action_memory_hub:
            self.action_memory_hub = ActionMemoryHubForTraining(
                hidden_dim=ACTION_DIM,
                long_memory_scale_number=self.config.long_memory_scale_number,
                short_memory_length=self.config.short_memory_length,
                tau=self.config.tau,
                beta_min=self.config.beta_min,
                beta_max=self.config.beta_max,
                gamma=self.config.gamma,
                lmbd=self.config.lmbd,
                bias=self.config.bias,
                prefix_type=self.config.mb_prefix_type,
            )
            if self.config.mb_prefix_type == "separate":
                self.action_memory_fc_long = nn.Linear(ACTION_DIM, self.llm_backbone.embed_dim)
                self.action_memory_fc_short = nn.Linear(ACTION_DIM, self.llm_backbone.embed_dim)
            elif self.config.mb_prefix_type == "union":
                self.action_memory_fc = nn.Linear(ACTION_DIM, self.llm_backbone.embed_dim)
            overwatch.info(f"Initialized Action Memory Hub with prefix type {self.config.mb_prefix_type}.")

        if self.config.use_reason_token:
            self.reason_token_encoder = PromptEncoder(
                d_model=self.llm_backbone.embed_dim,
                m=self.config.num_reason_token,
                hidden=self.config.reason_hidden,
                p_drop=self.config.reason_p_drop,
                out_scale=self.config.reason_out_scale,
            )
            if self.config.use_contrastive_loss:
                self.reason_proj = nn.Linear(self.llm_backbone.embed_dim, self.llm_backbone.embed_dim, bias=False)
                self.action_proj = nn.Linear(self.llm_backbone.embed_dim, self.llm_backbone.embed_dim, bias=False)
                self.input_proj = nn.Linear(self.llm_backbone.embed_dim, self.llm_backbone.embed_dim, bias=False)
                overwatch.info("Initialized Reason Token Contrastive Loss.")
            overwatch.info("Initialized Action Reason Token Encoders.")
            
        if self.config.use_timestamp:
            self.time_encoder = SinusoidalPositionalEncoding(dim=self.llm_backbone.embed_dim)
            overwatch.info("Initialized Time Encoder for action tokens.")

    def set_data_stats(self, norm_stats: Dict[str, Dict[str, np.ndarray]]) -> None:
        self.norm_stats = norm_stats
     
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def freeze_backbones(self, stage: str) -> None:
        if stage == "lv":
            self.llm_backbone.requires_grad_(False)
            self.vision_backbone.requires_grad_(False)
            self.projector.requires_grad_(True)
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> LLM Backbone", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector", ctx_level=1)
        elif stage == 'lvp':
            self.llm_backbone.requires_grad_(False)
            self.vision_backbone.requires_grad_(False)
            self.projector.requires_grad_(False)
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> LLM Backbone", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> Projector", ctx_level=1)
        else:
            raise ValueError(f"Unknown freeze stage `{stage}`")
        
        for name, param in self.vision_backbone.named_parameters():
            if "infusion" in name:
                param.requires_grad = True
        overwatch.info(f"[TRAINABLE] 🔥 =>> Infusion Layers", ctx_level=1)
        
        if self.config.use_proprio:
            self.proprio_projector.requires_grad_(True)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Proprio Projector", ctx_level=1)
            
        self.action_head.requires_grad_(True)
        overwatch.info(f"[TRAINABLE] 🔥 =>> L1 Action Head", ctx_level=1)

        if self.config.use_vision_memory_hub:
            self.vision_memory_hub.requires_grad_(True)
            if self.config.mb_prefix_type == "separate":
                self.vision_memory_fc_long.requires_grad_(True)
                self.vision_memory_fc_short.requires_grad_(True)
            elif self.config.mb_prefix_type == "union":
                self.vision_memory_fc.requires_grad_(True)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Vision Memory Hub", ctx_level=1)

        if self.config.use_proprio and self.config.use_proprio_memory_hub:
            self.proprio_memory_hub.requires_grad_(True)
            if self.config.mb_prefix_type == "separate":
                self.proprio_memory_fc_long.requires_grad_(True)
                self.proprio_memory_fc_short.requires_grad_(True)
            elif self.config.mb_prefix_type == "union":
                self.proprio_memory_fc.requires_grad_(True)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Proprio Memory Hub", ctx_level=1)
            
        if self.config.use_action_memory_hub:
            self.action_memory_hub.requires_grad_(True)
            if self.config.mb_prefix_type == "separate":
                self.action_memory_fc_long.requires_grad_(True)
                self.action_memory_fc_short.requires_grad_(True)
            elif self.config.mb_prefix_type == "union":
                self.action_memory_fc.requires_grad_(True)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Action Memory Hub", ctx_level=1)
        
        if self.config.use_timestamp:
            self.time_encoder.requires_grad_(True)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Time Encoder", ctx_level=1)
        
        if self.config.use_reason_token:
            self.reason_token_encoder.requires_grad_(True)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Action Reason Token Modules", ctx_level=1)
            if self.config.use_contrastive_loss:
                self.reason_proj.requires_grad_(True)
                self.action_proj.requires_grad_(True)
                self.input_proj.requires_grad_(True)
                overwatch.info(f"[TRAINABLE] 🔥 =>> Reason Token Contrastive Projection Heads", ctx_level=1)

    def _build_multimodal_data(self,
        input_embeddings: torch.Tensor,
        projected_patch_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        action_token_mask: torch.Tensor,
        reason_token_mask: torch.Tensor,
        ):
        multimodal_embeddings = torch.cat(
            [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
        )

        projected_patch_attention_mask = torch.full(
            (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
            fill_value=True,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        multimodal_attention_mask = torch.cat(
            [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
        )
        
        projected_patch_labels = torch.full(
            (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
            fill_value=IGNORE_INDEX,
            dtype=labels.dtype,
            device=labels.device,
        )
        multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
        
        multimodal_action_token_mask = torch.cat(
            [action_token_mask[:, :1], torch.full(
                (action_token_mask.shape[0], projected_patch_embeddings.shape[1], 1),
                fill_value=False,
                dtype=action_token_mask.dtype,
                device=action_token_mask.device,
            ), action_token_mask[:, 1:]], dim=1
        )
        multimodal_reason_token_mask = torch.cat(
            [reason_token_mask[:, :1], torch.full(
                (reason_token_mask.shape[0], projected_patch_embeddings.shape[1], 1),
                fill_value=False,
                dtype=reason_token_mask.dtype,
                device=reason_token_mask.device,
            ), reason_token_mask[:, 1:]], dim=1
        )

        return multimodal_embeddings, multimodal_attention_mask, multimodal_labels, multimodal_action_token_mask, multimodal_reason_token_mask

    def _replace_input_embeddings(
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

    def build_mixed_causal_bidi_mask(self, 
                                    pad_mask: torch.Tensor,
                                    bidi_mask: torch.Tensor,
                                    neg_inf: float = -1e9) -> torch.Tensor:
        """
        pad_mask: (B, L)  1 = real token, 0 = padding  (from tokenizer.attention_mask)
        bidi_mask: (B, L) 1 = this position has bidirectional attention as a query
                        0 = this position has causal attention as a query
        Returns:
            additive attention mask of shape (B, 1, L, L), float: 0 for allowed, neg_inf for disallowed.
        """
        assert pad_mask.shape == bidi_mask.shape
        B, L = pad_mask.shape
        device = pad_mask.device

        # ---- 1. Build base causal mask (same for all batch) ----
        idxs = torch.arange(L, device=device)
        i = idxs.unsqueeze(1)  # (L, 1) query positions
        j = idxs.unsqueeze(0)  # (1, L) key positions

        causal = (j <= i)  # (L, L) bool, lower-triangular
        causal = causal.unsqueeze(0).unsqueeze(1)  # (1, 1, L, L)\

        # Full bidirectional mask: everything allowed
        bidir = torch.ones(1, 1, L, L, dtype=torch.bool, device=device)  # (1, 1, L, L)

        # ---- 2. For each query position, choose causal vs bidirectional ----
        # bidi_mask: (B, L)
        # is_bidi_query[b, 0, i, 0] says whether query i in batch b is bidirectional
        is_bidi_query = bidi_mask.bool().unsqueeze(1).unsqueeze(3)  # (B, 1, L, 1)

        # mask_struct[b, 0, i, j] = bidir if is_bidi_query[b,i] else causal[i,j]
        mask_struct = torch.where(is_bidi_query, bidir, causal)     # (B, 1, L, L) bool

        # ---- 3. Apply padding: keys must be non-padding ----
        # pad_mask: 1 = real token, 0 = padding
        key_nonpad = pad_mask.bool().unsqueeze(1).unsqueeze(1)      # (B, 1, 1, L)

        allowed = mask_struct & key_nonpad                          # (B, 1, L, L) bool

        # ---- 4. Convert to additive attention mask ----
        attn_mask = torch.zeros(B, 1, L, L, device=device, dtype=torch.float32)
        attn_mask = attn_mask.masked_fill(~allowed, neg_inf)

        return attn_mask

    def _info_nce_loss(self, q: torch.Tensor, k: torch.Tensor, tau: float) -> torch.Tensor:
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        logits = torch.matmul(q, k.t()) / tau  # (B, B)
        labels = torch.arange(q.size(0), device=q.device)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        current_action: torch.Tensor,
        current_action_chunk: torch.Tensor,
        history_actions: torch.Tensor,
        pixel_values: Dict[str, torch.Tensor],
        pixel_values_history: Dict[str, torch.Tensor],
        proprio: Optional[torch.Tensor],
        proprio_history: Optional[torch.Tensor],
        pad_mask: torch.Tensor,
    ) -> MIRTHOutput:
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        current_action = current_action.to(self.device) if current_action is not None else None
        current_action_chunk = current_action_chunk.to(self.device) if current_action_chunk is not None else None
        history_actions = history_actions.to(self.device) if history_actions is not None else None
        pixel_values = {k: v.to(self.device) for k, v in pixel_values.items()}
        pixel_values_history = {k: v.to(self.device) for k, v in pixel_values_history.items()}
        proprio = proprio.to(self.device) if (proprio is not None and self.config.use_proprio) else None
        proprio_history = proprio_history.to(self.device) if (proprio_history is not None and self.config.use_proprio) else None
        pad_mask = pad_mask.to(self.device)

        action_token_mask = get_action_tokens_mask(labels).unsqueeze(-1)  # (B, seq_len, 1)
        reason_token_mask = get_reasoning_tokens_mask(labels).unsqueeze(-1)  # (B, seq_len, 1)

        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)  # (B, seq_len, D)

        # Vision Memory Hub and Infusion
        if self.config.use_vision_memory_hub:
            vision_return_dict = self.vision_memory_hub(pixel_values, pixel_values_history, self.vision_backbone, pad_mask)
        patch_embeddings = self.vision_backbone(pixel_values, None)  # (B, num_patches, D)
        projected_patch_embeddings = self.projector(patch_embeddings)  # (B, num_patches, D)
        
        # Proprioceptive Memory Hub
        if self.config.use_proprio:
            proprio = proprio.reshape(projected_patch_embeddings.shape[0], PROPRIO_DIM)  # (bsz, proprio_dim)
            if self.config.use_proprio_memory_hub:
                proprio_return_dict = self.proprio_memory_hub(proprio, proprio_history, pad_mask)
            proprio_features = self.proprio_projector(proprio)  # (bsz, llm_dim)
            proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)
            projected_patch_embeddings = torch.cat((projected_patch_embeddings, proprio_features), dim=1)
        
        # Action Memory Hub
        if self.config.use_action_memory_hub:
            action_return_dict = self.action_memory_hub(current_action, history_actions, pad_mask) # (B, action_dim)
        
        # Add timestamp embeddings to action tokens if enabled   
        if self.config.use_timestamp:
            time_embeddings = self.time_encoder(torch.arange(0, action_token_mask[0].sum().long().detach().item()).to(self.device))  # (action_len, D)
            time_embeddings = time_embeddings.unsqueeze(0).expand(input_ids.shape[0], -1, -1) # (B, action_len, D)
            input_embeddings = self._replace_input_embeddings(input_embeddings, action_token_mask.squeeze(-1), time_embeddings)
        else:
            input_embeddings = input_embeddings * ~action_token_mask

        if self.config.mb_prefix_type == "union":
            unioned_memory_embeddings = []
            if self.config.use_vision_memory_hub:
                unioned_memory_embeddings.append(self.vision_memory_fc(vision_return_dict['mem']))  # (B, num_patches, D)
            if self.config.use_proprio and self.config.use_proprio_memory_hub:
                unioned_memory_embeddings.append(self.proprio_memory_fc(proprio_return_dict['mem']).unsqueeze(1))  # (B, 1, D)
            if self.config.use_action_memory_hub:
                unioned_memory_embeddings.append(self.action_memory_fc(action_return_dict['mem']).unsqueeze(1))  # (B, 1, D)
            unioned_memory_embeddings = torch.cat(unioned_memory_embeddings, dim=1)  # (B, num_patches, D)
            projected_patch_embeddings = torch.cat((unioned_memory_embeddings, projected_patch_embeddings), dim=1)  # (B, 2*num_patches, D)
        elif self.config.mb_prefix_type == "separate":
            long_memory_embeddings = []
            short_memory_embeddings = []
            if self.config.use_vision_memory_hub:
                long_memory_embeddings.append(self.vision_memory_fc_long(vision_return_dict['workspace']))  # (B, num_patches, D)
                short_memory_embeddings.append(self.vision_memory_fc_short(vision_return_dict['short']))  # (B, num_patches, D)
            if self.config.use_proprio and self.config.use_proprio_memory_hub:
                long_memory_embeddings.append(self.proprio_memory_fc_long(proprio_return_dict['workspace']).unsqueeze(1))  # (B, 1, D)
                short_memory_embeddings.append(self.proprio_memory_fc_short(proprio_return_dict['short']).unsqueeze(1))  # (B, 1, D)
            if self.config.use_action_memory_hub:
                long_memory_embeddings.append(self.action_memory_fc_long(action_return_dict['workspace']).unsqueeze(1))  # (B, 1, D)
                short_memory_embeddings.append(self.action_memory_fc_short(action_return_dict['short']).unsqueeze(1))  # (B, 1, D)
            long_memory_embeddings = torch.cat(long_memory_embeddings, dim=1)  # (B, num_patches, D)
            short_memory_embeddings = torch.cat(short_memory_embeddings, dim=1)  # (B, num_patches, D)   
            projected_patch_embeddings = torch.cat((long_memory_embeddings, short_memory_embeddings, projected_patch_embeddings), dim=1)  # (B, 3*num_patches, D)
        else:
            raise ValueError(f"Unknown memory hub prefix type `{self.config.mb_prefix_type}`")

        # Build multimodal embeddings & attention mask & labels
        multimodal_embeddings, multimodal_attention_mask, multimodal_labels, multimodal_action_token_mask, multimodal_reason_token_mask = self._build_multimodal_data(
            input_embeddings=input_embeddings,
            projected_patch_embeddings=projected_patch_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            action_token_mask=action_token_mask,
            reason_token_mask=reason_token_mask,
        )
        
        # Insert action reason tokens
        if self.config.use_reason_token:
            action_reason_embeddings = self.reason_token_encoder(multimodal_embeddings, multimodal_attention_mask)  # (B, M, D)
            multimodal_embeddings = self._replace_input_embeddings(multimodal_embeddings, multimodal_reason_token_mask.squeeze(-1), action_reason_embeddings)

        # Build mixed causal + bidirectional attention mask if action_biattnn is enabled
        if self.config.action_biattnn:
            # Build mixed causal + bidirectional attention mask
            multimodal_attention_mask = self.build_mixed_causal_bidi_mask(
                pad_mask=multimodal_attention_mask,
                bidi_mask=multimodal_action_token_mask.squeeze(-1),
                neg_inf=-1e9,
            )  # (B, 1, L, L)

        # Dispatch to language model
        output = self.llm_backbone(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=multimodal_labels,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        
        metrics_return: Dict[str, float] = {}

        # Get action hidden states
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        actions_hidden_states = last_hidden_states[multimodal_action_token_mask.squeeze(-1)] # (B, action_len, D)
        if self.config.action_token_type == "one_for_action_dim":
            actions_hidden_states = actions_hidden_states.reshape(input_ids.shape[0], NUM_ACTIONS_CHUNK * ACTION_DIM, self.llm_backbone.embed_dim)
        elif self.config.action_token_type == "one_for_action_chunk":
            actions_hidden_states = actions_hidden_states.reshape(input_ids.shape[0], 1, self.llm_backbone.embed_dim)
        elif self.config.action_token_type == "one_for_action_step":
            actions_hidden_states = actions_hidden_states.reshape(input_ids.shape[0], NUM_ACTIONS_CHUNK, self.llm_backbone.embed_dim)
        else:
            raise ValueError(f"Unknown action token type `{self.config.action_token_type}`")

        # Predict actions and compute loss
        predicted_actions = self.action_head.predict_action(actions_hidden_states)
        loss = self.action_prediction_loss_fn(current_action_chunk, predicted_actions)
        
        # Compute contrastive loss if enabled
        if self.config.use_reason_token and self.config.use_contrastive_loss:
            reason_mask_bool = multimodal_reason_token_mask.squeeze(-1).bool()  # (B, L)
            reason_states = last_hidden_states[reason_mask_bool].view(input_ids.shape[0], self.config.num_reason_token, self.llm_backbone.embed_dim)  # (B, M, D)
            reason_repr = reason_states.mean(dim=1)  # (B, D)
            reason_repr_proj = self.reason_proj(reason_repr)  # (B, D)

            action_mask_bool = multimodal_action_token_mask.squeeze(-1).bool()  # (B, L)
            action_count = int(action_mask_bool.sum(dim=1)[0].item())
            action_states = last_hidden_states[action_mask_bool].view(input_ids.shape[0], action_count, self.llm_backbone.embed_dim)  # (B, K, D)
            action_repr = action_states.mean(dim=1)  # (B, D)
            action_repr_proj = self.action_proj(action_repr)  # (B, D)
            contrastive_loss_ra = self._info_nce_loss(reason_repr_proj, action_repr_proj, self.config.contrastive_tau_ra)

            reason_mask_with_fallback = torch.cat([reason_mask_bool, torch.ones(input_ids.shape[0], 1, dtype=torch.bool, device=input_ids.device)], dim=1)
            first_reason_pos = torch.argmax(reason_mask_with_fallback.int(), dim=1)  # [batch_size]
            has_reason_token = reason_mask_bool.any(dim=1)  # [batch_size]
            first_reason_pos = torch.where(has_reason_token, first_reason_pos, torch.zeros_like(first_reason_pos))
            extract_pos = torch.clamp(first_reason_pos - 1, min=0)  # [batch_size]
            batch_indices = torch.arange(input_ids.shape[0], device=last_hidden_states.device)
            input_repr = last_hidden_states[batch_indices, extract_pos, :]
            input_repr_proj = self.input_proj(input_repr)
            contrastive_loss_rx = self._info_nce_loss(reason_repr_proj, input_repr_proj, self.config.contrastive_tau_rx)
            contrastive_loss = self.config.lambda_contrastive_ra * contrastive_loss_ra + self.config.lambda_contrastive_rx * contrastive_loss_rx

            loss = loss + contrastive_loss
            metrics_return['contrastive_loss'] = contrastive_loss.item()
            metrics_return['contrastive_loss_ra'] = contrastive_loss_ra.item()
            metrics_return['contrastive_loss_rx'] = contrastive_loss_rx.item()
        
        # Update metrics
        metrics_return['loss_value'] = loss.item()
        ground_truth_curr_action = current_action_chunk[:, 0]
        ground_truth_next_actions = current_action_chunk[:, 1:]
        predicted_curr_action = predicted_actions[:, 0]
        predicted_next_actions = predicted_actions[:, 1:]
        curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
        next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
        metrics_return['current_action_l1_loss'] = curr_action_l1_loss.item()
        metrics_return['next_actions_l1_loss'] = next_actions_l1_loss.item()

        return MIRTHOutput(
            loss=loss,
            logits=output.logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            metrics=metrics_return,
            actions=predicted_actions,
            
            x_embeddings=input_repr_proj if self.config.use_reason_token and self.config.use_contrastive_loss else None,
            r_embeddings=reason_repr_proj if self.config.use_reason_token and self.config.use_contrastive_loss else None,
            a_embeddings=action_repr_proj if self.config.use_reason_token and self.config.use_contrastive_loss else None,
        )

    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        """Unnormalize actions using dataset statistics"""
        action_norm_stats = self.norm_stats[unnorm_key]["action"]

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        actions = np.where(mask, 0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low, normalized_actions)

        return actions

    def predict_action(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        current_action: torch.Tensor,
        current_action_chunk: torch.Tensor,
        history_actions: torch.Tensor,
        pixel_values: Dict[str, torch.Tensor],
        pixel_values_history: Dict[str, torch.Tensor],
        proprio: Optional[torch.Tensor],
        proprio_history: Optional[torch.Tensor],
        pad_mask: torch.Tensor,
        unnorm_key: Optional[str] = None,
    ) -> np.ndarray:
        
        assert 29871 in input_ids[0], "Input ids must contain at least one action token!"
        
        outputs: MIRTHOutput = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            current_action=current_action.to(torch.bfloat16),
            current_action_chunk=current_action_chunk.to(torch.bfloat16),
            history_actions=history_actions.to(torch.bfloat16),
            pixel_values=pixel_values,
            pixel_values_history=pixel_values_history,
            proprio=proprio.to(torch.bfloat16),
            proprio_history=proprio_history.to(torch.bfloat16),
            pad_mask=pad_mask,
        )
        predicted_actions = outputs.actions
        
        predicted_actions = predicted_actions.to(torch.float32).cpu().numpy()
        predicted_actions = np.clip(predicted_actions, -1.0, 1.0)
        actions = self._unnormalize_actions(predicted_actions, unnorm_key)
        
        if outputs.x_embeddings is not None and outputs.r_embeddings is not None and outputs.a_embeddings is not None:
            return actions, predicted_actions, outputs.x_embeddings.cpu(), outputs.r_embeddings.cpu(), outputs.a_embeddings.cpu()
        return actions, predicted_actions, None, None, None