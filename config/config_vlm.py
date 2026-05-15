from dataclasses import dataclass

@dataclass
class Prism_7B_DINOSigLIP_224px():    
    vision_backbone_id: str = "dinosiglip-vit-so-224px"
    default_image_size: int = 224
    dino_url: str = "vit_large_patch14_reg4_dinov2.lvd142m"
    siglip_url: str = "vit_so400m_patch14_siglip_224"

    llm_backbone_hf_id: str = "meta-llama/Llama-2-7b-hf"
    llm_max_length: int = 2048
    
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision_training: bool = True
    reduce_in_full_precision: bool = False

