from models.vision_encoders import get_Prism_7B_DINOSigLIP_224px_backbone_and_transform
from models.infuse_wrapper import InfusedDinoSigLIPViTBackbone
import torch

# For images, we must use the square size from cameras
print('=' * 60)
print('Testing Vision Encoders')
print('=' * 60)
B = 2
vision_encoder, image_transform = get_Prism_7B_DINOSigLIP_224px_backbone_and_transform()
vision_encoder = vision_encoder.cuda()
print(vision_encoder.default_image_resolution, vision_encoder.embed_dim, vision_encoder.num_patches)

pixel_values = {'dino': torch.ones(B, 3, 224, 224).cuda(), 'siglip': torch.ones(B, 3, 224, 224).cuda()}
model_output = vision_encoder(pixel_values)
print("Model output shape:", model_output.shape)  # Expected: (B, num_patches, embed_dim)

print('=' * 60)
print('Testing Infusion')
print('=' * 60)
num_cameras = 2
infused_vision_encoder = InfusedDinoSigLIPViTBackbone(vision_encoder, infuse=True, infuse_layers_ratio=0.25).cuda()
pixel_values = {'dino': torch.ones(B, num_cameras, 3, 224, 224).cuda(), 'siglip': torch.ones(B, num_cameras, 3, 224, 224).cuda()}
fused_memory = torch.randn(B, num_cameras, vision_encoder.num_patches, vision_encoder.embed_dim).cuda()
combined_patches = infused_vision_encoder(pixel_values, fused_memory=fused_memory)
print("Combined patches shape:", combined_patches.shape)  # Expected: (B, num_patches, embed_dim)