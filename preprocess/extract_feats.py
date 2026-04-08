"""
Feature extraction for RSIC-Count dataset.
Extracts CNN features (fc and attention) for all images.
Supports configurable backbones via config.yaml.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import yaml


# Backbone configurations: (model_fn, feature_dim, spatial_dim)
BACKBONE_CONFIGS = {
    # ResNet family
    'resnet50': (lambda: models.resnet50(pretrained=True), 2048, 7),
    'resnet101': (lambda: models.resnet101(pretrained=True), 2048, 7),
    'resnet152': (lambda: models.resnet152(pretrained=True), 2048, 7),
    
    # DenseNet family
    'densenet121': (lambda: models.densenet121(pretrained=True), 1024, 7),
    'densenet161': (lambda: models.densenet161(pretrained=True), 2208, 7),
    'densenet169': (lambda: models.densenet169(pretrained=True), 1664, 7),
    'densenet201': (lambda: models.densenet201(pretrained=True), 1920, 7),
    
    # EfficientNet family (requires torchvision >= 0.11)
    'efficientnet_b0': (lambda: models.efficientnet_b0(pretrained=True), 1280, 7),
    'efficientnet_b1': (lambda: models.efficientnet_b1(pretrained=True), 1280, 7),
    'efficientnet_b2': (lambda: models.efficientnet_b2(pretrained=True), 1408, 7),
    'efficientnet_b3': (lambda: models.efficientnet_b3(pretrained=True), 1536, 7),
    'efficientnet_b4': (lambda: models.efficientnet_b4(pretrained=True), 1792, 7),
    'efficientnet_b5': (lambda: models.efficientnet_b5(pretrained=True), 2048, 7),
    'efficientnet_b6': (lambda: models.efficientnet_b6(pretrained=True), 2304, 7),
    'efficientnet_b7': (lambda: models.efficientnet_b7(pretrained=True), 2560, 7),
    
    # ConvNeXt family (requires torchvision >= 0.13)
    'convnext_tiny': (lambda: models.convnext_tiny(pretrained=True), 768, 7),
    'convnext_small': (lambda: models.convnext_small(pretrained=True), 768, 7),
    'convnext_base': (lambda: models.convnext_base(pretrained=True), 1024, 7),
    
    # ResNeXt family (requires torchvision >= 0.12)
    'resnext50_32x4d': (lambda: models.resnext50_32x4d(pretrained=True), 2048, 7),
    'resnext101_32x8d': (lambda: models.resnext101_32x8d(pretrained=True), 2048, 7),
    'resnext101_64x4d': (lambda: models.resnext101_64x4d(pretrained=True), 2048, 7),
    
    # Swin Transformer family (requires torchvision >= 0.12)
    'swin_t': (lambda: models.swin_t(pretrained=True), 768, 7),
    'swin_s': (lambda: models.swin_s(pretrained=True), 768, 7),
    'swin_b': (lambda: models.swin_b(pretrained=True), 1024, 7),
    
    # Vision Transformer (ViT) family (requires torchvision >= 0.13)
    'vit_b_16': (lambda: models.vit_b_16(pretrained=True), 768, 14),
    'vit_b_32': (lambda: models.vit_b_32(pretrained=True), 768, 7),
    'vit_l_16': (lambda: models.vit_l_16(pretrained=True), 1024, 14),
    'vit_l_32': (lambda: models.vit_l_32(pretrained=True), 1024, 7),
}


def get_backbone_info(name):
    """Get backbone configuration"""
    if name not in BACKBONE_CONFIGS:
        raise ValueError(f"Unknown backbone: {name}. Supported: {list(BACKBONE_CONFIGS.keys())}")
    return BACKBONE_CONFIGS[name]


class FeatureExtractor:
    def __init__(self, backbone_name='resnet50', device='cuda'):
        self.device = device
        self.backbone_name = backbone_name
        
        # Get backbone config
        model_fn, self.feature_dim, self.spatial_dim = get_backbone_info(backbone_name)
        
        # Load pretrained model
        print(f"Loading {backbone_name} (dim={self.feature_dim}, spatial={self.spatial_dim}x{self.spatial_dim})...")
        backbone = model_fn()
        
        # Create encoders based on architecture type
        if 'resnet' in backbone_name:
            # ResNet: remove final FC and avgpool
            self.att_encoder = nn.Sequential(*list(backbone.children())[:-2])
            self.fc_encoder = nn.Sequential(*list(backbone.children())[:-1])
            
        elif 'densenet' in backbone_name:
            # DenseNet: features are in features layer, classifier is separate
            self.att_encoder = backbone.features
            # For FC, add adaptive pooling
            self.fc_encoder = nn.Sequential(
                backbone.features,
                nn.AdaptiveAvgPool2d(1)
            )
            
        elif 'efficientnet' in backbone_name:
            # EfficientNet: features and classifier structure
            self.att_encoder = backbone.features
            self.fc_encoder = nn.Sequential(
                backbone.features,
                nn.AdaptiveAvgPool2d(1)
            )
            
        elif 'convnext' in backbone_name:
            # ConvNeXt: features structure
            self.att_encoder = backbone.features
            self.fc_encoder = nn.Sequential(
                backbone.features,
                nn.AdaptiveAvgPool2d(1)
            )
            
        elif 'resnext' in backbone_name:
            # ResNeXt: same structure as ResNet
            self.att_encoder = nn.Sequential(*list(backbone.children())[:-2])
            self.fc_encoder = nn.Sequential(*list(backbone.children())[:-1])
            
        elif 'swin' in backbone_name:
            # Swin Transformer: outputs [B, H, W, C] permuted to [B, C, H, W]
            # Swin-B features output [B, 7, 7, 1024] - need to permute to [B, 1024, 7, 7]
            class SwinFeatureWrapper(nn.Module):
                def __init__(self, features, feat_dim):
                    super().__init__()
                    self.features = features
                    self.feat_dim = feat_dim
                def forward(self, x):
                    x = self.features(x)
                    # Swin outputs [B, H, W, C], permute to [B, C, H, W] for consistency
                    # Check if last dim matches feat_dim (indicating [B, H, W, C] format)
                    if x.dim() == 4 and x.shape[-1] == self.feat_dim:
                        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
                    return x
            
            self.att_encoder = SwinFeatureWrapper(backbone.features, self.feature_dim)
            self.fc_encoder = nn.Sequential(
                SwinFeatureWrapper(backbone.features, self.feature_dim),
                nn.AdaptiveAvgPool2d(1)
            )
            
        elif 'vit' in backbone_name:
            # Vision Transformer: special handling needed
            self.att_encoder = backbone
            self.fc_encoder = backbone
            # ViT outputs are different - we'll handle in extract()
        
        self.att_encoder = self.att_encoder.to(device).eval()
        self.fc_encoder = self.fc_encoder.to(device).eval()
        
        # Image preprocessing (standard ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def extract(self, image_path, augment_type=None):
        """Extract both attention and FC features
        
        Args:
            image_path: Path to image
            augment_type: None, 'hflip', 'vflip', 'rot90', 'rot180', 'rot270'
        """
        img = Image.open(image_path).convert('RGB')
        
        # Apply augmentation
        if augment_type == 'hflip':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif augment_type == 'vflip':
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        elif augment_type == 'rot90':
            img = img.transpose(Image.ROTATE_90)
        elif augment_type == 'rot180':
            img = img.transpose(Image.ROTATE_180)
        elif augment_type == 'rot270':
            img = img.transpose(Image.ROTATE_270)
        
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Attention features: [1, feature_dim, spatial_dim, spatial_dim]
        att_feat = self.att_encoder(img_tensor)
        
        # FC features: [1, feature_dim]
        fc_feat = self.fc_encoder(img_tensor)
        
        # Handle Vision Transformer outputs (special case)
        if 'vit' in self.backbone_name:
            # ViT outputs [batch, num_patches+1, hidden_dim] where first token is class token
            # Extract class token for FC features
            if fc_feat.dim() == 2 and fc_feat.shape[1] > self.feature_dim:
                # Take the class token (first token)
                fc_feat = fc_feat[:, 0]
            # For attention features, reshape patches to spatial grid
            if att_feat.dim() == 2 and att_feat.shape[1] > self.feature_dim:
                # Remove class token, reshape remaining patches
                patches = att_feat[:, 1:]  # Skip class token
                # Reshape to [1, feature_dim, spatial, spatial]
                grid_size = int(patches.shape[1] ** 0.5)
                patches = patches.transpose(1, 2)  # [1, feature_dim, num_patches]
                att_feat = patches.view(1, self.feature_dim, grid_size, grid_size)
        
        # Handle different output shapes
        if fc_feat.dim() == 4:
            fc_feat = fc_feat.squeeze(-1).squeeze(-1)
        elif fc_feat.dim() == 2:
            pass  # Already [1, feature_dim]
        else:
            raise ValueError(f"Unexpected FC feature shape: {fc_feat.shape}")
        
        return att_feat.cpu().numpy(), fc_feat.cpu().numpy()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Extract CNN features")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config.yaml')
    parser.add_argument('--images_dir', type=str, default=None,
                        help='Directory containing images (overrides config)')
    parser.add_argument('--captions', type=str, default=None,
                        help='Path to captions.json (overrides config)')
    parser.add_argument('--att_output', type=str, default=None,
                        help='Output directory for attention features (overrides config)')
    parser.add_argument('--fc_output', type=str, default=None,
                        help='Output directory for FC features (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cuda or cpu (overrides config)')
    parser.add_argument('--backbone', type=str, default=None,
                        help=f'Backbone model (overrides config). Options: {list(BACKBONE_CONFIGS.keys())}')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get values from args or config
    backbone_name = args.backbone or config.get('backbone', {}).get('name', 'resnet50')
    device = args.device or config.get('system', {}).get('device', 'cuda')
    images_dir = args.images_dir or config.get('data', {}).get('images_dir', 'data/images')
    captions_path = args.captions or config.get('data', {}).get('captions', 'data/captions.json')
    att_output = args.att_output or config.get('data', {}).get('att_features', 'data/att_features')
    fc_output = args.fc_output or config.get('data', {}).get('fc_features', 'data/fc_features')
    
    # Validate backbone
    if backbone_name not in BACKBONE_CONFIGS:
        print(f"ERROR: Unknown backbone '{backbone_name}'")
        print(f"Supported backbones: {list(BACKBONE_CONFIGS.keys())}")
        return
    
    # Create output directories
    Path(att_output).mkdir(parents=True, exist_ok=True)
    Path(fc_output).mkdir(parents=True, exist_ok=True)
    
    # Load image list from captions
    with open(captions_path, 'r') as f:
        captions_data = json.load(f)
    
    # Get unique original image IDs from captions
    if isinstance(captions_data, list):
        image_ids = []
        for item in captions_data:
            if 'filename' in item:
                img_id = item['filename'].replace('.png', '').replace('.jpg', '')
                if img_id not in image_ids:
                    image_ids.append(img_id)
    else:
        raise ValueError("Unsupported captions.json format - expected list")
    
    # Initialize extractor
    extractor = FeatureExtractor(backbone_name=backbone_name, device=device)
    
    # Define augmentation types
    aug_types = [None, 'hflip', 'vflip', 'rot90', 'rot180', 'rot270']
    aug_suffixes = ['', '_hflip', '_vflip', '_rot90', '_rot180', '_rot270']
    
    print(f"\n{'='*60}")
    print(f"Feature Extraction with {backbone_name.upper()} + FULL AUGMENTATION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Feature dimension: {extractor.feature_dim}")
    print(f"Spatial size: {extractor.spatial_dim}x{extractor.spatial_dim}")
    print(f"Original images: {len(image_ids)}")
    print(f"Augmentations per image: {len(aug_types)}")
    print(f"Total feature sets: {len(image_ids) * len(aug_types)}")
    print(f"Augmentations: {aug_suffixes}")
    print(f"{'='*60}\n")
    
    processed = 0
    skipped = 0
    
    for img_id in tqdm(image_ids, desc="Processing images"):
        # Try different image extensions
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            test_path = Path(images_dir) / f"{img_id}{ext}"
            if test_path.exists():
                img_path = test_path
                break
        
        if img_path is None:
            print(f"\nWarning: Image {img_id} not found, skipping...")
            skipped += 1
            continue
        
        # Extract features for all augmentations
        for aug_type, aug_suffix in zip(aug_types, aug_suffixes):
            try:
                att_feat, fc_feat = extractor.extract(str(img_path), augment_type=aug_type)
                
                # Save features with augmentation suffix
                aug_id = f"{img_id}{aug_suffix}"
                np.save(Path(att_output) / f"{aug_id}.npy", att_feat)
                np.save(Path(fc_output) / f"{aug_id}.npy", fc_feat)
                processed += 1
            except Exception as e:
                print(f"\nError processing {img_id} with {aug_type}: {e}")
                skipped += 1
                continue
    
    print(f"\n{'='*60}")
    print(f"Feature extraction complete!")
    print(f"  Processed: {processed} feature sets")
    print(f"  Skipped: {skipped}")
    print(f"  Attention features: {att_output}")
    print(f"  FC features: {fc_output}")
    print(f"  Backbone: {backbone_name}")
    print(f"  Feature dim: {extractor.feature_dim}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
