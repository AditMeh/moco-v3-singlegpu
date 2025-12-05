"""
Multiple Instance Learning (MIL) module for patch-based image classification.

Takes a pretrained backbone, extracts patch embeddings, and uses attention
pooling to aggregate them for classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Attention-based pooling to aggregate patch embeddings into a single vector.
    Uses a small attention network to compute attention weights over patches.
    """
    def __init__(self, embed_dim, hidden_dim=512):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, num_patches, embed_dim)
        Returns:
            pooled: (B, embed_dim)
            attention_weights: (B, num_patches)
        """
        # Compute attention scores
        attn_scores = self.attention(x)  # (B, num_patches, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, num_patches, 1)
        
        # Weighted sum of patch embeddings
        pooled = torch.sum(x * attn_weights, dim=1)  # (B, embed_dim)
        
        return pooled, attn_weights.squeeze(-1)


class MILClassifier(nn.Module):
    """
    Multiple Instance Learning classifier that:
    1. Splits input image into patches
    2. Embeds each patch using a frozen backbone
    3. Aggregates patch embeddings using attention
    4. Classifies using a small MLP
    """
    def __init__(self, backbone, embed_dim=2048, num_classes=10, 
                 patch_size=32, input_size=224, attention_hidden_dim=512,
                 mlp_hidden_dim=512, projection_dim=512):
        """
        Args:
            backbone: Pretrained model with fc/head removed (outputs embed_dim features)
            embed_dim: Dimension of backbone output (2048 for ResNet50)
            num_classes: Number of output classes
            patch_size: Size of each patch (patches are patch_size x patch_size)
            input_size: Size to resize input images to (input_size x input_size)
            attention_hidden_dim: Hidden dimension for attention network
            mlp_hidden_dim: Hidden dimension for classification MLP
            projection_dim: Dimension to project patch embeddings to before attention
        """
        super().__init__()
        
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.input_size = input_size
        
        # Calculate number of patches
        assert input_size % patch_size == 0, \
            f"input_size ({input_size}) must be divisible by patch_size ({patch_size})"
        self.num_patches_per_side = input_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Patch embedding projector (projects from embed_dim to projection_dim)
        self.patch_projector = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # Attention pooling (operates on projected embeddings)
        self.attention_pool = AttentionPooling(projection_dim, attention_hidden_dim)
        
        # Classification MLP (takes projected/pooled embeddings)
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, num_classes)
        )
        
        # Initialize classifier weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.patch_projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        for m in self.attention_pool.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def extract_patches(self, x):
        """
        Extract non-overlapping patches from input images.
        
        Args:
            x: (B, C, H, W) input images, assumed to be input_size x input_size
        Returns:
            patches: (B * num_patches, C, patch_size, patch_size)
        """
        B, C, H, W = x.shape
        
        # Use unfold to extract patches
        # unfold(dimension, size, step)
        patches = x.unfold(2, self.patch_size, self.patch_size)  # (B, C, n_h, W, patch_size)
        patches = patches.unfold(3, self.patch_size, self.patch_size)  # (B, C, n_h, n_w, patch_size, patch_size)
        
        # Reshape to (B, num_patches, C, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, self.num_patches, C, self.patch_size, self.patch_size)
        
        # Reshape to (B * num_patches, C, patch_size, patch_size) for backbone
        patches = patches.view(B * self.num_patches, C, self.patch_size, self.patch_size)
        
        return patches
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, C, H, W) input images
        Returns:
            logits: (B, num_classes) classification logits
        """
        B = x.shape[0]
        
        # Extract patches: (B * num_patches, C, patch_size, patch_size)
        patches = self.extract_patches(x)
        
        # Resize patches to backbone expected size (224x224 for standard ResNet)
        # The backbone was trained on 224x224 images
        if self.patch_size != 224:
            patches = F.interpolate(patches, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Get patch embeddings from frozen backbone
        with torch.no_grad():
            self.backbone.eval()
            patch_embeddings = self.backbone(patches)  # (B * num_patches, embed_dim)
        
        # Reshape to (B, num_patches, embed_dim)
        patch_embeddings = patch_embeddings.view(B, self.num_patches, self.embed_dim)
        
        # Project patch embeddings
        patch_embeddings = self.patch_projector(patch_embeddings)  # (B, num_patches, projection_dim)
        
        # Attention pooling
        pooled, attn_weights = self.attention_pool(patch_embeddings)  # (B, projection_dim)
        
        # Classification
        logits = self.classifier(pooled)  # (B, num_classes)
        
        return logits
    
    def get_trainable_parameters(self):
        """Return only trainable parameters (projector + attention + classifier)."""
        params = []
        params.extend(self.patch_projector.parameters())
        params.extend(self.attention_pool.parameters())
        params.extend(self.classifier.parameters())
        return params


def create_backbone(model, arch='resnet50'):
    """
    Remove the final classification layer from a model to use as backbone.
    
    Args:
        model: Full model (e.g., ResNet50)
        arch: Architecture name to determine which layer to remove
    
    Returns:
        backbone: Model without final classification layer
        embed_dim: Dimension of backbone output
    """
    if arch.startswith('vit'):
        # For ViT, remove the head
        embed_dim = model.head.in_features
        model.head = nn.Identity()
    else:
        # For ResNet, replace fc with identity
        embed_dim = model.fc.in_features
        model.fc = nn.Identity()
    
    return model, embed_dim
