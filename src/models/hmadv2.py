#!/usr/bin/env python3
"""
HMAD Framework - Improved Version
Addressing the noise/random output issues in the original implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Dict, List, Optional

class SimplifiedHMADFramework(nn.Module):
    """
    Simplified HMAD Framework that addresses the noise output issues
    Key improvements:
    1. Progressive training strategy
    2. Better loss balancing
    3. Simplified architecture for better convergence
    4. Proper weight initialization
    """
    
    def __init__(self, 
                 mindbigdata_channels: int = 14,
                 crell_channels: int = 64,
                 d_model: int = 256,
                 image_size: int = 64):
        super().__init__()
        
        self.d_model = d_model
        self.image_size = image_size
        
        # Dataset-specific input projections with proper initialization
        self.mindbig_projection = nn.Sequential(
            nn.Conv1d(mindbigdata_channels, d_model//4, kernel_size=1),
            nn.BatchNorm1d(d_model//4),
            nn.ReLU(),
            nn.Conv1d(d_model//4, d_model//2, kernel_size=1),
            nn.BatchNorm1d(d_model//2),
            nn.ReLU()
        )
        
        self.crell_projection = nn.Sequential(
            nn.Conv1d(crell_channels, d_model//4, kernel_size=1), 
            nn.BatchNorm1d(d_model//4),
            nn.ReLU(),
            nn.Conv1d(d_model//4, d_model//2, kernel_size=1),
            nn.BatchNorm1d(d_model//2),
            nn.ReLU()
        )
        
        # Simplified feature extraction
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(d_model//2, d_model, kernel_size=16, stride=4, padding=8),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=8, stride=2, padding=4),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)  # Fixed output length
        )
        
        # Self-attention for sequence modeling
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=8, 
            batch_first=True,
            dropout=0.1
        )
        
        # Feature fusion and projection to image latent space
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model//2, d_model//4),
            nn.ReLU(),
            nn.Linear(d_model//4, 256)  # Latent space
        )
        
        # Simple but effective image decoder
        self.image_decoder = SimpleImageDecoder(latent_dim=256, image_size=image_size)
        
        # Loss components with learnable weights
        self.mse_weight = nn.Parameter(torch.tensor(1.0))
        self.perceptual_weight = nn.Parameter(torch.tensor(0.1))
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Proper weight initialization to prevent gradient issues"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, eeg_data: torch.Tensor, dataset_type: str, 
                target_images: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with improved stability
        """
        batch_size = eeg_data.shape[0]
        
        # Dataset-specific preprocessing
        if dataset_type == 'mindbigdata':
            x = self.mindbig_projection(eeg_data)
        else:  # crell
            x = self.crell_projection(eeg_data)
        
        # Temporal feature extraction
        temporal_features = self.temporal_encoder(x)  # (batch, d_model, seq_len)
        
        # Transpose for attention: (batch, seq_len, d_model)
        temporal_features = temporal_features.transpose(1, 2)
        
        # Self-attention
        attended_features, attention_weights = self.self_attention(
            temporal_features, temporal_features, temporal_features
        )
        
        # Global average pooling
        pooled_features = attended_features.mean(dim=1)  # (batch, d_model)
        
        # Feature fusion to latent space
        latent_features = self.feature_fusion(pooled_features)
        
        # Image generation
        generated_images = self.image_decoder(latent_features)
        
        outputs = {
            'generated_images': generated_images,
            'latent_features': latent_features,
            'attention_weights': attention_weights
        }
        
        # Compute losses if targets provided
        if target_images is not None:
            losses = self.compute_losses(generated_images, target_images)
            outputs.update(losses)
        
        return outputs
    
    def compute_losses(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Improved loss computation with better balance"""
        
        # MSE Loss
        mse_loss = F.mse_loss(generated, target)
        
        # Perceptual loss (simplified - comparing feature statistics)
        gen_mean = generated.mean(dim=[2, 3])
        target_mean = target.mean(dim=[2, 3])
        gen_std = generated.std(dim=[2, 3])
        target_std = target.std(dim=[2, 3])
        
        perceptual_loss = F.mse_loss(gen_mean, target_mean) + F.mse_loss(gen_std, target_std)
        
        # Total loss with clamping to prevent explosion
        total_loss = (torch.clamp(self.mse_weight, 0.1, 10.0) * mse_loss + 
                     torch.clamp(self.perceptual_weight, 0.01, 1.0) * perceptual_loss)
        
        return {
            'mse_loss': mse_loss,
            'perceptual_loss': perceptual_loss,
            'total_loss': total_loss
        }

class SimpleImageDecoder(nn.Module):
    """Simple but effective image decoder"""
    
    def __init__(self, latent_dim: int = 256, image_size: int = 64):
        super().__init__()
        
        self.image_size = image_size
        
        # Calculate initial spatial size
        init_size = image_size // 8  # 8x8 for 64x64 output
        
        # Linear projection to initial feature map
        self.fc = nn.Linear(latent_dim, 512 * init_size * init_size)
        
        # Upsampling layers
        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 16x16 -> 32x32  
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Final layer
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
        self.init_size = init_size
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (batch_size, latent_dim)
        Returns:
            images: (batch_size, 3, image_size, image_size)
        """
        batch_size = latent.shape[0]
        
        # Project to initial feature map
        x = self.fc(latent)
        x = x.view(batch_size, 512, self.init_size, self.init_size)
        
        # Decode to image
        x = self.decoder(x)
        
        return x

class ProgressiveTrainer:
    """Progressive training strategy to improve convergence"""
    
    def __init__(self, model: SimplifiedHMADFramework, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Different optimizers for different training phases
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=1e-4, 
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.training_phase = 'reconstruction'  # Start with basic reconstruction
        self.phase_epochs = {'reconstruction': 50, 'fine_tuning': 50}
        
    def train_step(self, eeg_data: torch.Tensor, target_images: torch.Tensor, 
                   dataset_type: str) -> Dict[str, float]:
        """Single training step with improved stability"""
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(eeg_data, dataset_type, target_images)
        
        # Get loss based on training phase
        if self.training_phase == 'reconstruction':
            # Focus on basic reconstruction first
            loss = outputs['mse_loss']
        else:
            # Use full loss in fine-tuning phase
            loss = outputs['total_loss']
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'mse_loss': outputs['mse_loss'].item(),
            'perceptual_loss': outputs.get('perceptual_loss', torch.tensor(0.0)).item()
        }
    
    def validate(self, eeg_data: torch.Tensor, target_images: torch.Tensor,
                dataset_type: str) -> Dict[str, float]:
        """Validation step"""
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(eeg_data, dataset_type, target_images)
            
            # Compute similarity metrics
            generated = outputs['generated_images']
            
            # Structural similarity (simplified)
            mse = F.mse_loss(generated, target_images)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            
            # Cosine similarity
            gen_flat = generated.view(generated.shape[0], -1)
            target_flat = target_images.view(target_images.shape[0], -1)
            cosine_sim = F.cosine_similarity(gen_flat, target_flat, dim=1).mean()
            
            return {
                'val_loss': outputs['total_loss'].item(),
                'psnr': psnr.item(),
                'cosine_similarity': cosine_sim.item(),
                'mse': mse.item()
            }

# Debugging and improvement utilities
def analyze_model_outputs(model: SimplifiedHMADFramework, eeg_data: torch.Tensor, 
                         dataset_type: str) -> Dict:
    """Analyze model outputs for debugging"""
    
    model.eval()
    with torch.no_grad():
        # Hook to capture intermediate outputs
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = {
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'min': output.min().item(),
                        'max': output.max().item(),
                        'shape': output.shape
                    }
            return hook
        
        # Register hooks
        if dataset_type == 'mindbigdata':
            model.mindbig_projection.register_forward_hook(hook_fn('projection'))
        else:
            model.crell_projection.register_forward_hook(hook_fn('projection'))
            
        model.temporal_encoder.register_forward_hook(hook_fn('temporal'))
        model.feature_fusion.register_forward_hook(hook_fn('fusion'))
        model.image_decoder.register_forward_hook(hook_fn('decoder'))
        
        # Forward pass
        outputs = model(eeg_data, dataset_type)
        
        return {
            'activations': activations,
            'output_stats': {
                'generated_mean': outputs['generated_images'].mean().item(),
                'generated_std': outputs['generated_images'].std().item(),
                'generated_min': outputs['generated_images'].min().item(),
                'generated_max': outputs['generated_images'].max().item()
            }
        }

def create_improved_hmad_model(config: Dict) -> SimplifiedHMADFramework:
    """Factory function for creating improved HMAD model"""
    
    model = SimplifiedHMADFramework(
        mindbigdata_channels=config.get('mindbigdata_channels', 14),
        crell_channels=config.get('crell_channels', 64),
        d_model=config.get('d_model', 256),
        image_size=config.get('image_size', 64)
    )
    
    return model

# Training recommendations
TRAINING_RECOMMENDATIONS = """
Key Issues and Solutions for HMAD Implementation:

1. **Complexity Overload**: 
   - Original model too complex for limited data
   - Solution: Start with simplified architecture, add complexity gradually

2. **Poor Weight Initialization**:
   - Random weights causing gradient issues
   - Solution: Proper Xavier/Kaiming initialization

3. **Loss Function Balance**:
   - Multiple loss components competing
   - Solution: Learnable loss weights with clamping

4. **Training Strategy**:
   - End-to-end training too ambitious
   - Solution: Progressive training (reconstruction → fine-tuning)

5. **Data Preprocessing**:
   - EEG signals need proper normalization
   - Solution: Z-score normalization per channel

6. **Gradient Flow**:
   - Complex architecture blocking gradients
   - Solution: Residual connections, batch norm, gradient clipping

Training Protocol:
1. Phase 1 (50 epochs): MSE loss only, focus on basic reconstruction
2. Phase 2 (50 epochs): Add perceptual loss, fine-tune details
3. Monitor: PSNR > 15dB, Cosine similarity > 0.3 as success indicators
4. Learning rate: Start 1e-4, reduce on plateau
5. Batch size: 8-16 for stable gradients
"""

if __name__ == "__main__":
    print("HMAD Improved Framework")
    print("="*50)
    print(TRAINING_RECOMMENDATIONS)
    
    # Example usage
    config = {
        'mindbigdata_channels': 14,
        'crell_channels': 64,
        'd_model': 256,
        'image_size': 64
    }
    
    model = create_improved_hmad_model(config)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with dummy data
    batch_size = 4
    mindbig_eeg = torch.randn(batch_size, 14, 256)
    target_imgs = torch.rand(batch_size, 3, 64, 64)
    
    outputs = model(mindbig_eeg, 'mindbigdata', target_imgs)
    print(f"Output shape: {outputs['generated_images'].shape}")
    print(f"Loss: {outputs['total_loss'].item():.4f}")