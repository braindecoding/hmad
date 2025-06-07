#!/usr/bin/env python3
"""
Simplified test script untuk HMAD Framework
Menjalankan model dengan komponen yang disederhanakan untuk testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Optional
from test_hmad import load_mindbigdata_sample, load_crell_sample, load_stimulus_images

# Import additional components for Phase 2
class ChannelAttention(nn.Module):
    """Channel attention mechanism untuk EEG spatial features"""

    def __init__(self, num_channels: int, reduction: int = 16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(num_channels, max(1, num_channels // reduction)),
            nn.ReLU(),
            nn.Linear(max(1, num_channels // reduction), num_channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, time_points)
        """
        # Global pooling
        avg_out = self.avg_pool(x).squeeze(-1)  # (batch_size, channels)
        max_out = self.max_pool(x).squeeze(-1)  # (batch_size, channels)

        # Attention weights
        avg_weights = self.fc(avg_out)
        max_weights = self.fc(max_out)

        # Combine dan apply sigmoid
        attention_weights = self.sigmoid(avg_weights + max_weights)

        # Apply attention
        attention_weights = attention_weights.unsqueeze(-1)  # Add time dimension
        attended = x * attention_weights

        return attended

class TemporalBranch(nn.Module):
    """Temporal feature extraction dengan multiple resolutions"""

    def __init__(self, input_channels: int, d_model: int):
        super().__init__()

        # Multi-scale temporal convolutions (simplified)
        self.conv_4ms = nn.Conv1d(input_channels, d_model // 4, kernel_size=8, stride=2, padding=4)
        self.conv_8ms = nn.Conv1d(input_channels, d_model // 4, kernel_size=16, stride=4, padding=8)
        self.conv_16ms = nn.Conv1d(input_channels, d_model // 4, kernel_size=32, stride=8, padding=16)
        self.conv_32ms = nn.Conv1d(input_channels, d_model // 4, kernel_size=64, stride=16, padding=32)

        # Temporal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)  # Reduced for stability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, time_points)
        """
        # Multi-scale convolutions
        feat_4ms = F.relu(self.conv_4ms(x))
        feat_8ms = F.relu(self.conv_8ms(x))
        feat_16ms = F.relu(self.conv_16ms(x))
        feat_32ms = F.relu(self.conv_32ms(x))

        # Find minimum length untuk alignment
        min_len = min(feat_4ms.shape[2], feat_8ms.shape[2],
                     feat_16ms.shape[2], feat_32ms.shape[2])

        # Truncate to same length
        feat_4ms = feat_4ms[:, :, :min_len]
        feat_8ms = feat_8ms[:, :, :min_len]
        feat_16ms = feat_16ms[:, :, :min_len]
        feat_32ms = feat_32ms[:, :, :min_len]

        # Concatenate multi-scale features
        multi_scale = torch.cat([feat_4ms, feat_8ms, feat_16ms, feat_32ms], dim=1)

        # Transpose untuk transformer (batch, seq, features)
        multi_scale = multi_scale.transpose(1, 2)

        # Apply transformer
        temporal_features = self.transformer(multi_scale)

        return temporal_features

class SpatialBranch(nn.Module):
    """Spatial feature extraction dengan channel attention"""

    def __init__(self, input_channels: int, d_model: int):
        super().__init__()

        # Depthwise convolution untuk spatial patterns
        self.depthwise_conv = nn.Conv1d(
            input_channels, input_channels,
            kernel_size=64, groups=input_channels, padding=32
        )

        # Channel attention
        self.channel_attention = ChannelAttention(input_channels)

        # Pointwise convolution
        self.pointwise_conv = nn.Conv1d(input_channels, d_model, kernel_size=1)

        # Spatial transformer
        self.spatial_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, 8, batch_first=True),
            num_layers=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, time_points)
        """
        # Depthwise spatial convolution
        spatial_conv = F.relu(self.depthwise_conv(x))

        # Channel attention
        attended = self.channel_attention(spatial_conv)

        # Pointwise convolution
        features = F.relu(self.pointwise_conv(attended))

        # Transpose for transformer
        features = features.transpose(1, 2)

        # Spatial transformer
        spatial_features = self.spatial_transformer(features)

        return spatial_features

class SpectralBranch(nn.Module):
    """Spectral feature extraction dengan frequency band filters"""

    def __init__(self, input_channels: int, d_model: int):
        super().__init__()

        # Frequency band filters (simplified)
        self.freq_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

        # Ensure d_model is divisible by 5
        band_dim = d_model // 5
        remaining_dim = d_model - (band_dim * 4)  # Last band gets remaining dimensions

        # Learnable frequency filters
        self.band_filters = nn.ModuleDict()
        for i, band in enumerate(self.freq_bands):
            if i == len(self.freq_bands) - 1:  # Last band
                self.band_filters[band] = nn.Conv1d(input_channels, remaining_dim, kernel_size=32, padding=16)
            else:
                self.band_filters[band] = nn.Conv1d(input_channels, band_dim, kernel_size=32, padding=16)

        # Spectral attention
        self.spectral_attention = nn.MultiheadAttention(d_model, 8, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, time_points)
        """
        band_features = []

        # Extract features untuk each frequency band
        for band_name, conv_layer in self.band_filters.items():
            # Apply band filtering (simplified)
            band_feat = F.relu(conv_layer(x))
            band_features.append(band_feat)

        # Concatenate band features
        spectral_features = torch.cat(band_features, dim=1)

        # Debug: Check if we have correct dimension
        expected_channels = sum(conv.out_channels for conv in self.band_filters.values())
        print(f"    Spectral features shape before transpose: {spectral_features.shape}, expected channels: {expected_channels}")

        # Transpose for attention
        spectral_features = spectral_features.transpose(1, 2)

        # Apply spectral attention
        attended_features, _ = self.spectral_attention(
            spectral_features, spectral_features, spectral_features
        )

        return attended_features

class ConnectivityBranch(nn.Module):
    """Process graph connectivity features"""

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()

        self.projection = nn.Linear(input_dim, d_model)
        self.connectivity_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, 8, batch_first=True),
            num_layers=2
        )

    def forward(self, graph_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            graph_features: (batch_size, num_nodes, input_dim)
        """
        # Project to d_model
        projected = F.relu(self.projection(graph_features))

        # Apply transformer
        connectivity_features = self.connectivity_transformer(projected)

        return connectivity_features

class CrossModalAlignmentModule(nn.Module):
    """Advanced Cross-modal alignment dengan CLIP space - Phase 3"""

    def __init__(self, eeg_dim: int, clip_dim: int = 512):
        super().__init__()

        # Progressive alignment layers
        self.alignment_layers = nn.Sequential(
            nn.Linear(eeg_dim, eeg_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(eeg_dim // 2, clip_dim),
            nn.LayerNorm(clip_dim)
        )

        # Contrastive learning projection head
        self.projection_head = nn.Sequential(
            nn.Linear(clip_dim, clip_dim),
            nn.ReLU(),
            nn.Linear(clip_dim, clip_dim)
        )

        # Temperature parameter untuk contrastive learning
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, eeg_features: torch.Tensor) -> dict:
        """
        Args:
            eeg_features: (batch_size, eeg_dim) or (batch_size, seq_len, eeg_dim)
        Returns:
            CLIP-aligned features
        """
        # Handle both 2D and 3D inputs
        if len(eeg_features.shape) == 3:
            # Global average pooling across sequence
            pooled_features = eeg_features.mean(dim=1)  # (batch_size, eeg_dim)
        else:
            pooled_features = eeg_features

        # Alignment to CLIP space
        aligned_features = self.alignment_layers(pooled_features)

        # Projection untuk contrastive learning
        projected_features = self.projection_head(aligned_features)

        return {
            'aligned_features': aligned_features,
            'projected_features': projected_features
        }

    def contrastive_loss(self, eeg_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss untuk alignment"""
        # Normalize features
        eeg_norm = F.normalize(eeg_features, dim=-1)
        image_norm = F.normalize(image_features, dim=-1)

        # Compute similarity matrix
        similarity = torch.matmul(eeg_norm, image_norm.T) / self.temperature

        # Labels (diagonal elements are positive pairs)
        batch_size = eeg_features.shape[0]
        labels = torch.arange(batch_size, device=eeg_features.device)

        # Cross-entropy loss
        loss_eeg_to_image = F.cross_entropy(similarity, labels)
        loss_image_to_eeg = F.cross_entropy(similarity.T, labels)

        return (loss_eeg_to_image + loss_image_to_eeg) / 2

class EnhancedHMAD(nn.Module):
    """Enhanced HMAD dengan domain adaptation - Phase 5"""

    def __init__(self,
                 mindbigdata_channels: int = 14,
                 crell_channels: int = 64,
                 d_model: int = 256,
                 image_size: int = 64,
                 clip_dim: int = 512,
                 use_advanced_preprocessing: bool = True,
                 use_multi_branch: bool = True,
                 use_cross_modal_alignment: bool = True,
                 use_two_stage_diffusion: bool = True,
                 use_domain_adaptation: bool = True):
        super().__init__()

        self.use_advanced_preprocessing = use_advanced_preprocessing
        self.use_multi_branch = use_multi_branch
        self.use_cross_modal_alignment = use_cross_modal_alignment
        self.use_two_stage_diffusion = use_two_stage_diffusion
        self.use_domain_adaptation = use_domain_adaptation
        self.d_model = d_model
        self.clip_dim = clip_dim

        # Advanced preprocessing components (Phase 1)
        if use_advanced_preprocessing:
            # Import simplified versions dari hmad.py
            from hmad import HilbertHuangTransform, GraphConnectivityAnalyzer

            self.hht_transform = HilbertHuangTransform(num_imfs=4)  # Reduced for stability
            self.mindbig_graph_analyzer = GraphConnectivityAnalyzer(mindbigdata_channels, d_model // 4)
            self.crell_graph_analyzer = GraphConnectivityAnalyzer(crell_channels, d_model // 4)

        # Dataset-specific preprocessing (enhanced)
        self.mindbigdata_preprocessor = nn.Sequential(
            nn.Conv1d(mindbigdata_channels, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128)  # Fixed output length
        )

        self.crell_preprocessor = nn.Sequential(
            nn.Conv1d(crell_channels, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128)  # Fixed output length
        )

        # Multi-branch feature extraction (Phase 2)
        if use_multi_branch:
            # Branches untuk MindBigData
            self.mindbig_temporal_branch = TemporalBranch(mindbigdata_channels, d_model)
            self.mindbig_spatial_branch = SpatialBranch(mindbigdata_channels, d_model)
            self.mindbig_spectral_branch = SpectralBranch(mindbigdata_channels, d_model)
            self.mindbig_connectivity_branch = ConnectivityBranch(d_model // 4, d_model)

            # Branches untuk Crell
            self.crell_temporal_branch = TemporalBranch(crell_channels, d_model)
            self.crell_spatial_branch = SpatialBranch(crell_channels, d_model)
            self.crell_spectral_branch = SpectralBranch(crell_channels, d_model)
            self.crell_connectivity_branch = ConnectivityBranch(d_model // 4, d_model)

            # Feature fusion
            self.fusion_layer = nn.MultiheadAttention(d_model, 8, batch_first=True)
            self.output_projection = nn.Linear(d_model, d_model)
        
        # Simplified feature extractor (fallback)
        self.feature_extractor = nn.Sequential(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, 8, batch_first=True),
                num_layers=2
            )
        )

        # Cross-modal alignment (Phase 3)
        if use_cross_modal_alignment:
            self.alignment_module = CrossModalAlignmentModule(d_model, clip_dim)
        else:
            # Simplified alignment (fallback)
            self.alignment_module = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, clip_dim),
                nn.LayerNorm(clip_dim)
            )

        # Domain adaptation (Phase 5)
        if use_domain_adaptation:
            self.domain_adapter = DomainAdaptationModule(clip_dim, num_domains=2)

        # Two-stage diffusion generator (Phase 4)
        if use_two_stage_diffusion:
            self.image_generator = TwoStageDiffusionGenerator(clip_dim, clip_dim, image_size)  # Use clip_dim for both
        else:
            # Enhanced image generator (fallback)
            self.image_generator = EnhancedImageGenerator(clip_dim, image_size)

        # Loss weights for domain adaptation
        if use_domain_adaptation:
            self.domain_loss_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, eeg_data, dataset_type, target_images=None):
        """Enhanced forward pass dengan domain adaptation"""
        batch_size = eeg_data.shape[0]

        # Create domain ID
        if dataset_type == 'mindbigdata':
            domain_id = torch.zeros(batch_size, dtype=torch.long, device=eeg_data.device)
        else:  # crell
            domain_id = torch.ones(batch_size, dtype=torch.long, device=eeg_data.device)

        # Advanced preprocessing (Phase 1)
        advanced_features = {}
        graph_features = None

        if self.use_advanced_preprocessing:
            try:
                # HHT Transform
                hht_output = self.hht_transform(eeg_data)
                advanced_features['hht'] = hht_output['hht_spectrum']

                # Graph Connectivity Analysis
                if dataset_type == 'mindbigdata':
                    graph_output = self.mindbig_graph_analyzer(eeg_data)
                else:  # crell
                    graph_output = self.crell_graph_analyzer(eeg_data)

                advanced_features['graph'] = graph_output['graph_features']
                advanced_features['connectivity'] = graph_output['connectivity_matrix']
                graph_features = graph_output['graph_features']

                print(f"  Advanced preprocessing successful for {dataset_type}")

            except Exception as e:
                print(f"  Advanced preprocessing failed: {e}, falling back to basic")
                self.use_advanced_preprocessing = False

        # Multi-branch feature extraction (Phase 2)
        multi_branch_features = {}

        if self.use_multi_branch:
            try:
                if dataset_type == 'mindbigdata':
                    # Multi-branch extraction untuk MindBigData
                    print(f"    Input EEG shape: {eeg_data.shape}")
                    temporal_feat = self.mindbig_temporal_branch(eeg_data)
                    print(f"    Temporal feat shape: {temporal_feat.shape}")
                    spatial_feat = self.mindbig_spatial_branch(eeg_data)
                    print(f"    Spatial feat shape: {spatial_feat.shape}")
                    spectral_feat = self.mindbig_spectral_branch(eeg_data)
                    print(f"    Spectral feat shape: {spectral_feat.shape}")

                    if graph_features is not None:
                        print(f"    Graph features shape: {graph_features.shape}")
                        connectivity_feat = self.mindbig_connectivity_branch(graph_features)
                        print(f"    Connectivity feat shape: {connectivity_feat.shape}")
                    else:
                        # Fallback jika graph features tidak ada
                        connectivity_feat = torch.zeros_like(temporal_feat)
                        print(f"    Connectivity feat shape (fallback): {connectivity_feat.shape}")

                else:  # crell
                    # Multi-branch extraction untuk Crell
                    temporal_feat = self.crell_temporal_branch(eeg_data)
                    spatial_feat = self.crell_spatial_branch(eeg_data)
                    spectral_feat = self.crell_spectral_branch(eeg_data)

                    if graph_features is not None:
                        connectivity_feat = self.crell_connectivity_branch(graph_features)
                    else:
                        # Fallback jika graph features tidak ada
                        connectivity_feat = torch.zeros_like(temporal_feat)

                # Store branch features
                multi_branch_features = {
                    'temporal': temporal_feat,
                    'spatial': spatial_feat,
                    'spectral': spectral_feat,
                    'connectivity': connectivity_feat
                }

                # Debug shapes before fusion
                print(f"    Branch shapes - Temporal: {temporal_feat.shape}, Spatial: {spatial_feat.shape}")
                print(f"    Branch shapes - Spectral: {spectral_feat.shape}, Connectivity: {connectivity_feat.shape}")

                # Ensure all features have same sequence length
                min_seq_len = min(temporal_feat.shape[1], spatial_feat.shape[1],
                                spectral_feat.shape[1], connectivity_feat.shape[1])

                # Truncate to same sequence length
                temporal_feat = temporal_feat[:, :min_seq_len, :]
                spatial_feat = spatial_feat[:, :min_seq_len, :]
                spectral_feat = spectral_feat[:, :min_seq_len, :]
                connectivity_feat = connectivity_feat[:, :min_seq_len, :]

                # Ensure all features have same feature dimension
                if (temporal_feat.shape[2] != self.d_model or
                    spatial_feat.shape[2] != self.d_model or
                    spectral_feat.shape[2] != self.d_model or
                    connectivity_feat.shape[2] != self.d_model):

                    # Project to correct dimension if needed
                    if not hasattr(self, 'branch_projections'):
                        self.branch_projections = nn.ModuleDict({
                            'temporal': nn.Linear(temporal_feat.shape[2], self.d_model),
                            'spatial': nn.Linear(spatial_feat.shape[2], self.d_model),
                            'spectral': nn.Linear(spectral_feat.shape[2], self.d_model),
                            'connectivity': nn.Linear(connectivity_feat.shape[2], self.d_model)
                        }).to(temporal_feat.device)

                    temporal_feat = self.branch_projections['temporal'](temporal_feat)
                    spatial_feat = self.branch_projections['spatial'](spatial_feat)
                    spectral_feat = self.branch_projections['spectral'](spectral_feat)
                    connectivity_feat = self.branch_projections['connectivity'](connectivity_feat)

                print(f"    After alignment - All shapes: {temporal_feat.shape}")

                # Multi-modal fusion menggunakan attention
                # Stack all features untuk fusion
                all_features = torch.stack([
                    temporal_feat,
                    spatial_feat,
                    spectral_feat,
                    connectivity_feat
                ], dim=1)  # (batch_size, 4, seq_len, d_model)

                # Reshape untuk attention
                all_features_reshaped = all_features.reshape(batch_size, -1, self.d_model)

                # Multi-modal fusion
                fused_features, attention_weights = self.fusion_layer(
                    all_features_reshaped,
                    all_features_reshaped,
                    all_features_reshaped
                )

                # Final projection
                extracted_features = self.output_projection(fused_features)

                # Global pooling
                pooled_features = extracted_features.mean(dim=1)  # (batch, d_model)

                print(f"  Multi-branch feature extraction successful for {dataset_type}")

            except Exception as e:
                print(f"  Multi-branch extraction failed: {e}, falling back to basic")
                self.use_multi_branch = False

                # Fallback to basic preprocessing
                if dataset_type == 'mindbigdata':
                    features = self.mindbigdata_preprocessor(eeg_data)
                else:
                    features = self.crell_preprocessor(eeg_data)

                features = features.transpose(1, 2)
                extracted_features = self.feature_extractor(features)
                pooled_features = extracted_features.mean(dim=1)

        else:
            # Basic preprocessing path
            if dataset_type == 'mindbigdata':
                features = self.mindbigdata_preprocessor(eeg_data)
            else:
                features = self.crell_preprocessor(eeg_data)

            features = features.transpose(1, 2)
            extracted_features = self.feature_extractor(features)
            pooled_features = extracted_features.mean(dim=1)
        
        # Cross-modal alignment (Phase 3)
        cross_modal_features = {}

        if self.use_cross_modal_alignment:
            try:
                # Advanced cross-modal alignment
                alignment_outputs = self.alignment_module(pooled_features)
                aligned_features = alignment_outputs['aligned_features']  # (batch, clip_dim)
                projected_features = alignment_outputs['projected_features']  # For contrastive learning

                cross_modal_features = {
                    'aligned_features': aligned_features,
                    'projected_features': projected_features
                }

                print(f"  Cross-modal alignment successful for {dataset_type}")

            except Exception as e:
                print(f"  Cross-modal alignment failed: {e}, falling back to basic")
                self.use_cross_modal_alignment = False
                aligned_features = self.alignment_module(pooled_features)
        else:
            # Basic alignment
            aligned_features = self.alignment_module(pooled_features)

        # Domain adaptation (Phase 5)
        domain_features = {}

        if self.use_domain_adaptation:
            try:
                # Domain adaptation
                adaptation_outputs = self.domain_adapter(aligned_features, domain_id)
                adapted_features = adaptation_outputs['adapted_features']
                domain_predictions = adaptation_outputs['domain_predictions']

                domain_features = {
                    'adapted_features': adapted_features,
                    'domain_predictions': domain_predictions,
                    'domain_loss': F.cross_entropy(domain_predictions, domain_id)
                }

                print(f"  Domain adaptation successful for {dataset_type}")

                # Use adapted features for image generation
                final_features = adapted_features

            except Exception as e:
                print(f"  Domain adaptation failed: {e}, falling back to aligned features")
                self.use_domain_adaptation = False
                final_features = aligned_features
        else:
            # Use aligned features directly
            final_features = aligned_features

        # Image generation (Phase 4)
        diffusion_features = {}

        if self.use_two_stage_diffusion:
            try:
                # Two-stage diffusion generation (use final_features)
                diffusion_outputs = self.image_generator(final_features, target_images)
                generated_images = diffusion_outputs['generated_images']

                diffusion_features = {
                    'diffusion_clip_latent': diffusion_outputs['clip_latent'],
                    'tsf_loss': diffusion_outputs.get('tsf_loss', None)
                }

                print(f"  Two-stage diffusion generation successful for {dataset_type}")

            except Exception as e:
                print(f"  Two-stage diffusion failed: {e}, falling back to enhanced generator")
                self.use_two_stage_diffusion = False
                generated_images = self.image_generator(aligned_features)
        else:
            # Enhanced image generation (fallback)
            generated_images = self.image_generator(aligned_features)
        
        outputs = {
            'generated_images': generated_images,
            'clip_latent': aligned_features,
            'features': pooled_features,
            'advanced_features': advanced_features,  # Phase 1 results
            'multi_branch_features': multi_branch_features if self.use_multi_branch else {},  # Phase 2 results
            'cross_modal_features': cross_modal_features if self.use_cross_modal_alignment else {},  # Phase 3 results
            'diffusion_features': diffusion_features if self.use_two_stage_diffusion else {},  # Phase 4 results
            'domain_features': domain_features if self.use_domain_adaptation else {}  # Phase 5 results
        }
        
        # Enhanced loss calculation if target provided
        if target_images is not None:
            # Base reconstruction loss
            reconstruction_loss = F.mse_loss(generated_images, target_images)

            # Add TSF loss if available
            if 'tsf_loss' in diffusion_features and diffusion_features['tsf_loss'] is not None:
                reconstruction_loss = diffusion_features['tsf_loss']

            # Add domain loss if available
            total_loss = reconstruction_loss
            if 'domain_loss' in domain_features and domain_features['domain_loss'] is not None:
                domain_loss = domain_features['domain_loss']
                total_loss = reconstruction_loss + self.domain_loss_weight * domain_loss
                outputs['domain_loss'] = domain_loss

            outputs['reconstruction_loss'] = reconstruction_loss
            outputs['total_loss'] = total_loss
            
        return outputs

class EnhancedImageGenerator(nn.Module):
    """Enhanced image generator dengan better architecture - Phase 3"""

    def __init__(self, clip_dim: int, image_size: int = 64):
        super().__init__()
        self.image_size = image_size

        # Enhanced generator dengan residual connections
        self.initial_projection = nn.Sequential(
            nn.Linear(clip_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(1024, 1024)
            ) for _ in range(3)
        ])

        # Final generation layers
        self.final_layers = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 3 * image_size * image_size),
            nn.Tanh()
        )

    def forward(self, clip_features):
        """Generate images from CLIP features dengan residual connections"""
        batch_size = clip_features.shape[0]

        # Initial projection
        x = self.initial_projection(clip_features)

        # Residual blocks
        for res_block in self.res_blocks:
            residual = x
            x = res_block(x)
            x = x + residual  # Residual connection
            x = F.relu(x)

        # Final generation
        flat_images = self.final_layers(x)

        # Reshape to image format
        images = flat_images.view(batch_size, 3, self.image_size, self.image_size)

        return images

class SimpleImageGenerator(nn.Module):
    """Simple image generator dari CLIP features (fallback)"""

    def __init__(self, clip_dim: int, image_size: int = 64):
        super().__init__()
        self.image_size = image_size

        # Simple upsampling network
        self.generator = nn.Sequential(
            nn.Linear(clip_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 3 * image_size * image_size),
            nn.Tanh()
        )

    def forward(self, clip_features):
        """Generate images from CLIP features"""
        batch_size = clip_features.shape[0]

        # Generate flattened images
        flat_images = self.generator(clip_features)

        # Reshape to image format
        images = flat_images.view(batch_size, 3, self.image_size, self.image_size)

        return images

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding untuk timestep encoding"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

class ConditionalResBlock(nn.Module):
    """Residual block dengan conditional input untuk diffusion"""

    def __init__(self, in_channels: int, out_channels: int, condition_dim: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Conditional modulation
        self.condition_proj = nn.Linear(condition_dim, out_channels * 2)

        # Normalization
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)

        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        skip = self.skip_conv(x)

        # First conv
        h = self.conv1(x)
        h = self.norm1(h)

        # Conditional modulation
        condition_params = self.condition_proj(condition)
        scale, shift = condition_params.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        h = h * (1 + scale) + shift
        h = F.relu(h)

        # Second conv
        h = self.conv2(h)
        h = self.norm2(h)

        # Residual connection
        return F.relu(h + skip)

class LatentDiffusionDecoder(nn.Module):
    """Simplified Latent diffusion decoder untuk generating images from CLIP features"""

    def __init__(self, clip_dim: int, image_size: int = 64):
        super().__init__()
        self.image_size = image_size

        # Conditional embedding
        self.condition_embedding = nn.Sequential(
            nn.Linear(clip_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )

        # Simplified U-Net architecture
        self.time_embedding = SinusoidalPositionEmbedding(512)  # Match condition dimension

        # Encoder blocks (simplified)
        self.encoder_blocks = nn.ModuleList([
            ConditionalResBlock(3, 32, 512),
            ConditionalResBlock(32, 64, 512),
            ConditionalResBlock(64, 128, 512)
        ])

        # Bottleneck
        self.bottleneck = ConditionalResBlock(128, 128, 512)

        # Decoder blocks dengan skip connections
        self.decoder_blocks = nn.ModuleList([
            ConditionalResBlock(256, 64, 512),   # 128 + 128 (skip)
            ConditionalResBlock(128, 32, 512),   # 64 + 64 (skip)
            ConditionalResBlock(64, 16, 512)     # 32 + 32 (skip)
        ])

        # Final output layer
        self.final_conv = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def forward(self, clip_features: torch.Tensor, timestep: float = 0.0) -> torch.Tensor:
        batch_size = clip_features.shape[0]

        # Condition embedding
        condition = self.condition_embedding(clip_features)

        # Time embedding
        t_emb = self.time_embedding(torch.tensor([timestep] * batch_size,
                                                device=clip_features.device))

        # Combine condition dan time
        combined_condition = condition + t_emb

        # Start dengan noise atau learned initialization
        x = torch.randn(batch_size, 3, self.image_size, self.image_size,
                       device=clip_features.device)

        # Encoder dengan skip connections
        skip_connections = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, combined_condition)
            skip_connections.append(x)
            x = F.max_pool2d(x, 2)

        # Bottleneck
        x = self.bottleneck(x, combined_condition)

        # Decoder dengan skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

            # Skip connection
            if i < len(skip_connections):
                skip = skip_connections[-(i+1)]
                # Resize skip jika perlu
                if x.shape[2:] != skip.shape[2:]:
                    skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)

            x = decoder_block(x, combined_condition)

        # Final output
        x = self.final_conv(x)
        x = torch.tanh(x)  # Output range [-1, 1]

        return x

class TwoStageDiffusionGenerator(nn.Module):
    """Two-stage diffusion: EEG->CLIP latent, CLIP latent->Image - Phase 4"""

    def __init__(self, eeg_dim: int, clip_dim: int = 512, image_size: int = 64):
        super().__init__()

        # Stage 1: EEG to CLIP latent mapping (enhanced)
        self.eeg_to_clip = nn.Sequential(
            nn.Linear(eeg_dim, eeg_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(eeg_dim // 2, clip_dim),
            nn.LayerNorm(clip_dim),
            nn.Tanh()  # Bounded output
        )

        # Stage 2: CLIP latent to image diffusion
        self.latent_to_image = LatentDiffusionDecoder(clip_dim, image_size)

        # Temporal-Spatial-Frequency loss components
        self.temporal_loss_weight = nn.Parameter(torch.tensor(1.0))
        self.spatial_loss_weight = nn.Parameter(torch.tensor(1.0))
        self.frequency_loss_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, eeg_features: torch.Tensor,
                target_images: Optional[torch.Tensor] = None,
                noise_level: float = 0.0) -> dict:
        # Stage 1: EEG to CLIP latent
        clip_latent = self.eeg_to_clip(eeg_features)

        # Stage 2: CLIP latent to image
        generated_images = self.latent_to_image(clip_latent, noise_level)

        outputs = {
            'clip_latent': clip_latent,
            'generated_images': generated_images
        }

        # Compute TSF loss if target provided
        if target_images is not None:
            tsf_loss = self.compute_tsf_loss(generated_images, target_images)
            outputs['tsf_loss'] = tsf_loss
            outputs['total_loss'] = tsf_loss

        return outputs

    def compute_tsf_loss(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Temporal-Spatial-Frequency loss function"""

        # Temporal loss (MSE in time domain)
        temporal_loss = F.mse_loss(generated, target)

        # Spatial loss (MSE in spatial domain)
        spatial_loss = F.mse_loss(
            generated.view(generated.shape[0], -1),
            target.view(target.shape[0], -1)
        )

        # Frequency loss (MSE in frequency domain using FFT)
        generated_fft = torch.fft.fft2(generated)
        target_fft = torch.fft.fft2(target)

        frequency_loss = F.mse_loss(
            torch.abs(generated_fft),
            torch.abs(target_fft)
        )

        # Weighted combination
        total_loss = (self.temporal_loss_weight * temporal_loss +
                     self.spatial_loss_weight * spatial_loss +
                     self.frequency_loss_weight * frequency_loss)

        return total_loss

def visualize_reconstruction_results(target_images, reconstructed_images, dataset_name, save_path=None):
    """
    Visualize comparison between target and reconstructed images

    Args:
        target_images: (batch_size, 3, H, W) - Target images
        reconstructed_images: (batch_size, 3, H, W) - Reconstructed images
        dataset_name: str - Name of dataset
        save_path: str - Path to save visualization
    """
    batch_size = target_images.shape[0]

    # Convert tensors to numpy and denormalize
    target_np = target_images.detach().cpu().numpy()
    recon_np = reconstructed_images.detach().cpu().numpy()

    # Denormalize from [-1, 1] to [0, 1]
    target_np = (target_np + 1) / 2
    recon_np = (recon_np + 1) / 2

    # Clip values to [0, 1]
    target_np = np.clip(target_np, 0, 1)
    recon_np = np.clip(recon_np, 0, 1)

    # Create figure
    fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 3, 6))
    if batch_size == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(f'HMAD Reconstruction Results - {dataset_name}', fontsize=16, fontweight='bold')

    for i in range(batch_size):
        # Target images (top row)
        target_img = np.transpose(target_np[i], (1, 2, 0))
        axes[0, i].imshow(target_img)
        axes[0, i].set_title(f'Target {i+1}', fontweight='bold')
        axes[0, i].axis('off')

        # Reconstructed images (bottom row)
        recon_img = np.transpose(recon_np[i], (1, 2, 0))
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title(f'Reconstructed {i+1}', fontweight='bold')
        axes[1, i].axis('off')

    # Add row labels
    axes[0, 0].text(-0.1, 0.5, 'Target\nImages', transform=axes[0, 0].transAxes,
                    fontsize=14, fontweight='bold', ha='center', va='center', rotation=90)
    axes[1, 0].text(-0.1, 0.5, 'HMAD\nReconstruction', transform=axes[1, 0].transAxes,
                    fontsize=14, fontweight='bold', ha='center', va='center', rotation=90)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Visualization saved to: {save_path}")

    plt.show()

    # Calculate and display metrics
    mse = F.mse_loss(reconstructed_images, target_images).item()
    mae = F.l1_loss(reconstructed_images, target_images).item()

    print(f"  Reconstruction Metrics for {dataset_name}:")
    print(f"    MSE: {mse:.4f}")
    print(f"    MAE: {mae:.4f}")

    return mse, mae

def create_synthetic_target_images(batch_size, image_size=64, dataset_type='mindbigdata'):
    """
    Create synthetic target images for visualization

    Args:
        batch_size: Number of images to create
        image_size: Size of images
        dataset_type: Type of dataset ('mindbigdata' or 'crell')
    """
    if dataset_type == 'mindbigdata':
        # Create digit-like patterns for MindBigData
        targets = []
        for i in range(batch_size):
            # Create a simple digit pattern
            img = torch.zeros(3, image_size, image_size)

            # Create different patterns for each sample
            if i % 3 == 0:  # Digit "0" pattern
                center_x, center_y = image_size // 2, image_size // 2
                radius = image_size // 4
                y, x = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
                circle_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) < radius ** 2
                ring_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) > (radius - 5) ** 2
                final_mask = circle_mask & ring_mask
                img[:, final_mask] = 0.8

            elif i % 3 == 1:  # Digit "1" pattern
                center_x = image_size // 2
                img[:, :, center_x-2:center_x+3] = 0.8

            else:  # Digit "8" pattern
                # Top circle
                center_x, center_y = image_size // 2, image_size // 3
                radius = image_size // 6
                y, x = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
                circle_mask1 = ((x - center_x) ** 2 + (y - center_y) ** 2) < radius ** 2
                ring_mask1 = ((x - center_x) ** 2 + (y - center_y) ** 2) > (radius - 3) ** 2
                img[:, circle_mask1 & ring_mask1] = 0.8

                # Bottom circle
                center_y = 2 * image_size // 3
                circle_mask2 = ((x - center_x) ** 2 + (y - center_y) ** 2) < radius ** 2
                ring_mask2 = ((x - center_x) ** 2 + (y - center_y) ** 2) > (radius - 3) ** 2
                img[:, circle_mask2 & ring_mask2] = 0.8

            targets.append(img)

    else:  # crell dataset
        # Create letter-like patterns for Crell
        targets = []
        for i in range(batch_size):
            img = torch.zeros(3, image_size, image_size)

            if i % 3 == 0:  # Letter "A" pattern
                # Vertical lines
                img[:, 10:50, 15:20] = 0.8
                img[:, 10:50, 45:50] = 0.8
                # Horizontal line
                img[:, 25:30, 15:50] = 0.8
                # Top connection
                img[:, 10:15, 20:45] = 0.8

            elif i % 3 == 1:  # Letter "B" pattern
                # Vertical line
                img[:, 10:50, 15:20] = 0.8
                # Top horizontal
                img[:, 10:15, 15:40] = 0.8
                # Middle horizontal
                img[:, 27:32, 15:35] = 0.8
                # Bottom horizontal
                img[:, 45:50, 15:40] = 0.8
                # Right curves (simplified)
                img[:, 15:27, 35:40] = 0.8
                img[:, 32:45, 35:40] = 0.8

            else:  # Letter "C" pattern
                # Left vertical
                img[:, 15:45, 15:20] = 0.8
                # Top horizontal
                img[:, 15:20, 15:45] = 0.8
                # Bottom horizontal
                img[:, 40:45, 15:45] = 0.8

            targets.append(img)

    # Stack and normalize to [-1, 1] range
    target_tensor = torch.stack(targets)
    target_tensor = target_tensor * 2 - 1  # Convert [0, 1] to [-1, 1]

    return target_tensor

class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer untuk adversarial training"""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output

class DomainAdaptationModule(nn.Module):
    """Domain adaptation untuk multi-dataset training - Phase 5"""

    def __init__(self, feature_dim: int, num_domains: int = 2):
        super().__init__()
        self.num_domains = num_domains

        # Domain classifier untuk adversarial training
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, num_domains)
        )

        # Domain-specific normalization layers
        self.domain_norms = nn.ModuleList([
            nn.LayerNorm(feature_dim) for _ in range(num_domains)
        ])

        # Gradient reversal layer
        self.gradient_reversal = GradientReversalLayer()

    def forward(self, features: torch.Tensor, domain_id: torch.Tensor) -> dict:
        """
        Args:
            features: (batch_size, feature_dim)
            domain_id: (batch_size,) - 0 for MindBigData, 1 for Crell
        """
        batch_size = features.shape[0]

        # Domain-specific normalization
        normalized_features = torch.zeros_like(features)
        for i in range(self.num_domains):
            mask = (domain_id == i)
            if mask.any():
                normalized_features[mask] = self.domain_norms[i](features[mask])

        # Domain classification untuk adversarial loss
        reversed_features = self.gradient_reversal.apply(normalized_features)
        domain_pred = self.domain_classifier(reversed_features)

        return {
            'adapted_features': normalized_features,
            'domain_predictions': domain_pred
        }

def visualize_reconstruction_results(reconstructed_images, dataset_name, eeg_labels=None, save_path=None):
    """
    Visualize HMAD reconstruction results from real EEG data

    Args:
        reconstructed_images: (batch_size, 3, H, W) - Images reconstructed from real EEG
        dataset_name: str - Name of dataset
        eeg_labels: Labels corresponding to EEG data (if available)
        save_path: str - Path to save visualization
    """
    batch_size = reconstructed_images.shape[0]

    # Convert tensors to numpy and denormalize
    recon_np = reconstructed_images.detach().cpu().numpy()

    # Denormalize from [-1, 1] to [0, 1]
    recon_np = (recon_np + 1) / 2

    # Clip values to [0, 1]
    recon_np = np.clip(recon_np, 0, 1)

    # Create figure
    fig, axes = plt.subplots(1, batch_size, figsize=(batch_size * 3, 3))
    if batch_size == 1:
        axes = [axes]

    fig.suptitle(f'HMAD EEG-to-Image Reconstruction - {dataset_name}', fontsize=16, fontweight='bold')

    for i in range(batch_size):
        # Reconstructed images
        recon_img = np.transpose(recon_np[i], (1, 2, 0))
        axes[i].imshow(recon_img)

        # Add label if available
        if eeg_labels is not None:
            if dataset_name.lower() == 'mindbigdata':
                title = f'Digit: {eeg_labels[i]}'
            else:  # crell
                title = f'Letter: {chr(65 + eeg_labels[i])}'  # Convert to A-Z
        else:
            title = f'Sample {i+1}'

        axes[i].set_title(title, fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Visualization saved to: {save_path}")

    plt.show()

    # Display reconstruction info
    print(f"  Reconstruction Info for {dataset_name}:")
    print(f"    Generated {batch_size} images from real EEG signals")
    print(f"    Image shape: {reconstructed_images.shape}")
    print(f"    Value range: [{recon_np.min():.3f}, {recon_np.max():.3f}]")

def test_enhanced_hmad():
    """Test enhanced HMAD framework dengan domain adaptation"""
    print("="*60)
    print("TESTING ENHANCED HMAD FRAMEWORK - PHASE 5")
    print("="*60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Configuration
    config = {
        'mindbigdata_channels': 14,
        'crell_channels': 64,
        'd_model': 256,
        'image_size': 64
    }
    
    # Create enhanced model
    print("\nCreating enhanced HMAD model...")
    model = EnhancedHMAD(**config)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load datasets
    print("\nLoading datasets...")
    
    # MindBigData
    mindbig_eeg, mindbig_labels = load_mindbigdata_sample("datasets/EP1.01.txt", max_samples=3)
    
    # Crell
    crell_eeg, crell_labels = load_crell_sample("datasets/S01.mat", max_samples=3)
    
    # Test forward pass
    print("\nTesting forward pass...")
    
    model.eval()
    with torch.no_grad():
        # Test MindBigData
        if mindbig_eeg is not None:
            print(f"\nTesting MindBigData (shape: {mindbig_eeg.shape})...")
            mindbig_eeg = mindbig_eeg.to(device)
            
            # Create dummy target images
            batch_size = mindbig_eeg.shape[0]
            target_images = torch.randn(batch_size, 3, config['image_size'], config['image_size']).to(device)
            
            try:
                outputs = model(mindbig_eeg, 'mindbigdata', target_images)
                print(f"✓ MindBigData forward pass successful!")
                print(f"  Generated images shape: {outputs['generated_images'].shape}")
                print(f"  CLIP latent shape: {outputs['clip_latent'].shape}")
                print(f"  Total loss: {outputs['total_loss'].item():.4f}")

                # Report advanced features (Phase 1)
                if 'advanced_features' in outputs and outputs['advanced_features']:
                    adv_feat = outputs['advanced_features']
                    print(f"  Advanced features (Phase 1):")
                    if 'hht' in adv_feat:
                        print(f"    HHT spectrum shape: {adv_feat['hht'].shape}")
                    if 'graph' in adv_feat:
                        print(f"    Graph features shape: {adv_feat['graph'].shape}")
                    if 'connectivity' in adv_feat:
                        print(f"    Connectivity matrix shape: {adv_feat['connectivity'].shape}")

                # Report multi-branch features (Phase 2)
                if 'multi_branch_features' in outputs and outputs['multi_branch_features']:
                    mb_feat = outputs['multi_branch_features']
                    print(f"  Multi-branch features (Phase 2):")
                    if 'temporal' in mb_feat:
                        print(f"    Temporal branch shape: {mb_feat['temporal'].shape}")
                    if 'spatial' in mb_feat:
                        print(f"    Spatial branch shape: {mb_feat['spatial'].shape}")
                    if 'spectral' in mb_feat:
                        print(f"    Spectral branch shape: {mb_feat['spectral'].shape}")
                    if 'connectivity' in mb_feat:
                        print(f"    Connectivity branch shape: {mb_feat['connectivity'].shape}")

                # Report cross-modal features (Phase 3)
                if 'cross_modal_features' in outputs and outputs['cross_modal_features']:
                    cm_feat = outputs['cross_modal_features']
                    print(f"  Cross-modal features (Phase 3):")
                    if 'aligned_features' in cm_feat:
                        print(f"    CLIP-aligned features shape: {cm_feat['aligned_features'].shape}")
                    if 'projected_features' in cm_feat:
                        print(f"    Contrastive projection shape: {cm_feat['projected_features'].shape}")

                # Report diffusion features (Phase 4)
                if 'diffusion_features' in outputs and outputs['diffusion_features']:
                    diff_feat = outputs['diffusion_features']
                    print(f"  Diffusion features (Phase 4):")
                    if 'diffusion_clip_latent' in diff_feat:
                        print(f"    Diffusion CLIP latent shape: {diff_feat['diffusion_clip_latent'].shape}")
                    if 'tsf_loss' in diff_feat and diff_feat['tsf_loss'] is not None:
                        print(f"    TSF loss: {diff_feat['tsf_loss'].item():.4f}")

                # Report domain features (Phase 5)
                if 'domain_features' in outputs and outputs['domain_features']:
                    dom_feat = outputs['domain_features']
                    print(f"  Domain features (Phase 5):")
                    if 'adapted_features' in dom_feat:
                        print(f"    Domain-adapted features shape: {dom_feat['adapted_features'].shape}")
                    if 'domain_predictions' in dom_feat:
                        print(f"    Domain predictions shape: {dom_feat['domain_predictions'].shape}")
                    if 'domain_loss' in dom_feat and dom_feat['domain_loss'] is not None:
                        print(f"    Domain loss: {dom_feat['domain_loss'].item():.4f}")
                    
            except Exception as e:
                print(f"✗ MindBigData forward pass failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Test Crell
        if crell_eeg is not None:
            print(f"\nTesting Crell (shape: {crell_eeg.shape})...")
            crell_eeg = crell_eeg.to(device)
            
            # Create dummy target images
            batch_size = crell_eeg.shape[0]
            target_images = torch.randn(batch_size, 3, config['image_size'], config['image_size']).to(device)
            
            try:
                outputs = model(crell_eeg, 'crell', target_images)
                print(f"✓ Crell forward pass successful!")
                print(f"  Generated images shape: {outputs['generated_images'].shape}")
                print(f"  CLIP latent shape: {outputs['clip_latent'].shape}")
                print(f"  Total loss: {outputs['total_loss'].item():.4f}")

                # Report advanced features (Phase 1)
                if 'advanced_features' in outputs and outputs['advanced_features']:
                    adv_feat = outputs['advanced_features']
                    print(f"  Advanced features (Phase 1):")
                    if 'hht' in adv_feat:
                        print(f"    HHT spectrum shape: {adv_feat['hht'].shape}")
                    if 'graph' in adv_feat:
                        print(f"    Graph features shape: {adv_feat['graph'].shape}")
                    if 'connectivity' in adv_feat:
                        print(f"    Connectivity matrix shape: {adv_feat['connectivity'].shape}")

                # Report multi-branch features (Phase 2)
                if 'multi_branch_features' in outputs and outputs['multi_branch_features']:
                    mb_feat = outputs['multi_branch_features']
                    print(f"  Multi-branch features (Phase 2):")
                    if 'temporal' in mb_feat:
                        print(f"    Temporal branch shape: {mb_feat['temporal'].shape}")
                    if 'spatial' in mb_feat:
                        print(f"    Spatial branch shape: {mb_feat['spatial'].shape}")
                    if 'spectral' in mb_feat:
                        print(f"    Spectral branch shape: {mb_feat['spectral'].shape}")
                    if 'connectivity' in mb_feat:
                        print(f"    Connectivity branch shape: {mb_feat['connectivity'].shape}")

                # Report cross-modal features (Phase 3)
                if 'cross_modal_features' in outputs and outputs['cross_modal_features']:
                    cm_feat = outputs['cross_modal_features']
                    print(f"  Cross-modal features (Phase 3):")
                    if 'aligned_features' in cm_feat:
                        print(f"    CLIP-aligned features shape: {cm_feat['aligned_features'].shape}")
                    if 'projected_features' in cm_feat:
                        print(f"    Contrastive projection shape: {cm_feat['projected_features'].shape}")

                # Report diffusion features (Phase 4)
                if 'diffusion_features' in outputs and outputs['diffusion_features']:
                    diff_feat = outputs['diffusion_features']
                    print(f"  Diffusion features (Phase 4):")
                    if 'diffusion_clip_latent' in diff_feat:
                        print(f"    Diffusion CLIP latent shape: {diff_feat['diffusion_clip_latent'].shape}")
                    if 'tsf_loss' in diff_feat and diff_feat['tsf_loss'] is not None:
                        print(f"    TSF loss: {diff_feat['tsf_loss'].item():.4f}")

                # Report domain features (Phase 5)
                if 'domain_features' in outputs and outputs['domain_features']:
                    dom_feat = outputs['domain_features']
                    print(f"  Domain features (Phase 5):")
                    if 'adapted_features' in dom_feat:
                        print(f"    Domain-adapted features shape: {dom_feat['adapted_features'].shape}")
                    if 'domain_predictions' in dom_feat:
                        print(f"    Domain predictions shape: {dom_feat['domain_predictions'].shape}")
                    if 'domain_loss' in dom_feat and dom_feat['domain_loss'] is not None:
                        print(f"    Domain loss: {dom_feat['domain_loss'].item():.4f}")

            except Exception as e:
                print(f"✗ Crell forward pass failed: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "="*60)
    print("ENHANCED HMAD FRAMEWORK PHASE 5 TEST COMPLETED")
    print("="*60)

    # Visualize reconstruction results
    print("\nGenerating visualizations...")

    # Test with MindBigData
    print("\nVisualizing MindBigData reconstructions...")
    with torch.no_grad():
        mindbig_outputs = model(mindbig_eeg, 'mindbigdata', target_images)
        visualize_reconstruction_results(
            mindbig_outputs['generated_images'],
            'MindBigData',
            mindbig_labels,
            save_path='hmad_mindbigdata_reconstruction.png'
        )

    # Test with Crell
    print("\nVisualizing Crell reconstructions...")
    with torch.no_grad():
        crell_outputs = model(crell_eeg, 'crell', target_images)
        visualize_reconstruction_results(
            crell_outputs['generated_images'],
            'Crell',
            crell_labels,
            save_path='hmad_crell_reconstruction.png'
        )
    
    # Test training step
    print("\nTesting training step...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    if mindbig_eeg is not None:
        try:
            mindbig_eeg = mindbig_eeg.to(device)
            target_images = torch.randn(mindbig_eeg.shape[0], 3, config['image_size'], config['image_size']).to(device)
            
            optimizer.zero_grad()
            outputs = model(mindbig_eeg, 'mindbigdata', target_images)
            loss = outputs['total_loss']
            loss.backward()
            optimizer.step()
            
            print(f"✓ Training step successful! Loss: {loss.item():.4f}")
            
        except Exception as e:
            print(f"✗ Training step failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_hmad()
