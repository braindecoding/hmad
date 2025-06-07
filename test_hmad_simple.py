#!/usr/bin/env python3
"""
Simplified test script untuk HMAD Framework
Menjalankan model dengan komponen yang disederhanakan untuk testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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

        # Learnable frequency filters
        self.band_filters = nn.ModuleDict({
            band: nn.Conv1d(input_channels, d_model // 5, kernel_size=32, padding=16)
            for band in self.freq_bands
        })

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

class EnhancedHMAD(nn.Module):
    """Enhanced HMAD dengan multi-branch feature extraction - Phase 2"""

    def __init__(self,
                 mindbigdata_channels: int = 14,
                 crell_channels: int = 64,
                 d_model: int = 256,
                 image_size: int = 64,
                 use_advanced_preprocessing: bool = True,
                 use_multi_branch: bool = True):
        super().__init__()

        self.use_advanced_preprocessing = use_advanced_preprocessing
        self.use_multi_branch = use_multi_branch
        self.d_model = d_model

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
        
        # Simplified feature extractor
        self.feature_extractor = nn.Sequential(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, 8, batch_first=True),
                num_layers=2
            )
        )
        
        # Cross-modal alignment (simplified)
        self.alignment_module = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 512),  # CLIP dimension
            nn.LayerNorm(512)
        )
        
        # Simple image generator
        self.image_generator = SimpleImageGenerator(512, image_size)
        
    def forward(self, eeg_data, dataset_type, target_images=None):
        """Enhanced forward pass dengan multi-branch feature extraction"""
        batch_size = eeg_data.shape[0]

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
                    temporal_feat = self.mindbig_temporal_branch(eeg_data)
                    spatial_feat = self.mindbig_spatial_branch(eeg_data)
                    spectral_feat = self.mindbig_spectral_branch(eeg_data)

                    if graph_features is not None:
                        connectivity_feat = self.mindbig_connectivity_branch(graph_features)
                    else:
                        # Fallback jika graph features tidak ada
                        connectivity_feat = torch.zeros_like(temporal_feat)

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
        
        # Cross-modal alignment
        aligned_features = self.alignment_module(pooled_features)  # (batch, 512)
        
        # Image generation
        generated_images = self.image_generator(aligned_features)  # (batch, 3, 64, 64)
        
        outputs = {
            'generated_images': generated_images,
            'clip_latent': aligned_features,
            'features': pooled_features,
            'advanced_features': advanced_features,  # Phase 1 results
            'multi_branch_features': multi_branch_features if self.use_multi_branch else {}  # Phase 2 results
        }
        
        # Simple loss if target provided
        if target_images is not None:
            loss = F.mse_loss(generated_images, target_images)
            outputs['total_loss'] = loss
            
        return outputs

class SimpleImageGenerator(nn.Module):
    """Simple image generator dari CLIP features"""
    
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

def test_enhanced_hmad():
    """Test enhanced HMAD framework dengan advanced preprocessing"""
    print("="*60)
    print("TESTING ENHANCED HMAD FRAMEWORK - PHASE 1")
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
                    
            except Exception as e:
                print(f"✗ Crell forward pass failed: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "="*60)
    print("ENHANCED HMAD FRAMEWORK PHASE 2 TEST COMPLETED")
    print("="*60)
    
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
