# HMAD Framework - Legacy Backup

## Overview
This folder contains the original implementation of the **Hierarchical Multi-Modal Attention Diffusion (HMAD) Framework** before simplifications for testing.

## Key Novelties Preserved

### 1. Advanced Signal Processing
- **Hilbert-Huang Transform (HHT)** with Empirical Mode Decomposition (EMD)
- **Graph Connectivity Analysis** using phase synchronization
- **Multi-scale temporal processing** with different resolutions

### 2. Multi-Modal Attention Mechanisms
- **Time-Frequency Multi-Head Cross-Attention (TF-MCA)**
- **Hierarchical Feature Extraction** with multiple branches:
  - Temporal Branch (multi-scale convolutions)
  - Spatial Branch (channel attention)
  - Spectral Branch (frequency band filters)
  - Connectivity Branch (graph features)

### 3. Cross-Modal Alignment
- **CLIP-aligned feature space** for EEG-to-image mapping
- **Contrastive learning** for cross-modal alignment
- **Two-stage diffusion generation**: EEG→CLIP latent→Image

### 4. Domain Adaptation
- **Multi-dataset training** (MindBigData + Crell)
- **Adversarial domain adaptation** with gradient reversal
- **Domain-specific normalization layers**

### 5. Advanced Loss Functions
- **Temporal-Spatial-Frequency (TSF) Loss**
- **Contrastive alignment loss**
- **Domain adaptation loss**
- **Learnable loss weights**

## Files Backed Up

### hmad_original.py
- Complete HMAD framework implementation
- All novel components preserved
- Advanced HHT with cubic spline interpolation
- Full graph connectivity analysis
- Complete diffusion architecture

### test_hmad_original.py
- Original test script
- Dataset loading for both MindBigData and Crell
- Forward pass testing

## Known Issues in Original
- **Spline interpolation errors** in EMD implementation
- **Complex dependencies** causing runtime issues
- **Memory intensive** operations

## Simplifications Made for Testing
The working version includes:
- Simplified EMD using filtering approach
- Reduced model complexity for testing
- Maintained core architectural principles
- Preserved multi-modal attention mechanisms

## Research Contributions
1. **Novel EEG preprocessing** with HHT and graph connectivity
2. **Multi-scale attention fusion** for EEG feature extraction
3. **Cross-modal diffusion** for EEG-to-image generation
4. **Domain adaptation** for multi-dataset learning
5. **Comprehensive loss design** for multi-objective optimization

## Target Performance
- **SSIM > 0.5** for image reconstruction
- **Real-time processing** capability
- **Cross-subject generalization**
- **Multi-dataset compatibility**

## Usage Note
Refer to this backup when implementing the full research version or when publishing the methodology. The simplified version is for testing and demonstration purposes only.
