#!/usr/bin/env python3
"""
Analyze Best Reconstruction - Real Digit 6
==========================================

Menganalisis mengapa rekonstruksi Real Digit 6 (paling kiri) paling bagus
dan bagaimana mengoptimasi rekonstruksi lainnya untuk mencapai kualitas serupa.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from hmadv2 import create_improved_hmad_model
from test_hmad import load_mindbigdata_sample, load_stimulus_images

def create_target_images_from_labels(labels, dataset_type, stimulus_images, image_size=64, device='cpu'):
    """Create target images berdasarkan labels menggunakan stimulus ASLI dari dataset"""
    batch_size = len(labels)
    target_images = torch.zeros(batch_size, 3, image_size, image_size, device=device)
    
    for i, label in enumerate(labels):
        try:
            if dataset_type == 'mindbigdata':
                digit = label.item()
                stimulus_key = f'digit_{digit}'
                if stimulus_key in stimulus_images:
                    target_images[i] = stimulus_images[stimulus_key].to(device)
                    print(f"✓ Using REAL digit stimulus for label {digit}")
        except Exception as e:
            print(f"Error loading stimulus for label {label.item()}: {e}")
    
    return target_images

def analyze_best_reconstruction():
    """Analyze mengapa rekonstruksi digit 6 paling bagus"""
    
    print("="*70)
    print("ANALYZING BEST RECONSTRUCTION - REAL DIGIT 6")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration
    config = {
        'mindbigdata_channels': 14,
        'crell_channels': 64,
        'd_model': 256,
        'image_size': 64
    }
    
    # Create model
    print("\n1. CREATING HMADV2 MODEL...")
    model = create_improved_hmad_model(config)
    model = model.to(device)
    
    # Load MindBigData samples
    print("\n2. LOADING MINDBIGDATA SAMPLES...")
    mindbig_eeg, mindbig_labels = load_mindbigdata_sample("datasets/EP1.01.txt", max_samples=8)
    
    if mindbig_eeg is None:
        print("❌ Failed to load MindBigData")
        return
    
    print(f"✓ Loaded EEG: {mindbig_eeg.shape}")
    print(f"✓ Labels: {mindbig_labels.tolist()}")
    
    # Load stimulus images
    stimulus_images = load_stimulus_images("datasets", image_size=64)
    print(f"✓ Loaded {len(stimulus_images)} stimulus images")
    
    # Move to device
    mindbig_eeg = mindbig_eeg.to(device)
    mindbig_labels = mindbig_labels.to(device)
    
    # Create target images
    target_images = create_target_images_from_labels(
        mindbig_labels, 'mindbigdata', stimulus_images, 64, device
    )
    
    print("\n3. ANALYZING INDIVIDUAL RECONSTRUCTIONS...")
    
    model.eval()
    with torch.no_grad():
        # Forward pass
        outputs = model(mindbig_eeg, 'mindbigdata', target_images)
        generated_images = outputs['generated_images']
        
        # Analyze each sample individually
        sample_metrics = []
        
        for i in range(len(mindbig_labels)):
            digit = mindbig_labels[i].item()
            
            # Individual metrics
            target_single = target_images[i:i+1]
            generated_single = generated_images[i:i+1]
            
            # MSE Loss
            mse = torch.nn.functional.mse_loss(generated_single, target_single)
            
            # PSNR
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            
            # Cosine Similarity
            gen_flat = generated_single.view(1, -1)
            target_flat = target_single.view(1, -1)
            cosine_sim = torch.nn.functional.cosine_similarity(gen_flat, target_flat, dim=1)
            
            # Structural Similarity (simplified)
            gen_mean = generated_single.mean()
            target_mean = target_single.mean()
            gen_std = generated_single.std()
            target_std = target_single.std()
            
            # Feature correlation
            correlation = torch.corrcoef(torch.stack([gen_flat.squeeze(), target_flat.squeeze()]))[0, 1]
            
            metrics = {
                'sample_idx': i,
                'digit': digit,
                'mse': mse.item(),
                'psnr': psnr.item(),
                'cosine_similarity': cosine_sim.item(),
                'gen_mean': gen_mean.item(),
                'target_mean': target_mean.item(),
                'gen_std': gen_std.item(),
                'target_std': target_std.item(),
                'correlation': correlation.item() if not torch.isnan(correlation) else 0.0
            }
            
            sample_metrics.append(metrics)
            
            print(f"Sample {i} (Digit {digit}):")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  PSNR: {metrics['psnr']:.2f} dB")
            print(f"  Cosine Sim: {metrics['cosine_similarity']:.4f}")
            print(f"  Correlation: {metrics['correlation']:.4f}")
            print(f"  Mean diff: {abs(metrics['gen_mean'] - metrics['target_mean']):.4f}")
            print(f"  Std diff: {abs(metrics['gen_std'] - metrics['target_std']):.4f}")
    
    print("\n4. IDENTIFYING BEST RECONSTRUCTION...")
    
    # Find best reconstruction (highest PSNR and cosine similarity)
    best_psnr_idx = max(range(len(sample_metrics)), key=lambda i: sample_metrics[i]['psnr'])
    best_cosine_idx = max(range(len(sample_metrics)), key=lambda i: sample_metrics[i]['cosine_similarity'])
    best_correlation_idx = max(range(len(sample_metrics)), key=lambda i: sample_metrics[i]['correlation'])
    
    print(f"Best PSNR: Sample {best_psnr_idx} (Digit {sample_metrics[best_psnr_idx]['digit']}) - {sample_metrics[best_psnr_idx]['psnr']:.2f} dB")
    print(f"Best Cosine: Sample {best_cosine_idx} (Digit {sample_metrics[best_cosine_idx]['digit']}) - {sample_metrics[best_cosine_idx]['cosine_similarity']:.4f}")
    print(f"Best Correlation: Sample {best_correlation_idx} (Digit {sample_metrics[best_correlation_idx]['digit']}) - {sample_metrics[best_correlation_idx]['correlation']:.4f}")
    
    # Analyze EEG characteristics of best sample
    print(f"\n5. ANALYZING EEG CHARACTERISTICS OF BEST SAMPLE...")
    
    best_idx = best_psnr_idx  # Use best PSNR as reference
    best_eeg = mindbig_eeg[best_idx:best_idx+1]
    
    # EEG signal analysis
    eeg_stats = analyze_eeg_characteristics(best_eeg, mindbig_eeg, best_idx)
    
    print(f"Best sample (Index {best_idx}, Digit {sample_metrics[best_idx]['digit']}):")
    print(f"  EEG Mean: {eeg_stats['best_mean']:.4f}")
    print(f"  EEG Std: {eeg_stats['best_std']:.4f}")
    print(f"  EEG Energy: {eeg_stats['best_energy']:.4f}")
    print(f"  Signal-to-Noise: {eeg_stats['best_snr']:.4f}")
    
    print(f"\nComparison with other samples:")
    print(f"  Average EEG Mean: {eeg_stats['avg_mean']:.4f}")
    print(f"  Average EEG Std: {eeg_stats['avg_std']:.4f}")
    print(f"  Average EEG Energy: {eeg_stats['avg_energy']:.4f}")
    print(f"  Average Signal-to-Noise: {eeg_stats['avg_snr']:.4f}")
    
    print("\n6. CREATING DETAILED ANALYSIS VISUALIZATION...")
    create_detailed_analysis_visualization(
        mindbig_eeg, target_images, generated_images, mindbig_labels, 
        sample_metrics, best_idx, device
    )
    
    print("\n7. OPTIMIZATION RECOMMENDATIONS...")
    provide_optimization_recommendations(sample_metrics, eeg_stats, best_idx)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETED!")
    print(f"{'='*70}")

def analyze_eeg_characteristics(best_eeg, all_eeg, best_idx):
    """Analyze EEG characteristics of best sample vs others"""
    
    # Best sample stats
    best_mean = best_eeg.mean().item()
    best_std = best_eeg.std().item()
    best_energy = (best_eeg ** 2).mean().item()
    best_snr = best_mean / (best_std + 1e-8)
    
    # All samples stats
    all_means = []
    all_stds = []
    all_energies = []
    all_snrs = []
    
    for i in range(all_eeg.shape[0]):
        if i != best_idx:  # Exclude best sample from average
            sample_eeg = all_eeg[i:i+1]
            sample_mean = sample_eeg.mean().item()
            sample_std = sample_eeg.std().item()
            sample_energy = (sample_eeg ** 2).mean().item()
            sample_snr = sample_mean / (sample_std + 1e-8)
            
            all_means.append(sample_mean)
            all_stds.append(sample_std)
            all_energies.append(sample_energy)
            all_snrs.append(sample_snr)
    
    return {
        'best_mean': best_mean,
        'best_std': best_std,
        'best_energy': best_energy,
        'best_snr': best_snr,
        'avg_mean': np.mean(all_means),
        'avg_std': np.mean(all_stds),
        'avg_energy': np.mean(all_energies),
        'avg_snr': np.mean(all_snrs)
    }

def create_detailed_analysis_visualization(eeg_data, target_images, generated_images, 
                                         labels, sample_metrics, best_idx, device):
    """Create detailed visualization of analysis results"""
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Detailed Analysis: Why Digit 6 Reconstruction is Best', fontsize=16, fontweight='bold')
    
    # Row 1: Real stimuli
    for i in range(min(4, len(labels))):
        digit = labels[i].item()
        real_img = target_images[i].cpu().numpy().transpose(1, 2, 0)
        
        axes[0, i].imshow(real_img)
        title_color = 'green' if i == best_idx else 'black'
        axes[0, i].set_title(f'Real Digit {digit}', color=title_color, fontweight='bold' if i == best_idx else 'normal')
        axes[0, i].axis('off')
        
        if i == best_idx:
            axes[0, i].add_patch(plt.Rectangle((0, 0), 63, 63, fill=False, edgecolor='green', linewidth=3))
    
    # Row 2: Reconstructions
    for i in range(min(4, len(labels))):
        digit = labels[i].item()
        gen_img = generated_images[i].cpu().numpy().transpose(1, 2, 0)
        gen_img = np.clip(gen_img, 0, 1)
        
        axes[1, i].imshow(gen_img)
        title_color = 'green' if i == best_idx else 'black'
        psnr = sample_metrics[i]['psnr']
        cosine = sample_metrics[i]['cosine_similarity']
        axes[1, i].set_title(f'Recon Digit {digit}\nPSNR: {psnr:.1f}dB\nCos: {cosine:.3f}', 
                           color=title_color, fontweight='bold' if i == best_idx else 'normal', fontsize=9)
        axes[1, i].axis('off')
        
        if i == best_idx:
            axes[1, i].add_patch(plt.Rectangle((0, 0), 63, 63, fill=False, edgecolor='green', linewidth=3))
    
    # Row 3: EEG signals (first 4 channels)
    for i in range(min(4, len(labels))):
        eeg_sample = eeg_data[i, :4, :].cpu().numpy()  # First 4 channels
        
        for ch in range(4):
            axes[2, i].plot(eeg_sample[ch], alpha=0.7, label=f'Ch{ch+1}')
        
        title_color = 'green' if i == best_idx else 'black'
        axes[2, i].set_title(f'EEG Sample {i} (Digit {labels[i].item()})', 
                           color=title_color, fontweight='bold' if i == best_idx else 'normal')
        axes[2, i].set_xlabel('Time')
        axes[2, i].set_ylabel('Amplitude')
        axes[2, i].grid(True, alpha=0.3)
        
        if i == best_idx:
            axes[2, i].patch.set_edgecolor('green')
            axes[2, i].patch.set_linewidth(3)
    
    # Row 4: Metrics comparison
    metrics_names = ['PSNR (dB)', 'Cosine Sim', 'Correlation', 'MSE']
    metrics_values = [
        [m['psnr'] for m in sample_metrics[:4]],
        [m['cosine_similarity'] for m in sample_metrics[:4]],
        [m['correlation'] for m in sample_metrics[:4]],
        [m['mse'] for m in sample_metrics[:4]]
    ]
    
    for i, (name, values) in enumerate(zip(metrics_names, metrics_values)):
        bars = axes[3, i].bar(range(len(values)), values)
        
        # Highlight best sample
        if best_idx < len(values):
            bars[best_idx].set_color('green')
            bars[best_idx].set_alpha(0.8)
        
        axes[3, i].set_title(name, fontweight='bold')
        axes[3, i].set_xlabel('Sample Index')
        axes[3, i].set_ylabel('Value')
        axes[3, i].set_xticks(range(len(values)))
        axes[3, i].set_xticklabels([f'S{j}' for j in range(len(values))])
        axes[3, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('detailed_reconstruction_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Detailed analysis visualization saved to: detailed_reconstruction_analysis.png")

def provide_optimization_recommendations(sample_metrics, eeg_stats, best_idx):
    """Provide recommendations for optimizing other reconstructions"""
    
    print("OPTIMIZATION RECOMMENDATIONS:")
    print("="*50)
    
    best_sample = sample_metrics[best_idx]
    
    print(f"\n🎯 TARGET QUALITY (Sample {best_idx}, Digit {best_sample['digit']}):")
    print(f"   PSNR: {best_sample['psnr']:.2f} dB")
    print(f"   Cosine Similarity: {best_sample['cosine_similarity']:.4f}")
    print(f"   Correlation: {best_sample['correlation']:.4f}")
    
    print(f"\n📊 KEY SUCCESS FACTORS:")
    print(f"   1. EEG Signal Quality:")
    print(f"      - Signal-to-Noise Ratio: {eeg_stats['best_snr']:.4f}")
    print(f"      - Signal Energy: {eeg_stats['best_energy']:.4f}")
    print(f"      - Signal Stability: {eeg_stats['best_std']:.4f}")
    
    print(f"\n🔧 OPTIMIZATION STRATEGIES:")
    print(f"   1. DATA PREPROCESSING:")
    print(f"      - Apply adaptive filtering to improve SNR")
    print(f"      - Normalize signal energy across samples")
    print(f"      - Remove artifacts and noise")
    
    print(f"   2. MODEL ARCHITECTURE:")
    print(f"      - Add attention mechanism to focus on high-quality EEG features")
    print(f"      - Implement adaptive loss weighting based on signal quality")
    print(f"      - Use ensemble of models trained on different signal characteristics")
    
    print(f"   3. TRAINING STRATEGY:")
    print(f"      - Weight training samples by reconstruction quality")
    print(f"      - Use curriculum learning (easy → hard samples)")
    print(f"      - Implement progressive training with quality-based scheduling")
    
    print(f"   4. LOSS FUNCTION:")
    print(f"      - Add perceptual loss for better visual quality")
    print(f"      - Include feature matching loss")
    print(f"      - Use adaptive loss scaling based on signal characteristics")
    
    # Identify worst samples for specific recommendations
    worst_samples = sorted(sample_metrics, key=lambda x: x['psnr'])[:2]
    
    print(f"\n⚠️  SAMPLES NEEDING MOST IMPROVEMENT:")
    for sample in worst_samples:
        print(f"   Sample {sample['sample_idx']} (Digit {sample['digit']}):")
        print(f"     Current PSNR: {sample['psnr']:.2f} dB (Target: {best_sample['psnr']:.2f} dB)")
        print(f"     Improvement needed: {best_sample['psnr'] - sample['psnr']:.2f} dB")
        print(f"     Current Cosine: {sample['cosine_similarity']:.4f} (Target: {best_sample['cosine_similarity']:.4f})")

if __name__ == "__main__":
    analyze_best_reconstruction()
