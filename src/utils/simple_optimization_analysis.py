#!/usr/bin/env python3
"""
Simple Optimization Analysis
============================

Analisis sederhana untuk memahami mengapa rekonstruksi tertentu lebih baik
dan cara meningkatkan kualitas semua rekonstruksi.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        except Exception as e:
            print(f"Error loading stimulus for label {label.item()}: {e}")
    
    return target_images

def simple_optimization_analysis():
    """Analisis sederhana untuk optimasi rekonstruksi"""
    
    print("="*70)
    print("SIMPLE OPTIMIZATION ANALYSIS")
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
    
    # Load data
    print("\n2. LOADING DATA...")
    mindbig_eeg, mindbig_labels = load_mindbigdata_sample("datasets/EP1.01.txt", max_samples=8)
    stimulus_images = load_stimulus_images("datasets", image_size=64)
    
    mindbig_eeg = mindbig_eeg.to(device)
    mindbig_labels = mindbig_labels.to(device)
    target_images = create_target_images_from_labels(
        mindbig_labels, 'mindbigdata', stimulus_images, 64, device
    )
    
    print(f"✓ Data loaded: {mindbig_eeg.shape}")
    print(f"✓ Labels: {mindbig_labels.tolist()}")
    
    # Generate reconstructions
    print("\n3. GENERATING RECONSTRUCTIONS...")
    model.eval()
    with torch.no_grad():
        outputs = model(mindbig_eeg, 'mindbigdata', target_images)
        generated_images = outputs['generated_images']
        latent_features = outputs['latent_features']
        attention_weights = outputs['attention_weights']
    
    # Analyze individual samples
    print("\n4. ANALYZING INDIVIDUAL SAMPLES...")
    sample_analysis = []
    
    for i in range(len(mindbig_labels)):
        digit = mindbig_labels[i].item()
        
        # Compute metrics
        target_single = target_images[i:i+1]
        generated_single = generated_images[i:i+1]
        
        mse = F.mse_loss(generated_single, target_single)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        gen_flat = generated_single.view(1, -1)
        target_flat = target_single.view(1, -1)
        cosine_sim = F.cosine_similarity(gen_flat, target_flat, dim=1)
        
        # EEG signal characteristics
        eeg_sample = mindbig_eeg[i]
        eeg_mean = eeg_sample.mean().item()
        eeg_std = eeg_sample.std().item()
        eeg_energy = (eeg_sample ** 2).mean().item()
        
        # Latent features characteristics
        latent_sample = latent_features[i]
        latent_mean = latent_sample.mean().item()
        latent_std = latent_sample.std().item()
        latent_norm = torch.norm(latent_sample).item()
        
        analysis = {
            'sample_idx': i,
            'digit': digit,
            'mse': mse.item(),
            'psnr': psnr.item(),
            'cosine_similarity': cosine_sim.item(),
            'eeg_mean': eeg_mean,
            'eeg_std': eeg_std,
            'eeg_energy': eeg_energy,
            'latent_mean': latent_mean,
            'latent_std': latent_std,
            'latent_norm': latent_norm
        }
        
        sample_analysis.append(analysis)
        
        print(f"Sample {i} (Digit {digit}):")
        print(f"  PSNR: {analysis['psnr']:.2f} dB")
        print(f"  Cosine Sim: {analysis['cosine_similarity']:.4f}")
        print(f"  EEG Energy: {analysis['eeg_energy']:.0f}")
        print(f"  Latent Norm: {analysis['latent_norm']:.2f}")
    
    # Find best and worst samples
    best_idx = max(range(len(sample_analysis)), key=lambda i: sample_analysis[i]['psnr'])
    worst_idx = min(range(len(sample_analysis)), key=lambda i: sample_analysis[i]['psnr'])
    
    print(f"\n5. BEST vs WORST ANALYSIS...")
    print(f"BEST: Sample {best_idx} (Digit {sample_analysis[best_idx]['digit']})")
    print(f"  PSNR: {sample_analysis[best_idx]['psnr']:.2f} dB")
    print(f"  Cosine: {sample_analysis[best_idx]['cosine_similarity']:.4f}")
    print(f"  EEG Energy: {sample_analysis[best_idx]['eeg_energy']:.0f}")
    print(f"  Latent Norm: {sample_analysis[best_idx]['latent_norm']:.2f}")
    
    print(f"\nWORST: Sample {worst_idx} (Digit {sample_analysis[worst_idx]['digit']})")
    print(f"  PSNR: {sample_analysis[worst_idx]['psnr']:.2f} dB")
    print(f"  Cosine: {sample_analysis[worst_idx]['cosine_similarity']:.4f}")
    print(f"  EEG Energy: {sample_analysis[worst_idx]['eeg_energy']:.0f}")
    print(f"  Latent Norm: {sample_analysis[worst_idx]['latent_norm']:.2f}")
    
    # Create comprehensive visualization
    print("\n6. CREATING COMPREHENSIVE VISUALIZATION...")
    create_comprehensive_analysis_visualization(
        target_images, generated_images, mindbig_labels, sample_analysis, 
        best_idx, worst_idx, attention_weights
    )
    
    # Provide optimization strategies
    print("\n7. OPTIMIZATION STRATEGIES...")
    provide_optimization_strategies(sample_analysis, best_idx, worst_idx)
    
    print(f"\n{'='*70}")
    print("SIMPLE OPTIMIZATION ANALYSIS COMPLETED!")
    print(f"{'='*70}")

def create_comprehensive_analysis_visualization(target_images, generated_images, labels, 
                                              sample_analysis, best_idx, worst_idx, attention_weights):
    """Create comprehensive visualization of analysis"""
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Comprehensive Analysis: Understanding Quality Differences', 
                 fontsize=16, fontweight='bold')
    
    # Row 1: Real stimuli
    for i in range(min(4, len(labels))):
        digit = labels[i].item()
        real_img = target_images[i].cpu().numpy().transpose(1, 2, 0)
        
        axes[0, i].imshow(real_img)
        border_color = 'green' if i == best_idx else ('red' if i == worst_idx else 'black')
        axes[0, i].set_title(f'Real Digit {digit}', color=border_color, fontweight='bold')
        axes[0, i].axis('off')
        
        if i == best_idx or i == worst_idx:
            axes[0, i].add_patch(plt.Rectangle((0, 0), 63, 63, fill=False, 
                                             edgecolor=border_color, linewidth=3))
    
    # Row 2: Reconstructions with quality metrics
    for i in range(min(4, len(labels))):
        digit = labels[i].item()
        gen_img = generated_images[i].cpu().numpy().transpose(1, 2, 0)
        gen_img = np.clip(gen_img, 0, 1)
        
        axes[1, i].imshow(gen_img)
        border_color = 'green' if i == best_idx else ('red' if i == worst_idx else 'black')
        psnr = sample_analysis[i]['psnr']
        cosine = sample_analysis[i]['cosine_similarity']
        
        quality_label = "BEST" if i == best_idx else ("WORST" if i == worst_idx else "")
        title = f'Recon {digit} {quality_label}\nPSNR: {psnr:.1f}dB\nCos: {cosine:.3f}'
        axes[1, i].set_title(title, color=border_color, fontweight='bold', fontsize=9)
        axes[1, i].axis('off')
        
        if i == best_idx or i == worst_idx:
            axes[1, i].add_patch(plt.Rectangle((0, 0), 63, 63, fill=False, 
                                             edgecolor=border_color, linewidth=3))
    
    # Row 3: Quality metrics comparison
    metrics_names = ['PSNR (dB)', 'Cosine Sim', 'EEG Energy', 'Latent Norm']
    metrics_data = [
        [s['psnr'] for s in sample_analysis[:4]],
        [s['cosine_similarity'] for s in sample_analysis[:4]],
        [s['eeg_energy']/1000 for s in sample_analysis[:4]],  # Scale down for visualization
        [s['latent_norm'] for s in sample_analysis[:4]]
    ]
    
    for i, (name, values) in enumerate(zip(metrics_names, metrics_data)):
        bars = axes[2, i].bar(range(len(values)), values)
        
        # Color code bars
        for j, bar in enumerate(bars):
            if j == best_idx:
                bar.set_color('green')
                bar.set_alpha(0.8)
            elif j == worst_idx:
                bar.set_color('red')
                bar.set_alpha(0.8)
            else:
                bar.set_color('blue')
                bar.set_alpha(0.6)
        
        axes[2, i].set_title(name, fontweight='bold')
        axes[2, i].set_xlabel('Sample')
        axes[2, i].set_ylabel('Value')
        axes[2, i].set_xticks(range(len(values)))
        axes[2, i].set_xticklabels([f'S{j}' for j in range(len(values))])
        axes[2, i].grid(True, alpha=0.3)
    
    # Row 4: Analysis summary and recommendations
    best_sample = sample_analysis[best_idx]
    worst_sample = sample_analysis[worst_idx]
    
    summary_text = f"""
QUALITY ANALYSIS SUMMARY:

BEST SAMPLE (Index {best_idx}, Digit {best_sample['digit']}):
• PSNR: {best_sample['psnr']:.2f} dB
• Cosine Similarity: {best_sample['cosine_similarity']:.4f}
• EEG Energy: {best_sample['eeg_energy']:.0f}
• Latent Norm: {best_sample['latent_norm']:.2f}

WORST SAMPLE (Index {worst_idx}, Digit {worst_sample['digit']}):
• PSNR: {worst_sample['psnr']:.2f} dB
• Cosine Similarity: {worst_sample['cosine_similarity']:.4f}
• EEG Energy: {worst_sample['eeg_energy']:.0f}
• Latent Norm: {worst_sample['latent_norm']:.2f}

QUALITY DIFFERENCE:
• PSNR Gap: {best_sample['psnr'] - worst_sample['psnr']:.2f} dB
• Cosine Gap: {best_sample['cosine_similarity'] - worst_sample['cosine_similarity']:.4f}
"""
    
    axes[3, 0].text(0.05, 0.95, summary_text, transform=axes[3, 0].transAxes, 
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    axes[3, 0].axis('off')
    
    # Optimization recommendations
    opt_text = f"""
OPTIMIZATION RECOMMENDATIONS:

1. EEG SIGNAL PREPROCESSING:
   • Normalize energy across samples
   • Apply adaptive filtering
   • Remove artifacts consistently

2. MODEL ARCHITECTURE:
   • Add quality-aware attention
   • Implement adaptive loss weighting
   • Use feature normalization

3. TRAINING STRATEGY:
   • Weight samples by quality
   • Progressive difficulty training
   • Ensemble of specialized models

4. LOSS FUNCTION:
   • Add perceptual loss
   • Include feature matching
   • Quality-aware weighting
"""
    
    axes[3, 1].text(0.05, 0.95, opt_text, transform=axes[3, 1].transAxes, 
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    axes[3, 1].axis('off')
    
    # Success factors
    success_text = f"""
SUCCESS FACTORS (From Best Sample):

SIGNAL CHARACTERISTICS:
• High EEG energy: {best_sample['eeg_energy']:.0f}
• Stable signal: {best_sample['eeg_std']:.2f}
• Good SNR ratio

LATENT REPRESENTATION:
• Strong features: {best_sample['latent_norm']:.2f}
• Balanced distribution
• Rich information content

TARGET METRICS:
• PSNR > 7.0 dB ✓
• Cosine Sim > 0.4 ✓
• Consistent quality across digits
"""
    
    axes[3, 2].text(0.05, 0.95, success_text, transform=axes[3, 2].transAxes, 
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    axes[3, 2].axis('off')
    
    # Implementation steps
    impl_text = f"""
IMPLEMENTATION STEPS:

IMMEDIATE (Easy):
1. Normalize EEG energy
2. Add gradient clipping
3. Improve weight initialization
4. Use better data augmentation

MEDIUM (Moderate):
1. Implement attention mechanism
2. Add perceptual loss
3. Quality-aware training
4. Progressive learning

ADVANCED (Complex):
1. Multi-scale architecture
2. Adversarial training
3. Meta-learning approach
4. Ensemble methods
"""
    
    axes[3, 3].text(0.05, 0.95, impl_text, transform=axes[3, 3].transAxes, 
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
    axes[3, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('comprehensive_optimization_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Comprehensive analysis saved to: comprehensive_optimization_analysis.png")

def provide_optimization_strategies(sample_analysis, best_idx, worst_idx):
    """Provide specific optimization strategies"""
    
    best_sample = sample_analysis[best_idx]
    worst_sample = sample_analysis[worst_idx]
    
    print("SPECIFIC OPTIMIZATION STRATEGIES:")
    print("="*50)
    
    print(f"\n🎯 TARGET QUALITY (Best Sample):")
    print(f"   PSNR: {best_sample['psnr']:.2f} dB")
    print(f"   Cosine Similarity: {best_sample['cosine_similarity']:.4f}")
    print(f"   EEG Energy: {best_sample['eeg_energy']:.0f}")
    
    print(f"\n📊 IMPROVEMENT NEEDED (Worst Sample):")
    print(f"   Current PSNR: {worst_sample['psnr']:.2f} dB")
    print(f"   Target PSNR: {best_sample['psnr']:.2f} dB")
    print(f"   Gap: {best_sample['psnr'] - worst_sample['psnr']:.2f} dB")
    
    print(f"\n🔧 IMMEDIATE ACTIONS:")
    print(f"   1. EEG Energy Normalization:")
    print(f"      - Best sample energy: {best_sample['eeg_energy']:.0f}")
    print(f"      - Worst sample energy: {worst_sample['eeg_energy']:.0f}")
    print(f"      - Normalize all samples to target energy level")
    
    print(f"   2. Latent Feature Enhancement:")
    print(f"      - Best latent norm: {best_sample['latent_norm']:.2f}")
    print(f"      - Worst latent norm: {worst_sample['latent_norm']:.2f}")
    print(f"      - Apply feature normalization and enhancement")
    
    print(f"   3. Quality-Aware Training:")
    print(f"      - Weight training samples by current quality")
    print(f"      - Focus more on improving worst samples")
    print(f"      - Use curriculum learning strategy")

if __name__ == "__main__":
    simple_optimization_analysis()
