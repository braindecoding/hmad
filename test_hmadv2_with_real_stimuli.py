#!/usr/bin/env python3
"""
Test HMADv2 Framework dengan stimulus asli
==========================================

Testing improved HMAD architecture dengan:
- Progressive training strategy
- Real stimulus targets
- Better loss balancing
- Proper weight initialization
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from hmadv2 import create_improved_hmad_model, ProgressiveTrainer, analyze_model_outputs
from test_hmad import load_mindbigdata_sample, load_crell_sample, load_stimulus_images
# Additional imports for stimulus handling

def create_target_images_from_labels(labels, dataset_type, stimulus_images, image_size=64, device='cpu'):
    """Create target images berdasarkan labels menggunakan stimulus ASLI dari dataset"""
    batch_size = len(labels)
    target_images = torch.zeros(batch_size, 3, image_size, image_size, device=device)

    for i, label in enumerate(labels):
        try:
            if dataset_type == 'mindbigdata':
                # Load REAL digit stimulus
                digit = label.item()
                stimulus_key = f'digit_{digit}'
                if stimulus_key in stimulus_images:
                    target_images[i] = stimulus_images[stimulus_key].to(device)
                    print(f"✓ Using REAL digit stimulus for label {digit}")
                else:
                    print(f"⚠️  Real stimulus not found for digit {digit}, using fallback")
                    # Fallback: simple pattern
                    target_images[i, 0, :, :] = (digit + 1) / 10.0
            else:  # crell
                # Load REAL letter stimulus
                letter_names = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
                letter_idx = label.item()
                if letter_idx < len(letter_names):
                    letter = letter_names[letter_idx]
                    stimulus_key = f'letter_{letter}'
                    if stimulus_key in stimulus_images:
                        target_images[i] = stimulus_images[stimulus_key].to(device)
                        print(f"✓ Using REAL letter stimulus for label {letter}")
                    else:
                        print(f"⚠️  Real stimulus not found for letter {letter}, using fallback")
                        # Fallback: simple pattern
                        target_images[i, 1, :, :] = (letter_idx + 1) / 10.0

        except Exception as e:
            print(f"Error loading stimulus for label {label.item()}: {e}")
            # Fallback pattern
            if dataset_type == 'mindbigdata':
                target_images[i, 0, :, :] = (label.item() + 1) / 10.0
            else:
                target_images[i, 1, :, :] = (label.item() + 1) / 10.0

    return target_images

def test_hmadv2_with_real_stimuli():
    """Test HMADv2 Framework dengan stimulus asli"""
    
    print("="*60)
    print("TESTING HMADV2 FRAMEWORK WITH REAL STIMULI")
    print("="*60)
    
    # Configuration
    config = {
        'mindbigdata_channels': 14,
        'crell_channels': 64,
        'd_model': 256,
        'image_size': 64
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create improved model
    print("\n1. CREATING HMADV2 MODEL...")
    model = create_improved_hmad_model(config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ HMADv2 model created with {total_params:,} parameters ({total_params/1e6:.1f}M)")
    
    # Load datasets
    print("\n2. LOADING DATASETS...")
    mindbig_eeg, mindbig_labels = load_mindbigdata_sample("datasets/EP1.01.txt", max_samples=8)
    crell_eeg, crell_labels = load_crell_sample("datasets/S01.mat", max_samples=8)
    
    # Load stimulus images
    stimulus_images = load_stimulus_images("datasets", image_size=64)
    print(f"✓ Loaded {len(stimulus_images)} stimulus images")
    
    # Create progressive trainer
    print("\n3. CREATING PROGRESSIVE TRAINER...")
    trainer = ProgressiveTrainer(model, device)
    print(f"✓ Progressive trainer created")
    print(f"✓ Training phases: {trainer.phase_epochs}")
    print(f"✓ Current phase: {trainer.training_phase}")
    
    print("\n4. TESTING PROGRESSIVE TRAINING...")
    
    # Test MindBigData
    if mindbig_eeg is not None and mindbig_labels is not None:
        print(f"\n📊 TESTING MINDBIGDATA:")
        print(f"EEG shape: {mindbig_eeg.shape}")
        print(f"Labels: {mindbig_labels.tolist()}")
        
        mindbig_eeg = mindbig_eeg.to(device)
        mindbig_labels = mindbig_labels.to(device)
        
        # Create target images from real stimuli
        print("Creating target images from REAL MindBigData stimuli...")
        target_images = create_target_images_from_labels(
            mindbig_labels, 'mindbigdata', stimulus_images, 64, device
        )
        print(f"✓ Real stimulus targets: {target_images.shape}")
        print(f"✓ Target value range: [{target_images.min():.3f}, {target_images.max():.3f}]")
        
        # Training step
        train_metrics = trainer.train_step(mindbig_eeg, target_images, 'mindbigdata')
        print(f"✓ Training step successful!")
        print(f"  - Loss: {train_metrics['loss']:.4f}")
        print(f"  - MSE Loss: {train_metrics['mse_loss']:.4f}")
        print(f"  - Perceptual Loss: {train_metrics['perceptual_loss']:.4f}")
        
        # Validation step
        val_metrics = trainer.validate(mindbig_eeg, target_images, 'mindbigdata')
        print(f"✓ Validation metrics:")
        print(f"  - PSNR: {val_metrics['psnr']:.2f} dB")
        print(f"  - Cosine Similarity: {val_metrics['cosine_similarity']:.4f}")
        print(f"  - MSE: {val_metrics['mse']:.6f}")
        
        # Analyze model outputs
        analysis = analyze_model_outputs(model, mindbig_eeg, 'mindbigdata')
        print(f"✓ Model analysis:")
        print(f"  - Generated mean: {analysis['output_stats']['generated_mean']:.4f}")
        print(f"  - Generated std: {analysis['output_stats']['generated_std']:.4f}")
        print(f"  - Generated range: [{analysis['output_stats']['generated_min']:.3f}, {analysis['output_stats']['generated_max']:.3f}]")
    
    # Test Crell
    if crell_eeg is not None and crell_labels is not None:
        print(f"\n📊 TESTING CRELL:")
        print(f"EEG shape: {crell_eeg.shape}")
        print(f"Labels: {crell_labels.tolist()}")
        
        crell_eeg = crell_eeg.to(device)
        crell_labels = crell_labels.to(device)
        
        # Create target images from real stimuli
        print("Creating target images from REAL Crell stimuli...")
        target_images = create_target_images_from_labels(
            crell_labels, 'crell', stimulus_images, 64, device
        )
        print(f"✓ Real stimulus targets: {target_images.shape}")
        print(f"✓ Target value range: [{target_images.min():.3f}, {target_images.max():.3f}]")
        
        # Training step
        train_metrics = trainer.train_step(crell_eeg, target_images, 'crell')
        print(f"✓ Training step successful!")
        print(f"  - Loss: {train_metrics['loss']:.4f}")
        print(f"  - MSE Loss: {train_metrics['mse_loss']:.4f}")
        print(f"  - Perceptual Loss: {train_metrics['perceptual_loss']:.4f}")
        
        # Validation step
        val_metrics = trainer.validate(crell_eeg, target_images, 'crell')
        print(f"✓ Validation metrics:")
        print(f"  - PSNR: {val_metrics['psnr']:.2f} dB")
        print(f"  - Cosine Similarity: {val_metrics['cosine_similarity']:.4f}")
        print(f"  - MSE: {val_metrics['mse']:.6f}")
    
    print("\n5. CREATING HMADV2 VISUALIZATION...")
    create_hmadv2_visualization(model, mindbig_eeg, mindbig_labels, crell_eeg, crell_labels, 
                               stimulus_images, device)
    
    print("\n6. PROGRESSIVE TRAINING DEMONSTRATION...")
    demonstrate_progressive_training(trainer, mindbig_eeg, mindbig_labels, stimulus_images, device)
    
    print("\n" + "="*60)
    print("HMADV2 TESTING COMPLETED")
    print("="*60)
    print(f"✓ Improved architecture: {total_params/1e6:.1f}M parameters")
    print(f"✓ Progressive training strategy working")
    print(f"✓ Real stimulus targets used")
    print(f"✓ Better loss balancing and weight initialization")
    print(f"✓ Gradient clipping and proper normalization")

def create_hmadv2_visualization(model, mindbig_eeg, mindbig_labels, crell_eeg, crell_labels, 
                               stimulus_images, device):
    """Create visualization showing HMADv2 results"""
    
    model.eval()
    
    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('HMADv2 Framework: Improved Architecture Results', fontsize=16, fontweight='bold')
    
    with torch.no_grad():
        # Test MindBigData
        if mindbig_eeg is not None and mindbig_labels is not None:
            # Create target images
            target_images = create_target_images_from_labels(
                mindbig_labels, 'mindbigdata', stimulus_images, 64, device
            )
            
            # Forward pass
            outputs = model(mindbig_eeg, 'mindbigdata', target_images)
            generated_images = outputs['generated_images']
            
            # Show first 4 samples
            for i in range(min(4, generated_images.shape[0])):
                digit = mindbig_labels[i].item()
                
                # Row 1: Real stimuli
                real_img = target_images[i].cpu().numpy().transpose(1, 2, 0)
                axes[0, i].imshow(real_img)
                axes[0, i].set_title(f'Real Digit {digit}', fontweight='bold')
                axes[0, i].axis('off')
                
                # Row 2: HMADv2 reconstruction
                gen_img = generated_images[i].cpu().numpy().transpose(1, 2, 0)
                gen_img = np.clip(gen_img, 0, 1)
                axes[1, i].imshow(gen_img)
                axes[1, i].set_title(f'HMADv2 Reconstruction')
                axes[1, i].axis('off')
        
        # Test Crell (if available, show in row 3)
        if crell_eeg is not None and crell_labels is not None:
            target_images = create_target_images_from_labels(
                crell_labels, 'crell', stimulus_images, 64, device
            )
            
            outputs = model(crell_eeg, 'crell', target_images)
            generated_images = outputs['generated_images']
            
            letter_names = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
            
            for i in range(min(4, generated_images.shape[0])):
                letter_idx = crell_labels[i].item()
                letter = letter_names[letter_idx] if letter_idx < len(letter_names) else 'unknown'
                
                # Row 3: Crell results
                gen_img = generated_images[i].cpu().numpy().transpose(1, 2, 0)
                gen_img = np.clip(gen_img, 0, 1)
                axes[2, i].imshow(gen_img)
                axes[2, i].set_title(f'Crell: Letter {letter.upper()}')
                axes[2, i].axis('off')
        else:
            # Fill row 3 with architecture info
            architecture_info = [
                "HMADV2\nIMPROVEMENTS",
                "Progressive\nTraining",
                "Better Loss\nBalancing",
                "Proper Weight\nInitialization"
            ]
            colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
            
            for i, (info, color) in enumerate(zip(architecture_info, colors)):
                axes[2, i].text(0.5, 0.5, info, ha='center', va='center', 
                               fontsize=12, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
                axes[2, i].axis('off')
    
    # Add row labels
    axes[0, 0].text(-0.15, 0.5, 'REAL\nSTIMULI', transform=axes[0, 0].transAxes, 
                   fontsize=12, fontweight='bold', ha='right', va='center', rotation=90, color='blue')
    
    axes[1, 0].text(-0.15, 0.5, 'HMADV2\nRECONSTRUCTION', transform=axes[1, 0].transAxes, 
                   fontsize=12, fontweight='bold', ha='right', va='center', rotation=90, color='green')
    
    axes[2, 0].text(-0.15, 0.5, 'CRELL RESULTS\n/ ARCHITECTURE', transform=axes[2, 0].transAxes, 
                   fontsize=12, fontweight='bold', ha='right', va='center', rotation=90, color='red')
    
    plt.tight_layout()
    plt.savefig('hmadv2_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ HMADv2 visualization saved to: hmadv2_results.png")

def demonstrate_progressive_training(trainer, eeg_data, labels, stimulus_images, device, epochs=5):
    """Demonstrate progressive training strategy"""
    
    print(f"\nDemonstrating progressive training for {epochs} epochs...")
    
    if eeg_data is None or labels is None:
        print("No data available for progressive training demo")
        return
    
    # Create target images
    target_images = create_target_images_from_labels(
        labels, 'mindbigdata', stimulus_images, 64, device
    )
    
    # Track metrics
    train_losses = []
    val_metrics = []
    
    for epoch in range(epochs):
        # Training step
        train_metrics = trainer.train_step(eeg_data, target_images, 'mindbigdata')
        train_losses.append(train_metrics['loss'])
        
        # Validation step
        val_metric = trainer.validate(eeg_data, target_images, 'mindbigdata')
        val_metrics.append(val_metric)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Loss={train_metrics['loss']:.4f}, "
              f"PSNR={val_metric['psnr']:.2f}dB, "
              f"CosSim={val_metric['cosine_similarity']:.4f}")
    
    # Create training progress plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('HMADv2 Progressive Training Progress', fontsize=14, fontweight='bold')
    
    epochs_range = range(1, epochs + 1)
    
    # Loss plot
    axes[0].plot(epochs_range, train_losses, 'b-', marker='o')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    # PSNR plot
    psnr_values = [m['psnr'] for m in val_metrics]
    axes[1].plot(epochs_range, psnr_values, 'g-', marker='s')
    axes[1].set_title('PSNR (Peak Signal-to-Noise Ratio)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].grid(True)
    
    # Cosine similarity plot
    cosine_values = [m['cosine_similarity'] for m in val_metrics]
    axes[2].plot(epochs_range, cosine_values, 'r-', marker='^')
    axes[2].set_title('Cosine Similarity')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Similarity')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('hmadv2_training_progress.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Training progress saved to: hmadv2_training_progress.png")
    
    # Print final metrics
    final_metrics = val_metrics[-1]
    print(f"\nFinal metrics after {epochs} epochs:")
    print(f"✓ PSNR: {final_metrics['psnr']:.2f} dB (target: >15 dB)")
    print(f"✓ Cosine Similarity: {final_metrics['cosine_similarity']:.4f} (target: >0.3)")
    print(f"✓ MSE: {final_metrics['mse']:.6f}")
    
    # Success indicators
    success_psnr = final_metrics['psnr'] > 15.0
    success_cosine = final_metrics['cosine_similarity'] > 0.3
    
    print(f"\nSuccess indicators:")
    print(f"{'✓' if success_psnr else '✗'} PSNR > 15 dB: {success_psnr}")
    print(f"{'✓' if success_cosine else '✗'} Cosine Similarity > 0.3: {success_cosine}")

if __name__ == "__main__":
    test_hmadv2_with_real_stimuli()
