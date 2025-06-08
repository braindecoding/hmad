#!/usr/bin/env python3
"""
Full Training HMADv2 with Proper Train/Test Split
=================================================

Full training implementation dengan:
1. Proper train/validation/test split
2. Extended training epochs
3. Learning rate scheduling
4. Model checkpointing
5. Comprehensive evaluation
6. Real-time monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from evaluation.test_hmad import load_mindbigdata_sample, load_crell_sample, load_stimulus_images
from models.hmadv2 import create_improved_hmad_model, ProgressiveTrainer
import time
import os
import random

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def create_target_images_from_labels(labels, dataset_type, stimulus_images, image_size=64, device='cpu'):
    """Create target images berdasarkan labels menggunakan stimulus ASLI dari dataset"""
    batch_size = len(labels)
    target_images = torch.zeros(batch_size, 3, image_size, image_size, device=device)
    
    for i, label in enumerate(labels):
        try:
            if dataset_type == 'mindbigdata':
                digit = label.item() if torch.is_tensor(label) else label
                stimulus_key = f'digit_{digit}'
                if stimulus_key in stimulus_images:
                    target_images[i] = stimulus_images[stimulus_key].to(device)
            else:  # crell
                letter_names = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
                letter_idx = label.item() if torch.is_tensor(label) else label
                if letter_idx < len(letter_names):
                    letter = letter_names[letter_idx]
                    stimulus_key = f'letter_{letter}'
                    if stimulus_key in stimulus_images:
                        target_images[i] = stimulus_images[stimulus_key].to(device)
        except Exception as e:
            print(f"Error loading stimulus for label {label}: {e}")
    
    return target_images

def create_train_val_test_split(dataset_type, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    """Create proper train/validation/test split"""
    
    print(f"\nCreating train/val/test split for {dataset_type}...")
    
    # Load maximum available data
    if dataset_type == 'mindbigdata':
        eeg_data, labels = load_mindbigdata_sample("data/raw/datasets/EP1.01.txt", max_samples=200)
    else:  # crell
        eeg_data, labels = load_crell_sample("data/raw/datasets/S01.mat", max_samples=50)
    
    if eeg_data is None or labels is None:
        print(f"Failed to load {dataset_type} data")
        return None
    
    print(f"Total samples: {len(labels)}")
    label_dist = Counter(labels.tolist())
    print(f"Label distribution: {dict(label_dist)}")
    
    # Convert to numpy
    eeg_numpy = eeg_data.cpu().numpy()
    labels_numpy = labels.cpu().numpy()
    
    # First split: train+val vs test
    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            eeg_numpy, labels_numpy,
            test_size=test_size,
            stratify=labels_numpy,
            random_state=random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=random_state
        )
        print("✓ Using stratified splits")
        
    except ValueError:
        print("⚠️  Using random splits (insufficient samples for stratification)")
        X_temp, X_test, y_temp, y_test = train_test_split(
            eeg_numpy, labels_numpy,
            test_size=test_size,
            random_state=random_state
        )
        
        val_size_adjusted = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state
        )
    
    # Convert back to tensors
    splits = {
        'train': {
            'eeg': torch.from_numpy(X_train).float(),
            'labels': torch.from_numpy(y_train).long()
        },
        'val': {
            'eeg': torch.from_numpy(X_val).float(),
            'labels': torch.from_numpy(y_val).long()
        },
        'test': {
            'eeg': torch.from_numpy(X_test).float(),
            'labels': torch.from_numpy(y_test).long()
        }
    }
    
    # Print split statistics
    for split_name, split_data in splits.items():
        split_dist = Counter(split_data['labels'].numpy())
        print(f"  {split_name}: {len(split_data['labels'])} samples, dist: {dict(split_dist)}")
    
    return splits

def full_training_loop(model, data_splits, dataset_type, stimulus_images, device, 
                      epochs=100, patience=15, save_dir='checkpoints'):
    """Full training loop with validation and checkpointing"""
    
    print(f"\nStarting full training for {dataset_type}...")
    print(f"Training epochs: {epochs}, Early stopping patience: {patience}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Move data to device
    train_eeg = data_splits['train']['eeg'].to(device)
    train_labels = data_splits['train']['labels'].to(device)
    val_eeg = data_splits['val']['eeg'].to(device)
    val_labels = data_splits['val']['labels'].to(device)
    
    # Create target images
    train_targets = create_target_images_from_labels(
        train_labels, dataset_type, stimulus_images, 64, device
    )
    val_targets = create_target_images_from_labels(
        val_labels, dataset_type, stimulus_images, 64, device
    )
    
    # Create trainer with enhanced settings
    trainer = ProgressiveTrainer(model, device)
    
    # Enhanced learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
    )
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_psnr': [],
        'val_psnr': [],
        'train_cosine': [],
        'val_cosine': [],
        'learning_rate': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    print(f"\nTraining on {len(train_labels)} samples, validating on {len(val_labels)} samples")
    print("="*70)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_metrics = trainer.train_step(train_eeg, train_targets, dataset_type)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_eeg, dataset_type, val_targets)
            val_loss = val_outputs['total_loss'].item()
            
            # Compute validation metrics
            val_metrics = compute_detailed_metrics(
                val_outputs['generated_images'], val_targets
            )
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Record history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_loss)
        history['train_psnr'].append(train_metrics.get('psnr', 0))
        history['val_psnr'].append(val_metrics['avg_psnr'])
        history['train_cosine'].append(train_metrics.get('cosine_similarity', 0))
        history['val_cosine'].append(val_metrics['avg_cosine'])
        history['learning_rate'].append(current_lr)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            best_model_path = os.path.join(save_dir, f'best_{dataset_type}_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'history': history
            }, best_model_path)
            
        else:
            patience_counter += 1
        
        # Print progress
        epoch_time = time.time() - epoch_start
        if (epoch + 1) % 5 == 0 or epoch < 10:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val PSNR: {val_metrics['avg_psnr']:.2f}dB | "
                  f"Val Cosine: {val_metrics['avg_cosine']:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s")
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(save_dir, f'{dataset_type}_checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
            break
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    print(f"Best model saved at epoch {best_epoch} with val_loss: {best_val_loss:.4f}")
    
    # Load best model
    best_model_path = os.path.join(save_dir, f'best_{dataset_type}_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded best model from epoch {checkpoint['epoch']}")
    
    return history

def compute_detailed_metrics(generated, target):
    """Compute detailed quality metrics"""
    batch_size = generated.shape[0]
    psnr_values = []
    cosine_values = []
    ssim_values = []
    
    for i in range(batch_size):
        # PSNR
        mse = F.mse_loss(generated[i:i+1], target[i:i+1])
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        psnr_values.append(psnr.item())
        
        # Cosine similarity
        gen_flat = generated[i:i+1].view(1, -1)
        target_flat = target[i:i+1].view(1, -1)
        cosine = F.cosine_similarity(gen_flat, target_flat, dim=1)
        cosine_values.append(cosine.item())
        
        # Simplified SSIM
        gen_mean = generated[i].mean()
        target_mean = target[i].mean()
        gen_std = generated[i].std()
        target_std = target[i].std()
        
        covariance = ((generated[i] - gen_mean) * (target[i] - target_mean)).mean()
        ssim = (2 * gen_mean * target_mean + 1e-8) / (gen_mean**2 + target_mean**2 + 1e-8) * \
               (2 * covariance + 1e-8) / (gen_std**2 + target_std**2 + 1e-8)
        ssim_values.append(ssim.item())
    
    return {
        'psnr_values': psnr_values,
        'cosine_values': cosine_values,
        'ssim_values': ssim_values,
        'avg_psnr': np.mean(psnr_values),
        'avg_cosine': np.mean(cosine_values),
        'avg_ssim': np.mean(ssim_values),
        'std_psnr': np.std(psnr_values),
        'std_cosine': np.std(cosine_values),
        'std_ssim': np.std(ssim_values)
    }

def evaluate_final_model(model, data_splits, dataset_type, stimulus_images, device):
    """Final evaluation on test set"""
    
    print(f"\nFinal evaluation on {dataset_type} test set...")
    
    test_eeg = data_splits['test']['eeg'].to(device)
    test_labels = data_splits['test']['labels'].to(device)
    test_targets = create_target_images_from_labels(
        test_labels, dataset_type, stimulus_images, 64, device
    )
    
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_eeg, dataset_type, test_targets)
        generated_images = test_outputs['generated_images']
        
        # Compute comprehensive metrics
        test_metrics = compute_detailed_metrics(generated_images, test_targets)
        
        print(f"✓ Final test results ({len(test_labels)} samples):")
        print(f"  PSNR: {test_metrics['avg_psnr']:.2f} ± {test_metrics['std_psnr']:.2f} dB")
        print(f"  Cosine Similarity: {test_metrics['avg_cosine']:.4f} ± {test_metrics['std_cosine']:.4f}")
        print(f"  SSIM: {test_metrics['avg_ssim']:.4f} ± {test_metrics['std_ssim']:.4f}")
        
        return test_metrics, generated_images, test_targets

def create_training_visualization(history, dataset_type):
    """Create comprehensive training visualization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Full Training Results - {dataset_type.upper()}', fontsize=16, fontweight='bold')
    
    epochs = history['epoch']
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PSNR curves
    axes[0, 1].plot(epochs, history['val_psnr'], 'g-', label='Validation PSNR', linewidth=2)
    axes[0, 1].axhline(y=15, color='orange', linestyle='--', alpha=0.7, label='Target (15dB)')
    axes[0, 1].set_title('PSNR Progress')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cosine similarity curves
    axes[0, 2].plot(epochs, history['val_cosine'], 'purple', label='Validation Cosine', linewidth=2)
    axes[0, 2].axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='Target (0.3)')
    axes[0, 2].set_title('Cosine Similarity Progress')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Cosine Similarity')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(epochs, history['learning_rate'], 'orange', linewidth=2)
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training summary
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    final_val_psnr = history['val_psnr'][-1]
    final_val_cosine = history['val_cosine'][-1]
    best_val_psnr = max(history['val_psnr'])
    best_val_cosine = max(history['val_cosine'])
    
    summary_text = f"""
TRAINING SUMMARY:

Final Results:
• Train Loss: {final_train_loss:.4f}
• Val Loss: {final_val_loss:.4f}
• Val PSNR: {final_val_psnr:.2f} dB
• Val Cosine: {final_val_cosine:.4f}

Best Results:
• Best PSNR: {best_val_psnr:.2f} dB
• Best Cosine: {best_val_cosine:.4f}

Training Info:
• Total Epochs: {len(epochs)}
• Final LR: {history['learning_rate'][-1]:.2e}
• Convergence: {'✓ Good' if final_val_loss < 0.2 else '⚠️ Check'}
"""
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    axes[1, 1].axis('off')
    
    # Success indicators
    success_text = f"""
SUCCESS INDICATORS:

PSNR Target (>7 dB):
{'✅ ACHIEVED' if final_val_psnr > 7 else '⚠️ In Progress'}

Cosine Target (>0.4):
{'✅ ACHIEVED' if final_val_cosine > 0.4 else '⚠️ In Progress'}

Training Stability:
{'✅ STABLE' if abs(final_train_loss - final_val_loss) < 0.1 else '⚠️ Check Overfitting'}

Convergence:
{'✅ CONVERGED' if final_val_loss < 0.2 else '⚠️ May Need More Training'}
"""
    
    axes[1, 2].text(0.05, 0.95, success_text, transform=axes[1, 2].transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'full_training_{dataset_type}_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training visualization saved: full_training_{dataset_type}_results.png")

def full_training_hmadv2():
    """Main function for full training"""
    
    print("="*70)
    print("FULL TRAINING HMADV2 WITH PROPER TRAIN/VAL/TEST SPLIT")
    print("="*70)
    
    # Set random seeds
    set_random_seeds(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load stimulus images
    stimulus_images = load_stimulus_images("data/raw/datasets", image_size=64)
    print(f"Loaded {len(stimulus_images)} stimulus images")
    
    # Configuration
    config = {
        'mindbigdata_channels': 14,
        'crell_channels': 64,
        'd_model': 256,
        'image_size': 64
    }
    
    # Process each dataset
    for dataset_type in ['mindbigdata', 'crell']:
        print(f"\n{'='*70}")
        print(f"FULL TRAINING: {dataset_type.upper()}")
        print(f"{'='*70}")
        
        # Create data splits
        data_splits = create_train_val_test_split(dataset_type)
        if data_splits is None:
            continue
        
        # Create fresh model
        model = create_improved_hmad_model(config)
        model = model.to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Full training
        history = full_training_loop(
            model, data_splits, dataset_type, stimulus_images, device,
            epochs=100, patience=15
        )
        
        # Create training visualization
        create_training_visualization(history, dataset_type)
        
        # Final evaluation on test set
        test_metrics, test_generated, test_targets = evaluate_final_model(
            model, data_splits, dataset_type, stimulus_images, device
        )
        
        # Save final results
        final_results = {
            'dataset_type': dataset_type,
            'config': config,
            'data_splits_info': {
                'train_samples': len(data_splits['train']['labels']),
                'val_samples': len(data_splits['val']['labels']),
                'test_samples': len(data_splits['test']['labels'])
            },
            'training_history': history,
            'final_test_metrics': test_metrics
        }
        
        results_path = f'full_training_{dataset_type}_results.pkl'
        torch.save(final_results, results_path)
        print(f"✓ Results saved: {results_path}")
    
    print(f"\n{'='*70}")
    print("FULL TRAINING COMPLETED!")
    print(f"{'='*70}")
    print("✓ Models trained with proper train/val/test splits")
    print("✓ Early stopping and checkpointing implemented")
    print("✓ Comprehensive evaluation completed")
    print("✓ Results saved and visualized")

if __name__ == "__main__":
    full_training_hmadv2()
