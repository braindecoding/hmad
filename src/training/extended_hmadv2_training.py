#!/usr/bin/env python3
"""
Extended Training untuk HMADv2 Framework
========================================

Full training protocol:
- Phase 1: Reconstruction (50 epochs) - MSE loss only
- Phase 2: Fine-tuning (50 epochs) - Full loss
- Real stimulus targets
- Comprehensive evaluation metrics
- Model checkpointing
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from hmadv2 import create_improved_hmad_model, ProgressiveTrainer
from test_hmadv2_with_real_stimuli import create_target_images_from_labels
from test_hmad import load_mindbigdata_sample, load_crell_sample, load_stimulus_images

def extended_training_hmadv2():
    """Extended training dengan full protocol"""
    
    print("="*70)
    print("EXTENDED TRAINING - HMADV2 FRAMEWORK")
    print("="*70)
    
    # Configuration
    config = {
        'mindbigdata_channels': 14,
        'crell_channels': 64,
        'd_model': 256,
        'image_size': 64
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("\n1. CREATING HMADV2 MODEL...")
    model = create_improved_hmad_model(config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {total_params:,} parameters ({total_params/1e6:.1f}M)")
    
    # Load datasets
    print("\n2. LOADING DATASETS...")
    mindbig_eeg, mindbig_labels = load_mindbigdata_sample("datasets/EP1.01.txt", max_samples=16)
    crell_eeg, crell_labels = load_crell_sample("datasets/S01.mat", max_samples=16)
    
    # Load stimulus images
    stimulus_images = load_stimulus_images("datasets", image_size=64)
    print(f"✓ Loaded {len(stimulus_images)} stimulus images")
    
    # Create trainer
    trainer = ProgressiveTrainer(model, device)
    
    # Prepare training data
    print("\n3. PREPARING TRAINING DATA...")
    
    # MindBigData
    if mindbig_eeg is not None and mindbig_labels is not None:
        mindbig_eeg = mindbig_eeg.to(device)
        mindbig_labels = mindbig_labels.to(device)
        mindbig_targets = create_target_images_from_labels(
            mindbig_labels, 'mindbigdata', stimulus_images, 64, device
        )
        print(f"✓ MindBigData: {mindbig_eeg.shape} EEG, {mindbig_targets.shape} targets")
    
    # Crell
    if crell_eeg is not None and crell_labels is not None:
        crell_eeg = crell_eeg.to(device)
        crell_labels = crell_labels.to(device)
        crell_targets = create_target_images_from_labels(
            crell_labels, 'crell', stimulus_images, 64, device
        )
        print(f"✓ Crell: {crell_eeg.shape} EEG, {crell_targets.shape} targets")
    
    # Training protocol
    training_protocol = {
        'reconstruction': {'epochs': 25, 'description': 'MSE loss only - basic reconstruction'},
        'fine_tuning': {'epochs': 25, 'description': 'Full loss - perceptual refinement'}
    }
    
    print(f"\n4. TRAINING PROTOCOL:")
    for phase, config in training_protocol.items():
        print(f"   {phase}: {config['epochs']} epochs - {config['description']}")
    
    # Start training
    print(f"\n5. STARTING EXTENDED TRAINING...")
    
    # Training history
    history = {
        'phase': [],
        'epoch': [],
        'mindbig_loss': [],
        'crell_loss': [],
        'mindbig_psnr': [],
        'crell_psnr': [],
        'mindbig_cosine': [],
        'crell_cosine': [],
        'learning_rate': []
    }
    
    total_epochs = sum(config['epochs'] for config in training_protocol.values())
    epoch_counter = 0
    
    for phase_name, phase_config in training_protocol.items():
        print(f"\n{'='*50}")
        print(f"PHASE: {phase_name.upper()}")
        print(f"{'='*50}")
        
        # Set training phase
        trainer.training_phase = phase_name
        
        phase_start_time = time.time()
        
        for epoch in range(phase_config['epochs']):
            epoch_counter += 1
            epoch_start_time = time.time()
            
            # Training steps
            mindbig_metrics = None
            crell_metrics = None
            
            # Train on MindBigData
            if mindbig_eeg is not None:
                mindbig_train = trainer.train_step(mindbig_eeg, mindbig_targets, 'mindbigdata')
                mindbig_val = trainer.validate(mindbig_eeg, mindbig_targets, 'mindbigdata')
                mindbig_metrics = {**mindbig_train, **mindbig_val}
            
            # Train on Crell
            if crell_eeg is not None:
                crell_train = trainer.train_step(crell_eeg, crell_targets, 'crell')
                crell_val = trainer.validate(crell_eeg, crell_targets, 'crell')
                crell_metrics = {**crell_train, **crell_val}
            
            # Update learning rate
            avg_loss = 0
            loss_count = 0
            if mindbig_metrics:
                avg_loss += mindbig_metrics['val_loss']
                loss_count += 1
            if crell_metrics:
                avg_loss += crell_metrics['val_loss']
                loss_count += 1
            
            if loss_count > 0:
                trainer.scheduler.step(avg_loss / loss_count)
            
            # Record history
            history['phase'].append(phase_name)
            history['epoch'].append(epoch_counter)
            history['mindbig_loss'].append(mindbig_metrics['loss'] if mindbig_metrics else 0)
            history['crell_loss'].append(crell_metrics['loss'] if crell_metrics else 0)
            history['mindbig_psnr'].append(mindbig_metrics['psnr'] if mindbig_metrics else 0)
            history['crell_psnr'].append(crell_metrics['psnr'] if crell_metrics else 0)
            history['mindbig_cosine'].append(mindbig_metrics['cosine_similarity'] if mindbig_metrics else 0)
            history['crell_cosine'].append(crell_metrics['cosine_similarity'] if crell_metrics else 0)
            history['learning_rate'].append(trainer.optimizer.param_groups[0]['lr'])
            
            # Print progress
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch_counter:2d}/{total_epochs} ({phase_name}) - {epoch_time:.1f}s")
            
            if mindbig_metrics:
                print(f"  MindBig: Loss={mindbig_metrics['loss']:.4f}, "
                      f"PSNR={mindbig_metrics['psnr']:.2f}dB, "
                      f"CosSim={mindbig_metrics['cosine_similarity']:.4f}")
            
            if crell_metrics:
                print(f"  Crell:   Loss={crell_metrics['loss']:.4f}, "
                      f"PSNR={crell_metrics['psnr']:.2f}dB, "
                      f"CosSim={crell_metrics['cosine_similarity']:.4f}")
            
            print(f"  LR: {trainer.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint every 10 epochs
            if epoch_counter % 10 == 0:
                checkpoint_path = f'hmadv2_checkpoint_epoch_{epoch_counter}.pth'
                torch.save({
                    'epoch': epoch_counter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict(),
                    'history': history,
                    'phase': phase_name
                }, checkpoint_path)
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        
        phase_time = time.time() - phase_start_time
        print(f"\n{phase_name.upper()} PHASE COMPLETED in {phase_time/60:.1f} minutes")
    
    print(f"\n6. CREATING TRAINING VISUALIZATIONS...")
    create_extended_training_plots(history)
    
    print(f"\n7. FINAL EVALUATION...")
    final_evaluation(model, mindbig_eeg, mindbig_targets, crell_eeg, crell_targets, 
                    mindbig_labels, crell_labels, stimulus_images, device)
    
    # Save final model
    final_model_path = 'hmadv2_final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history,
        'total_epochs': total_epochs
    }, final_model_path)
    
    print(f"\n{'='*70}")
    print(f"EXTENDED TRAINING COMPLETED!")
    print(f"{'='*70}")
    print(f"✓ Total epochs: {total_epochs}")
    print(f"✓ Final model saved: {final_model_path}")
    print(f"✓ Training history and plots created")
    print(f"✓ Model ready for deployment/evaluation")

def create_extended_training_plots(history):
    """Create comprehensive training plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('HMADv2 Extended Training Results', fontsize=16, fontweight='bold')
    
    epochs = history['epoch']
    
    # Loss plots
    axes[0, 0].plot(epochs, history['mindbig_loss'], 'b-', label='MindBigData', linewidth=2)
    axes[0, 0].plot(epochs, history['crell_loss'], 'r-', label='Crell', linewidth=2)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PSNR plots
    axes[0, 1].plot(epochs, history['mindbig_psnr'], 'b-', label='MindBigData', linewidth=2)
    axes[0, 1].plot(epochs, history['crell_psnr'], 'r-', label='Crell', linewidth=2)
    axes[0, 1].axhline(y=15, color='g', linestyle='--', alpha=0.7, label='Target (15dB)')
    axes[0, 1].set_title('PSNR (Peak Signal-to-Noise Ratio)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cosine similarity plots
    axes[0, 2].plot(epochs, history['mindbig_cosine'], 'b-', label='MindBigData', linewidth=2)
    axes[0, 2].plot(epochs, history['crell_cosine'], 'r-', label='Crell', linewidth=2)
    axes[0, 2].axhline(y=0.3, color='g', linestyle='--', alpha=0.7, label='Target (0.3)')
    axes[0, 2].set_title('Cosine Similarity')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Similarity')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Phase indicators
    phase_colors = {'reconstruction': 'lightblue', 'fine_tuning': 'lightcoral'}
    current_phase = None
    phase_start = 0
    
    for i, phase in enumerate(history['phase']):
        if phase != current_phase:
            if current_phase is not None:
                # Mark previous phase
                for ax in axes.flat:
                    ax.axvspan(phase_start, i, alpha=0.2, color=phase_colors.get(current_phase, 'gray'))
            current_phase = phase
            phase_start = i
    
    # Mark final phase
    if current_phase is not None:
        for ax in axes.flat:
            ax.axvspan(phase_start, len(epochs), alpha=0.2, color=phase_colors.get(current_phase, 'gray'))
    
    # Training summary
    final_mindbig_psnr = history['mindbig_psnr'][-1] if history['mindbig_psnr'] else 0
    final_crell_psnr = history['crell_psnr'][-1] if history['crell_psnr'] else 0
    final_mindbig_cosine = history['mindbig_cosine'][-1] if history['mindbig_cosine'] else 0
    final_crell_cosine = history['crell_cosine'][-1] if history['crell_cosine'] else 0
    
    summary_text = f"""
FINAL RESULTS:
MindBigData:
  PSNR: {final_mindbig_psnr:.2f} dB
  Cosine Sim: {final_mindbig_cosine:.4f}
  
Crell:
  PSNR: {final_crell_psnr:.2f} dB  
  Cosine Sim: {final_crell_cosine:.4f}

SUCCESS INDICATORS:
PSNR > 15 dB: {'✓' if max(final_mindbig_psnr, final_crell_psnr) > 15 else '✗'}
Cosine > 0.3: {'✓' if min(final_mindbig_cosine, final_crell_cosine) > 0.3 else '✗'}
"""
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
    axes[1, 1].axis('off')
    
    # Phase legend
    axes[1, 2].text(0.1, 0.9, "TRAINING PHASES:", transform=axes[1, 2].transAxes, 
                   fontsize=12, fontweight='bold')
    axes[1, 2].text(0.1, 0.7, "Reconstruction Phase", transform=axes[1, 2].transAxes, 
                   fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1, 2].text(0.1, 0.5, "Fine-tuning Phase", transform=axes[1, 2].transAxes, 
                   fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('hmadv2_extended_training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Extended training plots saved to: hmadv2_extended_training_results.png")

def final_evaluation(model, mindbig_eeg, mindbig_targets, crell_eeg, crell_targets,
                    mindbig_labels, crell_labels, stimulus_images, device):
    """Final comprehensive evaluation"""
    
    print("Performing final evaluation...")
    
    model.eval()
    with torch.no_grad():
        # Create final comparison visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('HMADv2 Final Results: Real Stimuli vs Reconstructions', fontsize=16, fontweight='bold')
        
        # MindBigData results
        if mindbig_eeg is not None:
            outputs = model(mindbig_eeg, 'mindbigdata', mindbig_targets)
            generated = outputs['generated_images']
            
            for i in range(min(4, generated.shape[0])):
                digit = mindbig_labels[i].item()
                
                # Real stimulus
                real_img = mindbig_targets[i].cpu().numpy().transpose(1, 2, 0)
                axes[0, i].imshow(real_img)
                axes[0, i].set_title(f'Real Digit {digit}', fontweight='bold')
                axes[0, i].axis('off')
                
                # Reconstruction
                gen_img = generated[i].cpu().numpy().transpose(1, 2, 0)
                gen_img = np.clip(gen_img, 0, 1)
                axes[1, i].imshow(gen_img)
                axes[1, i].set_title(f'HMADv2 Reconstruction')
                axes[1, i].axis('off')
        
        # Add row labels
        axes[0, 0].text(-0.15, 0.5, 'REAL\nSTIMULI', transform=axes[0, 0].transAxes, 
                       fontsize=12, fontweight='bold', ha='right', va='center', rotation=90, color='blue')
        
        axes[1, 0].text(-0.15, 0.5, 'HMADV2\nRECONSTRUCTION', transform=axes[1, 0].transAxes, 
                       fontsize=12, fontweight='bold', ha='right', va='center', rotation=90, color='green')
        
        plt.tight_layout()
        plt.savefig('hmadv2_final_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Final comparison saved to: hmadv2_final_comparison.png")

if __name__ == "__main__":
    extended_training_hmadv2()
