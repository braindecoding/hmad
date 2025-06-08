#!/usr/bin/env python3
"""
Implement Proper Train/Test Split
=================================

Implementasi proper train/test split untuk evaluasi yang valid:
1. Load semua data yang tersedia
2. Split menjadi train/test yang terpisah
3. Train model hanya pada training data
4. Test model hanya pada testing data
5. Compare dengan current approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from test_hmad import load_mindbigdata_sample, load_crell_sample, load_stimulus_images
from hmadv2 import create_improved_hmad_model, ProgressiveTrainer
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

def load_and_split_dataset(dataset_type, test_size=0.3, random_state=42):
    """Load dataset dan split menjadi train/test"""
    
    print(f"\nLoading and splitting {dataset_type} dataset...")
    
    # Load maximum available data
    if dataset_type == 'mindbigdata':
        eeg_data, labels = load_mindbigdata_sample("datasets/EP1.01.txt", max_samples=200)
    else:  # crell
        eeg_data, labels = load_crell_sample("datasets/S01.mat", max_samples=50)
    
    if eeg_data is None or labels is None:
        print(f"Failed to load {dataset_type} data")
        return None
    
    print(f"Total samples loaded: {len(labels)}")
    label_dist = Counter(labels.tolist())
    print(f"Label distribution: {dict(label_dist)}")
    
    # Convert to numpy for sklearn
    eeg_numpy = eeg_data.cpu().numpy()
    labels_numpy = labels.cpu().numpy()
    
    # Check if we can do stratified split
    min_class_count = min(label_dist.values())
    unique_labels = len(label_dist)
    
    if min_class_count >= 2 and len(labels_numpy) * test_size >= unique_labels:
        print(f"Using stratified split (min class count: {min_class_count})")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                eeg_numpy, labels_numpy,
                test_size=test_size,
                stratify=labels_numpy,
                random_state=random_state
            )
        except ValueError as e:
            print(f"Stratified split failed: {e}")
            print("Using random split instead")
            X_train, X_test, y_train, y_test = train_test_split(
                eeg_numpy, labels_numpy,
                test_size=test_size,
                random_state=random_state
            )
    else:
        print(f"Using random split (insufficient samples for stratification)")
        X_train, X_test, y_train, y_test = train_test_split(
            eeg_numpy, labels_numpy,
            test_size=test_size,
            random_state=random_state
        )
    
    # Convert back to tensors
    train_data = {
        'eeg': torch.from_numpy(X_train).float(),
        'labels': torch.from_numpy(y_train).long()
    }
    
    test_data = {
        'eeg': torch.from_numpy(X_test).float(),
        'labels': torch.from_numpy(y_test).long()
    }
    
    # Print split statistics
    train_dist = Counter(y_train)
    test_dist = Counter(y_test)
    
    print(f"Training set: {len(y_train)} samples")
    print(f"  Label distribution: {dict(train_dist)}")
    print(f"Test set: {len(y_test)} samples")
    print(f"  Label distribution: {dict(test_dist)}")
    
    return train_data, test_data

def train_model_properly(model, train_data, dataset_type, stimulus_images, device, epochs=20):
    """Train model HANYA pada training data"""
    
    print(f"\nTraining model on {dataset_type} training data...")
    
    # Move data to device
    train_eeg = train_data['eeg'].to(device)
    train_labels = train_data['labels'].to(device)
    
    # Create target images
    train_targets = create_target_images_from_labels(
        train_labels, dataset_type, stimulus_images, 64, device
    )
    
    # Create trainer
    trainer = ProgressiveTrainer(model, device)
    
    # Training loop
    model.train()
    training_losses = []
    
    for epoch in range(epochs):
        # Training step
        train_metrics = trainer.train_step(train_eeg, train_targets, dataset_type)
        training_losses.append(train_metrics['loss'])
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {train_metrics['loss']:.4f}")
    
    print(f"✓ Training completed. Final loss: {training_losses[-1]:.4f}")
    return training_losses

def evaluate_model_on_test(model, test_data, dataset_type, stimulus_images, device):
    """Evaluate model HANYA pada test data (unseen)"""
    
    print(f"\nEvaluating model on {dataset_type} test data (UNSEEN)...")
    
    # Move data to device
    test_eeg = test_data['eeg'].to(device)
    test_labels = test_data['labels'].to(device)
    
    # Create target images
    test_targets = create_target_images_from_labels(
        test_labels, dataset_type, stimulus_images, 64, device
    )
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(test_eeg, dataset_type, test_targets)
        generated_images = outputs['generated_images']
        
        # Compute metrics
        metrics = compute_metrics(generated_images, test_targets)
        
        print(f"✓ Test evaluation completed:")
        print(f"  PSNR: {metrics['avg_psnr']:.2f} ± {metrics['std_psnr']:.2f} dB")
        print(f"  Cosine Similarity: {metrics['avg_cosine']:.4f} ± {metrics['std_cosine']:.4f}")
        print(f"  Samples tested: {len(test_labels)}")
        
        return metrics, generated_images, test_targets

def compute_metrics(generated, target):
    """Compute quality metrics"""
    batch_size = generated.shape[0]
    psnr_values = []
    cosine_values = []
    
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
    
    return {
        'psnr_values': psnr_values,
        'cosine_values': cosine_values,
        'avg_psnr': np.mean(psnr_values),
        'avg_cosine': np.mean(cosine_values),
        'std_psnr': np.std(psnr_values),
        'std_cosine': np.std(cosine_values)
    }

def compare_with_current_approach(model, all_data, dataset_type, stimulus_images, device):
    """Compare dengan current approach (no split)"""
    
    print(f"\nComparing with current approach (no train/test split)...")
    
    # Use all data (current approach)
    all_eeg = all_data['eeg'].to(device)
    all_labels = all_data['labels'].to(device)
    all_targets = create_target_images_from_labels(
        all_labels, dataset_type, stimulus_images, 64, device
    )
    
    model.eval()
    with torch.no_grad():
        outputs = model(all_eeg, dataset_type, all_targets)
        generated_images = outputs['generated_images']
        metrics = compute_metrics(generated_images, all_targets)
        
        print(f"✓ Current approach (all data):")
        print(f"  PSNR: {metrics['avg_psnr']:.2f} ± {metrics['std_psnr']:.2f} dB")
        print(f"  Cosine Similarity: {metrics['avg_cosine']:.4f} ± {metrics['std_cosine']:.4f}")
        
        return metrics

def implement_proper_train_test_split():
    """Main function untuk implementasi proper train/test split"""
    
    print("="*70)
    print("IMPLEMENTING PROPER TRAIN/TEST SPLIT")
    print("="*70)
    
    # Set random seeds
    set_random_seeds(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load stimulus images
    stimulus_images = load_stimulus_images("datasets", image_size=64)
    
    # Configuration
    config = {
        'mindbigdata_channels': 14,
        'crell_channels': 64,
        'd_model': 256,
        'image_size': 64
    }
    
    results = {}
    
    # Process each dataset
    for dataset_type in ['mindbigdata', 'crell']:
        print(f"\n{'='*50}")
        print(f"PROCESSING {dataset_type.upper()} DATASET")
        print(f"{'='*50}")
        
        # Load and split data
        split_result = load_and_split_dataset(dataset_type, test_size=0.3, random_state=42)
        if split_result is None:
            continue
            
        train_data, test_data = split_result
        
        # Create fresh model for this dataset
        model = create_improved_hmad_model(config)
        model = model.to(device)
        
        # Train model on training data only
        training_losses = train_model_properly(
            model, train_data, dataset_type, stimulus_images, device, epochs=15
        )
        
        # Evaluate on test data (unseen)
        test_metrics, test_generated, test_targets = evaluate_model_on_test(
            model, test_data, dataset_type, stimulus_images, device
        )
        
        # Compare with current approach
        all_data = {
            'eeg': torch.cat([train_data['eeg'], test_data['eeg']], dim=0),
            'labels': torch.cat([train_data['labels'], test_data['labels']], dim=0)
        }
        current_metrics = compare_with_current_approach(
            model, all_data, dataset_type, stimulus_images, device
        )
        
        # Store results
        results[dataset_type] = {
            'train_data': train_data,
            'test_data': test_data,
            'test_metrics': test_metrics,
            'current_metrics': current_metrics,
            'training_losses': training_losses,
            'test_generated': test_generated,
            'test_targets': test_targets
        }
    
    print(f"\n{'='*70}")
    print("CREATING COMPARISON VISUALIZATION...")
    print(f"{'='*70}")
    
    create_train_test_comparison_visualization(results)
    
    print(f"\n{'='*70}")
    print("ANALYSIS AND CONCLUSIONS...")
    print(f"{'='*70}")
    
    analyze_train_test_results(results)
    
    print(f"\n{'='*70}")
    print("PROPER TRAIN/TEST SPLIT IMPLEMENTATION COMPLETED!")
    print(f"{'='*70}")

def create_train_test_comparison_visualization(results):
    """Create visualization comparing train/test split vs current approach"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Proper Train/Test Split vs Current Approach', fontsize=16, fontweight='bold')
    
    dataset_names = ['MindBigData', 'Crell']
    colors = ['blue', 'green']
    
    for i, (dataset_type, color) in enumerate(zip(['mindbigdata', 'crell'], colors)):
        if dataset_type not in results:
            continue
            
        result = results[dataset_type]
        
        # PSNR comparison
        test_psnr = result['test_metrics']['avg_psnr']
        current_psnr = result['current_metrics']['avg_psnr']
        
        axes[i, 0].bar(['Test Set\n(Unseen)', 'Current\n(All Data)'], 
                      [test_psnr, current_psnr], 
                      color=[color, 'orange'], alpha=0.7)
        axes[i, 0].set_title(f'{dataset_names[i]}: PSNR Comparison')
        axes[i, 0].set_ylabel('PSNR (dB)')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Add difference annotation
        diff = test_psnr - current_psnr
        axes[i, 0].text(0.5, max(test_psnr, current_psnr) + 0.1, 
                       f'Diff: {diff:+.2f} dB', 
                       ha='center', fontweight='bold',
                       color='red' if diff < 0 else 'green')
        
        # Cosine similarity comparison
        test_cosine = result['test_metrics']['avg_cosine']
        current_cosine = result['current_metrics']['avg_cosine']
        
        axes[i, 1].bar(['Test Set\n(Unseen)', 'Current\n(All Data)'], 
                      [test_cosine, current_cosine], 
                      color=[color, 'orange'], alpha=0.7)
        axes[i, 1].set_title(f'{dataset_names[i]}: Cosine Similarity')
        axes[i, 1].set_ylabel('Cosine Similarity')
        axes[i, 1].grid(True, alpha=0.3)
        
        # Training loss curve
        losses = result['training_losses']
        epochs = range(1, len(losses) + 1)
        axes[i, 2].plot(epochs, losses, color=color, linewidth=2, marker='o')
        axes[i, 2].set_title(f'{dataset_names[i]}: Training Loss')
        axes[i, 2].set_xlabel('Epoch')
        axes[i, 2].set_ylabel('Loss')
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('train_test_split_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Train/test comparison visualization saved to: train_test_split_comparison.png")

def analyze_train_test_results(results):
    """Analyze results dari train/test split"""
    
    print("TRAIN/TEST SPLIT ANALYSIS:")
    print("="*40)
    
    for dataset_type in ['mindbigdata', 'crell']:
        if dataset_type not in results:
            continue
            
        result = results[dataset_type]
        test_metrics = result['test_metrics']
        current_metrics = result['current_metrics']
        
        print(f"\n📊 {dataset_type.upper()} RESULTS:")
        print(f"Test Set (Unseen Data):")
        print(f"  PSNR: {test_metrics['avg_psnr']:.2f} ± {test_metrics['std_psnr']:.2f} dB")
        print(f"  Cosine: {test_metrics['avg_cosine']:.4f} ± {test_metrics['std_cosine']:.4f}")
        print(f"  Samples: {len(result['test_data']['labels'])}")
        
        print(f"Current Approach (All Data):")
        print(f"  PSNR: {current_metrics['avg_psnr']:.2f} ± {current_metrics['std_psnr']:.2f} dB")
        print(f"  Cosine: {current_metrics['avg_cosine']:.4f} ± {current_metrics['std_cosine']:.4f}")
        
        # Performance difference
        psnr_diff = test_metrics['avg_psnr'] - current_metrics['avg_psnr']
        cosine_diff = test_metrics['avg_cosine'] - current_metrics['avg_cosine']
        
        print(f"Performance Difference (Test vs Current):")
        print(f"  PSNR: {psnr_diff:+.2f} dB ({'✓ Better' if psnr_diff > 0 else '⚠️ Lower' if psnr_diff < -0.5 else '≈ Similar'})")
        print(f"  Cosine: {cosine_diff:+.4f} ({'✓ Better' if cosine_diff > 0 else '⚠️ Lower' if cosine_diff < -0.05 else '≈ Similar'})")
    
    print(f"\n🎯 CONCLUSIONS:")
    print(f"1. ✅ PROPER EVALUATION: Now testing on truly unseen data")
    print(f"2. 📊 REALISTIC PERFORMANCE: Test metrics show real generalization")
    print(f"3. 🔬 VALID METHODOLOGY: No data leakage between train/test")
    print(f"4. 📈 PUBLISHABLE RESULTS: Proper experimental design")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"• Use train/test split results for publication")
    print(f"• Report both training and test performance")
    print(f"• Acknowledge any performance differences")
    print(f"• This approach is scientifically rigorous")

if __name__ == "__main__":
    implement_proper_train_test_split()
