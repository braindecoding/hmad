#!/usr/bin/env python3
"""
Analyze Data Preparation and Testing Strategy
==============================================

Analisis mendalam tentang:
1. Data preparation untuk training vs testing
2. Label distribution untuk digits (0-9) dan letters
3. Data splitting strategy
4. Cross-validation approach
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from test_hmad import load_mindbigdata_sample, load_crell_sample, load_stimulus_images
import os

def analyze_data_preparation():
    """Comprehensive analysis of data preparation"""
    
    print("="*70)
    print("DATA PREPARATION AND TESTING STRATEGY ANALYSIS")
    print("="*70)
    
    print("\n1. ANALYZING MINDBIGDATA DATASET...")
    analyze_mindbigdata_distribution()
    
    print("\n2. ANALYZING CRELL DATASET...")
    analyze_crell_distribution()
    
    print("\n3. ANALYZING STIMULUS IMAGES...")
    analyze_stimulus_images()
    
    print("\n4. CURRENT DATA USAGE ANALYSIS...")
    analyze_current_data_usage()
    
    print("\n5. RECOMMENDATIONS FOR PROPER DATA SPLITTING...")
    recommend_data_splitting_strategy()
    
    print("\n6. CREATING COMPREHENSIVE DATA ANALYSIS VISUALIZATION...")
    create_data_analysis_visualization()
    
    print(f"\n{'='*70}")
    print("DATA PREPARATION ANALYSIS COMPLETED!")
    print(f"{'='*70}")

def analyze_mindbigdata_distribution():
    """Analyze MindBigData label distribution"""
    
    print("MINDBIGDATA DATASET ANALYSIS:")
    print("="*40)
    
    # Load different sample sizes to see distribution
    sample_sizes = [10, 20, 50, 100]
    
    for size in sample_sizes:
        try:
            eeg_data, labels = load_mindbigdata_sample("datasets/EP1.01.txt", max_samples=size)
            if eeg_data is not None and labels is not None:
                label_counts = Counter(labels.tolist())
                
                print(f"\nSample size {size}:")
                print(f"  Total samples: {len(labels)}")
                print(f"  Unique digits: {sorted(label_counts.keys())}")
                print(f"  Label distribution:")
                
                for digit in range(10):
                    count = label_counts.get(digit, 0)
                    percentage = (count / len(labels)) * 100 if len(labels) > 0 else 0
                    print(f"    Digit {digit}: {count} samples ({percentage:.1f}%)")
                
                # Check if we have all digits 0-9
                missing_digits = set(range(10)) - set(label_counts.keys())
                if missing_digits:
                    print(f"  ⚠️  Missing digits: {sorted(missing_digits)}")
                else:
                    print(f"  ✅ All digits 0-9 present")
                    
        except Exception as e:
            print(f"  Error loading {size} samples: {e}")

def analyze_crell_distribution():
    """Analyze Crell dataset label distribution"""
    
    print("CRELL DATASET ANALYSIS:")
    print("="*30)
    
    letter_names = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
    
    # Load different sample sizes
    sample_sizes = [10, 20, 50, 100]
    
    for size in sample_sizes:
        try:
            eeg_data, labels = load_crell_sample("datasets/S01.mat", max_samples=size)
            if eeg_data is not None and labels is not None:
                label_counts = Counter(labels.tolist())
                
                print(f"\nSample size {size}:")
                print(f"  Total samples: {len(labels)}")
                print(f"  Unique label indices: {sorted(label_counts.keys())}")
                print(f"  Label distribution:")
                
                for i, letter in enumerate(letter_names):
                    count = label_counts.get(i, 0)
                    percentage = (count / len(labels)) * 100 if len(labels) > 0 else 0
                    print(f"    Letter {letter} (idx {i}): {count} samples ({percentage:.1f}%)")
                
                # Check coverage
                present_letters = [letter_names[i] for i in label_counts.keys() if i < len(letter_names)]
                missing_letters = [letter_names[i] for i in range(len(letter_names)) if i not in label_counts.keys()]
                
                print(f"  ✅ Present letters: {present_letters}")
                if missing_letters:
                    print(f"  ⚠️  Missing letters: {missing_letters}")
                    
        except Exception as e:
            print(f"  Error loading {size} samples: {e}")

def analyze_stimulus_images():
    """Analyze available stimulus images"""
    
    print("STIMULUS IMAGES ANALYSIS:")
    print("="*35)
    
    try:
        stimulus_images = load_stimulus_images("datasets", image_size=64)
        
        print(f"Total stimulus images loaded: {len(stimulus_images)}")
        print(f"Stimulus keys: {list(stimulus_images.keys())}")
        
        # Analyze digit stimuli
        digit_stimuli = [key for key in stimulus_images.keys() if key.startswith('digit_')]
        letter_stimuli = [key for key in stimulus_images.keys() if key.startswith('letter_')]
        
        print(f"\nDigit stimuli ({len(digit_stimuli)}):")
        digit_numbers = []
        for key in digit_stimuli:
            try:
                digit = int(key.split('_')[1])
                digit_numbers.append(digit)
                print(f"  {key} -> Digit {digit}")
            except:
                print(f"  {key} -> Invalid format")
        
        missing_digits = set(range(10)) - set(digit_numbers)
        if missing_digits:
            print(f"  ⚠️  Missing digit stimuli: {sorted(missing_digits)}")
        else:
            print(f"  ✅ All digit stimuli (0-9) available")
        
        print(f"\nLetter stimuli ({len(letter_stimuli)}):")
        letter_chars = []
        for key in letter_stimuli:
            try:
                letter = key.split('_')[1]
                letter_chars.append(letter)
                print(f"  {key} -> Letter {letter}")
            except:
                print(f"  {key} -> Invalid format")
        
        expected_letters = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
        missing_letters = set(expected_letters) - set(letter_chars)
        if missing_letters:
            print(f"  ⚠️  Missing letter stimuli: {sorted(missing_letters)}")
        else:
            print(f"  ✅ All expected letter stimuli available")
            
    except Exception as e:
        print(f"Error loading stimulus images: {e}")

def analyze_current_data_usage():
    """Analyze how we're currently using data"""
    
    print("CURRENT DATA USAGE ANALYSIS:")
    print("="*40)
    
    print("\n🔍 CURRENT APPROACH:")
    print("1. Loading fixed number of samples (usually 8-20)")
    print("2. Using same samples for both training and testing")
    print("3. No explicit train/validation/test split")
    print("4. No cross-validation")
    print("5. No stratified sampling to ensure label balance")
    
    print("\n⚠️  POTENTIAL ISSUES:")
    print("1. OVERFITTING: Testing on training data")
    print("2. BIAS: Unbalanced label distribution")
    print("3. GENERALIZATION: No unseen data evaluation")
    print("4. REPRODUCIBILITY: Random sampling without seeds")
    print("5. STATISTICAL VALIDITY: Small sample sizes")
    
    print("\n📊 SAMPLE SIZE ANALYSIS:")
    
    # Test different sample sizes
    for dataset_name, load_func, dataset_file in [
        ("MindBigData", load_mindbigdata_sample, "datasets/EP1.01.txt"),
        ("Crell", load_crell_sample, "datasets/S01.mat")
    ]:
        print(f"\n{dataset_name}:")
        
        for size in [10, 20, 50, 100, 200]:
            try:
                eeg_data, labels = load_func(dataset_file, max_samples=size)
                if eeg_data is not None and labels is not None:
                    unique_labels = len(set(labels.tolist()))
                    print(f"  Size {size:3d}: {len(labels):3d} samples, {unique_labels:2d} unique labels")
                else:
                    print(f"  Size {size:3d}: Failed to load")
            except Exception as e:
                print(f"  Size {size:3d}: Error - {str(e)[:50]}...")

def recommend_data_splitting_strategy():
    """Recommend proper data splitting strategy"""
    
    print("RECOMMENDED DATA SPLITTING STRATEGY:")
    print("="*50)
    
    print("\n🎯 PROPER EXPERIMENTAL DESIGN:")
    
    print("\n1. STRATIFIED SAMPLING:")
    print("   • Ensure balanced representation of all labels")
    print("   • MindBigData: Equal samples for digits 0-9")
    print("   • Crell: Equal samples for letters a,d,e,f,j,n,o,s,t,v")
    
    print("\n2. TRAIN/VALIDATION/TEST SPLIT:")
    print("   • Training: 60% of data")
    print("   • Validation: 20% of data (for hyperparameter tuning)")
    print("   • Testing: 20% of data (for final evaluation)")
    print("   • NO OVERLAP between splits")
    
    print("\n3. CROSS-VALIDATION:")
    print("   • 5-fold or 10-fold cross-validation")
    print("   • Stratified to maintain label balance")
    print("   • Report mean ± std across folds")
    
    print("\n4. SAMPLE SIZE RECOMMENDATIONS:")
    print("   • Minimum: 50 samples per label")
    print("   • Recommended: 100+ samples per label")
    print("   • MindBigData: 500-1000 total samples (50-100 per digit)")
    print("   • Crell: 500-1000 total samples (50-100 per letter)")
    
    print("\n5. REPRODUCIBILITY:")
    print("   • Fixed random seeds")
    print("   • Documented data splits")
    print("   • Version-controlled datasets")
    
    print("\n📋 IMPLEMENTATION CHECKLIST:")
    print("   ✅ Load maximum available data")
    print("   ✅ Check label distribution")
    print("   ✅ Stratified train/val/test split")
    print("   ✅ Separate evaluation on test set")
    print("   ✅ Cross-validation for robust results")
    print("   ✅ Report confidence intervals")

def create_data_analysis_visualization():
    """Create comprehensive data analysis visualization"""
    
    print("Creating data analysis visualization...")
    
    # Load data for visualization
    try:
        # MindBigData
        mindbig_eeg, mindbig_labels = load_mindbigdata_sample("datasets/EP1.01.txt", max_samples=100)
        mindbig_distribution = Counter(mindbig_labels.tolist()) if mindbig_labels is not None else {}
        
        # Crell
        crell_eeg, crell_labels = load_crell_sample("datasets/S01.mat", max_samples=100)
        crell_distribution = Counter(crell_labels.tolist()) if crell_labels is not None else {}
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Data Preparation Analysis: Current State vs Recommended', fontsize=16, fontweight='bold')
        
        # MindBigData distribution
        if mindbig_distribution:
            digits = list(range(10))
            counts = [mindbig_distribution.get(d, 0) for d in digits]
            
            bars = axes[0, 0].bar(digits, counts, alpha=0.7, color='blue')
            axes[0, 0].set_title('MindBigData: Current Label Distribution')
            axes[0, 0].set_xlabel('Digit')
            axes[0, 0].set_ylabel('Sample Count')
            axes[0, 0].set_xticks(digits)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Highlight missing digits
            for i, count in enumerate(counts):
                if count == 0:
                    bars[i].set_color('red')
                    bars[i].set_alpha(0.5)
        
        # Crell distribution
        if crell_distribution:
            letter_names = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
            indices = list(range(len(letter_names)))
            counts = [crell_distribution.get(i, 0) for i in indices]
            
            bars = axes[0, 1].bar(indices, counts, alpha=0.7, color='green')
            axes[0, 1].set_title('Crell: Current Label Distribution')
            axes[0, 1].set_xlabel('Letter Index')
            axes[0, 1].set_ylabel('Sample Count')
            axes[0, 1].set_xticks(indices)
            axes[0, 1].set_xticklabels(letter_names)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Highlight missing letters
            for i, count in enumerate(counts):
                if count == 0:
                    bars[i].set_color('red')
                    bars[i].set_alpha(0.5)
        
        # Current vs Recommended approach
        current_approach = """
CURRENT APPROACH:
❌ Same data for train/test
❌ Small sample sizes (8-20)
❌ No stratification
❌ No cross-validation
❌ Unbalanced labels
❌ No statistical validation

ISSUES:
• Overfitting
• Poor generalization
• Unreliable results
• No confidence intervals
"""
        
        recommended_approach = """
RECOMMENDED APPROACH:
✅ Separate train/val/test
✅ Large sample sizes (500+)
✅ Stratified sampling
✅ Cross-validation
✅ Balanced labels
✅ Statistical validation

BENEFITS:
• Better generalization
• Reliable results
• Confidence intervals
• Reproducible research
"""
        
        axes[0, 2].text(0.05, 0.95, current_approach, transform=axes[0, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
        axes[0, 2].axis('off')
        
        axes[1, 0].text(0.05, 0.95, recommended_approach, transform=axes[1, 0].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        axes[1, 0].axis('off')
        
        # Recommended data split visualization
        split_labels = ['Training\n(60%)', 'Validation\n(20%)', 'Testing\n(20%)']
        split_sizes = [60, 20, 20]
        colors = ['lightblue', 'lightyellow', 'lightcoral']
        
        wedges, texts, autotexts = axes[1, 1].pie(split_sizes, labels=split_labels, colors=colors, 
                                                 autopct='%1.0f%%', startangle=90)
        axes[1, 1].set_title('Recommended Data Split')
        
        # Implementation roadmap
        roadmap_text = """
IMPLEMENTATION ROADMAP:

PHASE 1: Data Audit
• Load maximum available data
• Analyze label distributions
• Identify missing labels
• Document data quality

PHASE 2: Proper Splitting
• Implement stratified sampling
• Create train/val/test splits
• Ensure no data leakage
• Save split indices

PHASE 3: Evaluation
• Train on training set only
• Tune on validation set
• Final evaluation on test set
• Cross-validation analysis

PHASE 4: Reporting
• Statistical significance tests
• Confidence intervals
• Reproducibility documentation
"""
        
        axes[1, 2].text(0.05, 0.95, roadmap_text, transform=axes[1, 2].transAxes, 
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('data_preparation_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Data analysis visualization saved to: data_preparation_analysis.png")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")

if __name__ == "__main__":
    analyze_data_preparation()
