#!/usr/bin/env python3
"""
Main Execution Script for HMADv2
================================

Complete pipeline for EEG-to-Image reconstruction.
"""

import sys
import os
import argparse

# Add src to path
sys.path.append('src')

def main():
    parser = argparse.ArgumentParser(description='HMADv2 EEG-to-Image Reconstruction')
    parser.add_argument('--mode', choices=['train', 'eval', 'full'], default='full',
                       help='Execution mode')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("="*70)
    print("HMADv2: EEG-to-Image Reconstruction Pipeline")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Random Seed: {args.seed}")
    print("="*70)
    
    if args.mode in ['train', 'full']:
        print("\nSTARTING TRAINING PHASE...")
        from training.full_training_hmadv2 import full_training_hmadv2
        full_training_hmadv2()
    
    if args.mode in ['eval', 'full']:
        print("\nSTARTING EVALUATION PHASE...")
        from evaluation.create_comprehensive_results_summary import create_comprehensive_summary
        create_comprehensive_summary()
    
    print("\nPIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
