#!/usr/bin/env python3
"""
Test script untuk HMAD Framework
Menjalankan model dengan dataset yang tersedia
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io
from collections import defaultdict
from PIL import Image
import os

# Import HMAD framework
from hmad import HMADFramework, create_hmad_model, HMADTrainer

def load_mindbigdata_sample(filepath: str, max_samples: int = 10):
    """Load sample data dari MindBigData untuk testing"""
    print(f"Loading MindBigData from {filepath}...")
    
    signals_by_event = defaultdict(lambda: defaultdict(list))
    
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num > max_samples * 100:  # Limit untuk testing
                    break
                    
                parts = line.strip().split('\t')
                if len(parts) >= 7:
                    event_id = int(parts[1])
                    channel = parts[3]
                    digit_code = int(parts[4])
                    data_str = parts[6]
                    
                    if digit_code >= 0 and digit_code <= 9:  # Valid digits only
                        data_values = [float(x) for x in data_str.split(',')]
                        signals_by_event[event_id][channel].append({
                            'code': digit_code,
                            'data': np.array(data_values)
                        })
    except Exception as e:
        print(f"Error loading MindBigData: {e}")
        return None, None
    
    # Convert to tensor format
    eeg_data = []
    labels = []
    
    for event_id, channels_data in signals_by_event.items():
        if len(channels_data) >= 14:  # Ensure we have all channels
            # Get first sample from each channel
            channel_signals = []
            digit_code = None
            
            for channel in ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
                           'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']:
                if channel in channels_data and len(channels_data[channel]) > 0:
                    sample = channels_data[channel][0]
                    channel_signals.append(sample['data'])
                    if digit_code is None:
                        digit_code = sample['code']
            
            if len(channel_signals) == 14 and digit_code is not None:
                # Pad or truncate to fixed length (256 time points)
                fixed_length = 256
                processed_signals = []
                
                for signal in channel_signals:
                    if len(signal) >= fixed_length:
                        processed_signals.append(signal[:fixed_length])
                    else:
                        # Pad with zeros
                        padded = np.zeros(fixed_length)
                        padded[:len(signal)] = signal
                        processed_signals.append(padded)
                
                eeg_data.append(np.array(processed_signals))
                labels.append(digit_code)
                
                if len(eeg_data) >= max_samples:
                    break
    
    if len(eeg_data) > 0:
        eeg_tensor = torch.tensor(np.array(eeg_data), dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        print(f"Loaded {len(eeg_data)} MindBigData samples, shape: {eeg_tensor.shape}")
        return eeg_tensor, labels_tensor
    else:
        print("No valid MindBigData samples found")
        return None, None

def load_crell_sample(filepath: str, max_samples: int = 10):
    """Load sample data dari Crell untuk testing"""
    print(f"Loading Crell data from {filepath}...")
    
    try:
        data = scipy.io.loadmat(filepath)
        
        # Extract paradigm data (simplified)
        if 'paradigm_one' in data:
            paradigm_data = data['paradigm_one']
            
            # Extract EEG data (assuming structure exists)
            if len(paradigm_data) > 0 and len(paradigm_data[0]) > 0:
                round_data = paradigm_data[0, 0]
                
                if 'BrainVisionRDA_data' in round_data.dtype.names:
                    eeg_data = round_data['BrainVisionRDA_data'][0, 0]
                    
                    # Create sample data (simplified)
                    # In real implementation, would parse markers and extract epochs
                    num_channels, total_timepoints = eeg_data.shape
                    
                    # Create fixed-size samples
                    sample_length = 500  # 1 second at 500Hz
                    samples = []
                    labels = []
                    
                    for i in range(0, min(total_timepoints - sample_length, max_samples * sample_length), sample_length):
                        sample = eeg_data[:, i:i+sample_length]
                        samples.append(sample)
                        labels.append(0)  # Dummy label for now
                        
                        if len(samples) >= max_samples:
                            break
                    
                    if len(samples) > 0:
                        eeg_tensor = torch.tensor(np.array(samples), dtype=torch.float32)
                        labels_tensor = torch.tensor(labels, dtype=torch.long)
                        print(f"Loaded {len(samples)} Crell samples, shape: {eeg_tensor.shape}")
                        return eeg_tensor, labels_tensor
        
        print("Could not extract valid Crell data")
        return None, None
        
    except Exception as e:
        print(f"Error loading Crell data: {e}")
        return None, None

def load_stimulus_images(stimuli_dir: str, image_size: int = 64):
    """Load stimulus images untuk testing"""
    print(f"Loading stimulus images from {stimuli_dir}...")
    
    images = {}
    
    # MindBigData stimuli (digits)
    mindbig_dir = os.path.join(stimuli_dir, "MindbigdataStimuli")
    if os.path.exists(mindbig_dir):
        for i in range(10):
            img_path = os.path.join(mindbig_dir, f"{i}.jpg")
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((image_size, image_size))
                    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                    img_tensor = torch.tensor(img_array.transpose(2, 0, 1), dtype=torch.float32)
                    images[f'digit_{i}'] = img_tensor
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    # Crell stimuli (letters)
    crell_dir = os.path.join(stimuli_dir, "crellStimuli")
    if os.path.exists(crell_dir):
        letters = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
        for letter in letters:
            img_path = os.path.join(crell_dir, f"{letter}.png")
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((image_size, image_size))
                    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                    img_tensor = torch.tensor(img_array.transpose(2, 0, 1), dtype=torch.float32)
                    images[f'letter_{letter}'] = img_tensor
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    print(f"Loaded {len(images)} stimulus images")
    return images

def test_hmad_framework():
    """Test HMAD framework dengan data yang tersedia"""
    print("="*60)
    print("TESTING HMAD FRAMEWORK")
    print("="*60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Configuration
    config = {
        'mindbigdata_channels': 14,
        'crell_channels': 64,
        'd_model': 256,  # Reduced for testing
        'clip_dim': 256,
        'image_size': 64,
        'learning_rate': 1e-4,
        'batch_size': 4
    }
    
    # Create model
    print("\nCreating HMAD model...")
    model = create_hmad_model(config)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load datasets
    print("\nLoading datasets...")
    
    # MindBigData
    mindbig_eeg, mindbig_labels = load_mindbigdata_sample("datasets/EP1.01.txt", max_samples=5)
    
    # Crell
    crell_eeg, crell_labels = load_crell_sample("datasets/S01.mat", max_samples=5)
    
    # Stimulus images
    stimulus_images = load_stimulus_images("datasets", image_size=config['image_size'])
    
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
                
                if 'total_loss' in outputs:
                    print(f"  Total loss: {outputs['total_loss'].item():.4f}")
                    
            except Exception as e:
                print(f"✗ MindBigData forward pass failed: {e}")
        
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
                
                if 'total_loss' in outputs:
                    print(f"  Total loss: {outputs['total_loss'].item():.4f}")
                    
            except Exception as e:
                print(f"✗ Crell forward pass failed: {e}")
    
    print("\n" + "="*60)
    print("HMAD FRAMEWORK TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    test_hmad_framework()
