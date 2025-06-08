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
from models.hmad import HMADFramework
from models.hmadv2 import create_improved_hmad_model

def load_mindbigdata_sample(filepath: str, max_samples: int = 10):
    """Load sample data dari MindBigData EPOC untuk testing"""
    print(f"Loading MindBigData EPOC from {filepath}...")

    # EPOC channels dalam urutan yang benar
    epoc_channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    signals_by_event = defaultdict(lambda: defaultdict(list))

    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num > max_samples * 200:  # Limit untuk testing (lebih besar karena butuh semua channels)
                    break

                parts = line.strip().split('\t')
                if len(parts) >= 7:
                    device = parts[2]

                    # Hanya ambil data dari EPOC device
                    if device == "EP":
                        event_id = int(parts[1])
                        channel = parts[3]
                        digit_code = int(parts[4])
                        data_str = parts[6]

                        # Hanya ambil digit yang valid (0-9)
                        if digit_code >= 0 and digit_code <= 9 and channel in epoc_channels:
                            try:
                                data_values = [float(x) for x in data_str.split(',')]
                                signals_by_event[event_id][channel] = {
                                    'code': digit_code,
                                    'data': np.array(data_values)
                                }
                            except ValueError:
                                continue  # Skip malformed data

    except Exception as e:
        print(f"Error loading MindBigData: {e}")
        return None, None
    
    # Convert to tensor format
    eeg_data = []
    labels = []

    print(f"Found {len(signals_by_event)} events")

    for event_id, channels_data in signals_by_event.items():
        # Pastikan kita punya semua 14 channels EPOC
        if len(channels_data) == 14:
            channel_signals = []
            digit_code = None

            # Ambil data dalam urutan channel yang benar
            for channel in epoc_channels:
                if channel in channels_data:
                    sample = channels_data[channel]
                    channel_signals.append(sample['data'])
                    if digit_code is None:
                        digit_code = sample['code']
                else:
                    break  # Skip event jika ada channel yang missing

            if len(channel_signals) == 14 and digit_code is not None:
                # Pad atau truncate ke fixed length (256 time points untuk ~2 detik pada 128Hz)
                fixed_length = 256
                processed_signals = []

                for signal in channel_signals:
                    if len(signal) >= fixed_length:
                        processed_signals.append(signal[:fixed_length])
                    else:
                        # Pad dengan mean value untuk menghindari artifacts
                        padded = np.full(fixed_length, np.mean(signal))
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
    """Load sample data dari Crell untuk testing dengan proper marker parsing"""
    print(f"Loading Crell data from {filepath}...")

    # Letter mapping: ascii index to letter
    letter_mapping = {100: 'a', 103: 'd', 104: 'e', 105: 'f', 109: 'j',
                     113: 'n', 114: 'o', 118: 's', 119: 't', 121: 'v'}

    try:
        data = scipy.io.loadmat(filepath)

        eeg_samples = []
        labels = []

        # Process both paradigm rounds
        for paradigm_key in ['round01_paradigm', 'round02_paradigm']:
            if paradigm_key not in data:
                continue

            paradigm_data = data[paradigm_key]
            if len(paradigm_data) == 0:
                continue

            round_data = paradigm_data[0, 0]

            # Extract data arrays
            if 'BrainVisionRDA_data' not in round_data.dtype.names:
                continue

            # Correct way to access the data based on debug output
            eeg_data = round_data['BrainVisionRDA_data'].T  # (timepoints, 64) -> (64, timepoints)
            eeg_times = round_data['BrainVisionRDA_time'].flatten()
            marker_data = round_data['ParadigmMarker_data'].flatten()
            marker_times = round_data['ParadigmMarker_time'].flatten()

            print(f"  {paradigm_key}: EEG shape {eeg_data.shape}, {len(marker_data)} markers")

            # Parse markers untuk extract visual epochs
            visual_epochs = extract_visual_epochs_crell(
                eeg_data, eeg_times, marker_data, marker_times, letter_mapping
            )

            for epoch_data, letter_code in visual_epochs:
                eeg_samples.append(epoch_data)
                labels.append(letter_code)

                if len(eeg_samples) >= max_samples:
                    break

            if len(eeg_samples) >= max_samples:
                break

        if len(eeg_samples) > 0:
            eeg_tensor = torch.tensor(np.array(eeg_samples), dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            print(f"Loaded {len(eeg_samples)} Crell visual epochs, shape: {eeg_tensor.shape}")
            return eeg_tensor, labels_tensor
        else:
            print("No valid Crell visual epochs found")
            return None, None

    except Exception as e:
        print(f"Error loading Crell data: {e}")
        return None, None

def extract_visual_epochs_crell(eeg_data, eeg_times, marker_data, marker_times, letter_mapping):
    """Extract visual epochs dari Crell data berdasarkan markers"""
    epochs = []

    # Find letter presentation events
    letter_events = []
    current_letter = None
    fade_in_time = None
    fade_out_time = None

    for i, (marker, marker_time) in enumerate(zip(marker_data, marker_times)):
        if marker >= 100:  # Letter code
            current_letter = letter_mapping.get(marker, None)
        elif marker == 1:  # Fade in start
            fade_in_time = marker_time
        elif marker == 3:  # Fade out start (end of pure visual phase)
            fade_out_time = marker_time

            # Extract visual epoch (from fade in to fade out)
            if current_letter is not None and fade_in_time is not None:
                letter_events.append({
                    'letter': current_letter,
                    'start_time': fade_in_time,
                    'end_time': fade_out_time
                })

            # Reset for next letter
            current_letter = None
            fade_in_time = None
            fade_out_time = None

    print(f"    Found {len(letter_events)} letter events")

    # Extract EEG epochs
    for event in letter_events[:10]:  # Limit untuk testing
        start_time = event['start_time']
        end_time = event['end_time']

        # Find corresponding EEG indices
        start_idx = np.searchsorted(eeg_times, start_time)
        end_idx = np.searchsorted(eeg_times, end_time)

        if end_idx > start_idx and (end_idx - start_idx) > 100:  # Minimum epoch length
            # Extract epoch dan resample ke fixed length
            epoch_data = eeg_data[:, start_idx:end_idx]

            # Resample to fixed length (750 samples = 1.5s at 500Hz)
            target_length = 750
            if epoch_data.shape[1] != target_length:
                # Simple resampling by interpolation
                from scipy.interpolate import interp1d
                old_indices = np.linspace(0, epoch_data.shape[1]-1, epoch_data.shape[1])
                new_indices = np.linspace(0, epoch_data.shape[1]-1, target_length)

                resampled_epoch = np.zeros((64, target_length))
                for ch in range(64):
                    f = interp1d(old_indices, epoch_data[ch, :], kind='linear')
                    resampled_epoch[ch, :] = f(new_indices)

                epoch_data = resampled_epoch

            # Convert letter to numeric label
            letter_to_num = {'a': 0, 'd': 1, 'e': 2, 'f': 3, 'j': 4,
                           'n': 5, 'o': 6, 's': 7, 't': 8, 'v': 9}
            letter_label = letter_to_num.get(event['letter'], 0)

            epochs.append((epoch_data, letter_label))

    return epochs

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
    model = create_improved_hmad_model(config)
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
