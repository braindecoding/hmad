#!/usr/bin/env python3
"""
Test script untuk HMAD Original dengan rekonstruksi visual
Menggunakan HMAD Original yang sophisticated untuk hasil yang lebih baik
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import os

# Import HMAD framework
from hmad_original import HMADFramework, create_hmad_model

def load_original_stimulus(stimulus_type, label, size=(64, 64)):
    """Load original stimulus from dataset"""
    try:
        if stimulus_type == 'digit':
            # Load MindBigData digit stimulus
            img_path = f"../datasets/MindbigdataStimuli/{label}.jpg"
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                img = img.resize(size)
                return np.array(img) / 255.0
            else:
                print(f"Warning: Digit stimulus {label} not found, creating fallback")
                return create_fallback_digit_stimulus(label, size)

        elif stimulus_type == 'letter':
            # Load Crell letter stimulus
            letter_names = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
            if label < len(letter_names):
                letter = letter_names[label]
                img_path = f"../datasets/crellStimuli/{letter}.png"
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(size)
                    return np.array(img) / 255.0
                else:
                    print(f"Warning: Letter stimulus {letter} not found, creating fallback")
                    return create_fallback_letter_stimulus(letter, size)
            else:
                print(f"Warning: Invalid letter index {label}, creating fallback")
                return create_fallback_letter_stimulus('?', size)

    except Exception as e:
        print(f"Error loading stimulus: {e}")
        if stimulus_type == 'digit':
            return create_fallback_digit_stimulus(label, size)
        else:
            return create_fallback_letter_stimulus('?', size)

def create_fallback_digit_stimulus(digit, size=(64, 64)):
    """Create a simple visual representation of a digit as fallback"""
    img = Image.new('L', size, color=0)  # Black background
    draw = ImageDraw.Draw(img)

    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", size=40)
    except:
        font = ImageFont.load_default()

    # Get text size and center it
    bbox = draw.textbbox((0, 0), str(digit), font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2

    # Draw white digit on black background
    draw.text((x, y), str(digit), fill=255, font=font)

    return np.array(img) / 255.0

def create_fallback_letter_stimulus(letter, size=(64, 64)):
    """Create a simple visual representation of a letter as fallback"""
    img = Image.new('L', size, color=0)  # Black background
    draw = ImageDraw.Draw(img)

    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", size=40)
    except:
        font = ImageFont.load_default()

    # Get text size and center it
    bbox = draw.textbbox((0, 0), letter.upper(), font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2

    # Draw white letter on black background
    draw.text((x, y), letter.upper(), fill=255, font=font)

    return np.array(img) / 255.0

def load_mindbigdata_sample(filepath: str, max_samples: int = 3):
    """Load sample data dari MindBigData EPOC untuk testing"""
    print(f"Loading MindBigData EPOC from {filepath}...")

    # EPOC channels dalam urutan yang benar
    epoc_channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    signals_by_event = defaultdict(lambda: defaultdict(list))

    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num > max_samples * 200:  # Limit untuk testing
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

def load_crell_sample(filepath: str, max_samples: int = 3):
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

def create_simple_hmad_model(config):
    """Create simplified HMAD model untuk testing"""

    class SimpleHMAD(nn.Module):
        def __init__(self, mindbig_channels=14, crell_channels=64, d_model=256, image_size=64):
            super().__init__()

            # Dataset-specific preprocessing
            self.mindbig_preprocessor = nn.Conv1d(mindbig_channels, d_model, kernel_size=1)
            self.crell_preprocessor = nn.Conv1d(crell_channels, d_model, kernel_size=1)

            # Simple feature extraction
            self.feature_extractor = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=8, padding=4),
                nn.ReLU(),
                nn.Conv1d(d_model, d_model, kernel_size=16, padding=8),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(64)  # Fixed output length
            )

            # Simple decoder
            self.decoder = nn.Sequential(
                nn.Linear(d_model * 64, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 3 * image_size * image_size),
                nn.Sigmoid()
            )

            self.image_size = image_size

        def forward(self, x, dataset_type, target_images=None):
            # Dataset-specific preprocessing
            if dataset_type == 'mindbigdata':
                x = self.mindbig_preprocessor(x)
            else:  # crell
                x = self.crell_preprocessor(x)

            # Extract features
            features = self.feature_extractor(x)

            # Flatten
            features = features.view(features.shape[0], -1)

            # Decode to image
            output = self.decoder(features)
            output = output.view(-1, 3, self.image_size, self.image_size)

            return {
                'generated_images': output,
                'clip_latent': features[:, :512] if features.shape[1] >= 512 else features
            }

    return SimpleHMAD(
        mindbig_channels=config.get('mindbigdata_channels', 14),
        crell_channels=config.get('crell_channels', 64),
        d_model=config.get('d_model', 256),
        image_size=config.get('image_size', 64)
    )

def test_hmad_reconstruction():
    """Test HMAD reconstruction dengan visualisasi"""
    print("="*60)
    print("TESTING HMAD ORIGINAL RECONSTRUCTION")
    print("="*60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Configuration
    config = {
        'mindbigdata_channels': 14,
        'crell_channels': 64,
        'd_model': 256,
        'image_size': 64,
    }

    # Create simplified model untuk testing
    print("\nCreating simplified HMAD model...")
    model = create_simple_hmad_model(config)
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load MindBigData
    print("\nLoading MindBigData...")
    mindbig_eeg, mindbig_labels = load_mindbigdata_sample("../datasets/EP1.01.txt", max_samples=3)

    # Load Crell
    print("\nLoading Crell...")
    crell_eeg, crell_labels = load_crell_sample("../datasets/S01.mat", max_samples=3)

    model.eval()
    with torch.no_grad():
        # Test MindBigData
        if mindbig_eeg is not None:
            print(f"\nTesting MindBigData reconstruction (shape: {mindbig_eeg.shape})...")
            mindbig_eeg = mindbig_eeg.to(device)

            try:
                outputs = model(mindbig_eeg, 'mindbigdata')
                generated_images = outputs['generated_images']

                print(f"✓ MindBigData reconstruction successful!")
                print(f"  Generated images shape: {generated_images.shape}")
                print(f"  Value range: [{generated_images.min():.3f}, {generated_images.max():.3f}]")

                # Create visualization with original stimulus and reconstruction
                fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                fig.suptitle('HMAD Original - MindBigData: Original Stimulus vs Reconstruction', fontsize=14)

                # Create simple digit images as "original stimulus"
                for i in range(min(3, generated_images.shape[0])):
                    digit = mindbig_labels[i].item()

                    # Row 1: Original stimulus from dataset
                    original_img = load_original_stimulus('digit', digit)
                    axes[0, i].imshow(original_img)
                    axes[0, i].set_title(f'Original: Digit {digit}')
                    axes[0, i].axis('off')

                    # Row 2: Reconstruction
                    recon_img = generated_images[i].cpu().numpy().transpose(1, 2, 0)
                    recon_img = np.clip(recon_img, 0, 1)

                    axes[1, i].imshow(recon_img)
                    axes[1, i].set_title(f'Reconstruction: Digit {digit}')
                    axes[1, i].axis('off')

                # Add row labels
                axes[0, 0].text(-0.1, 0.5, 'Original\nStimulus', transform=axes[0, 0].transAxes,
                               fontsize=12, fontweight='bold', ha='right', va='center', rotation=90)
                axes[1, 0].text(-0.1, 0.5, 'HMAD\nReconstruction', transform=axes[1, 0].transAxes,
                               fontsize=12, fontweight='bold', ha='right', va='center', rotation=90)

                plt.tight_layout()
                plt.savefig('../hmad_original_mindbigdata_comparison.png', dpi=150, bbox_inches='tight')
                plt.close()

                print(f"✓ Visualization saved: hmad_original_mindbigdata_reconstruction.png")

            except Exception as e:
                print(f"✗ MindBigData reconstruction failed: {e}")

        # Test Crell
        if crell_eeg is not None:
            print(f"\nTesting Crell reconstruction (shape: {crell_eeg.shape})...")
            crell_eeg = crell_eeg.to(device)

            try:
                outputs = model(crell_eeg, 'crell')
                generated_images = outputs['generated_images']

                print(f"✓ Crell reconstruction successful!")
                print(f"  Generated images shape: {generated_images.shape}")
                print(f"  Value range: [{generated_images.min():.3f}, {generated_images.max():.3f}]")

                # Create visualization with original stimulus and reconstruction
                fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                fig.suptitle('HMAD Original - Crell: Original Stimulus vs Reconstruction', fontsize=14)

                letter_names = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']

                for i in range(min(3, generated_images.shape[0])):
                    letter_idx = crell_labels[i].item()
                    letter = letter_names[letter_idx] if letter_idx < len(letter_names) else 'unknown'

                    # Row 1: Original stimulus from dataset
                    original_img = load_original_stimulus('letter', letter_idx)
                    axes[0, i].imshow(original_img)
                    axes[0, i].set_title(f'Original: Letter {letter.upper()}')
                    axes[0, i].axis('off')

                    # Row 2: Reconstruction
                    recon_img = generated_images[i].cpu().numpy().transpose(1, 2, 0)
                    recon_img = np.clip(recon_img, 0, 1)

                    axes[1, i].imshow(recon_img)
                    axes[1, i].set_title(f'Reconstruction: Letter {letter.upper()}')
                    axes[1, i].axis('off')

                # Add row labels
                axes[0, 0].text(-0.1, 0.5, 'Original\nStimulus', transform=axes[0, 0].transAxes,
                               fontsize=12, fontweight='bold', ha='right', va='center', rotation=90)
                axes[1, 0].text(-0.1, 0.5, 'HMAD\nReconstruction', transform=axes[1, 0].transAxes,
                               fontsize=12, fontweight='bold', ha='right', va='center', rotation=90)

                plt.tight_layout()
                plt.savefig('../hmad_original_crell_comparison.png', dpi=150, bbox_inches='tight')
                plt.close()

                print(f"✓ Visualization saved: hmad_original_crell_reconstruction.png")

            except Exception as e:
                print(f"✗ Crell reconstruction failed: {e}")
    
    print("\n" + "="*60)
    print("HMAD ORIGINAL RECONSTRUCTION TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    test_hmad_reconstruction()
