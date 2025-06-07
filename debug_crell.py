#!/usr/bin/env python3
"""
Debug script untuk memeriksa struktur dataset Crell
"""

import scipy.io
import numpy as np

def debug_crell_structure(filepath: str):
    """Debug struktur file Crell .mat"""
    print(f"Debugging Crell file: {filepath}")
    print("="*60)
    
    try:
        data = scipy.io.loadmat(filepath)
        
        print("Top-level keys in .mat file:")
        for key in data.keys():
            if not key.startswith('__'):
                print(f"  {key}: {type(data[key])}, shape: {getattr(data[key], 'shape', 'N/A')}")
        
        print("\n" + "="*60)
        
        # Check round01_paradigm structure
        if 'round01_paradigm' in data:
            print("round01_paradigm structure:")
            paradigm_one = data['round01_paradigm']
            print(f"  Type: {type(paradigm_one)}")
            print(f"  Shape: {paradigm_one.shape}")

            if len(paradigm_one) > 0 and len(paradigm_one[0]) > 0:
                round_data = paradigm_one[0, 0]
                print(f"  Round data type: {type(round_data)}")

                if hasattr(round_data, 'dtype'):
                    print(f"  Available fields: {round_data.dtype.names}")

                    # Check each field
                    for field_name in round_data.dtype.names:
                        field_data = round_data[field_name]
                        print(f"    {field_name}: raw type {type(field_data)}, shape {getattr(field_data, 'shape', 'N/A')}")

                        # Try different ways to access the data
                        try:
                            if hasattr(field_data, 'shape') and len(field_data.shape) > 0:
                                if field_data.shape == (1, 1):
                                    actual_data = field_data[0, 0]
                                else:
                                    actual_data = field_data
                            else:
                                actual_data = field_data

                            print(f"      Extracted data shape: {getattr(actual_data, 'shape', 'N/A')}, dtype: {getattr(actual_data, 'dtype', 'N/A')}")

                            # Special check for marker data
                            if 'Marker' in field_name and hasattr(actual_data, 'flatten'):
                                sample_markers = actual_data.flatten()[:20]
                                print(f"      Sample markers: {sample_markers}")
                        except Exception as e:
                            print(f"      Error accessing {field_name}: {e}")

        print("\n" + "="*60)

        # Check round02_paradigm structure
        if 'round02_paradigm' in data:
            print("round02_paradigm structure:")
            paradigm_two = data['round02_paradigm']
            print(f"  Type: {type(paradigm_two)}")
            print(f"  Shape: {paradigm_two.shape}")

            if len(paradigm_two) > 0 and len(paradigm_two[0]) > 0:
                round_data = paradigm_two[0, 0]
                print(f"  Round data type: {type(round_data)}")

                if hasattr(round_data, 'dtype'):
                    print(f"  Available fields: {round_data.dtype.names}")
        
        print("\n" + "="*60)
        
        # Try to extract and analyze marker data
        print("Analyzing marker data...")

        for paradigm_key in ['round01_paradigm', 'round02_paradigm']:
            if paradigm_key not in data:
                continue
                
            print(f"\n{paradigm_key}:")
            paradigm_data = data[paradigm_key]
            
            if len(paradigm_data) > 0 and len(paradigm_data[0]) > 0:
                round_data = paradigm_data[0, 0]
                
                if hasattr(round_data, 'dtype') and 'ParadigmMarker_data' in round_data.dtype.names:
                    marker_data = round_data['ParadigmMarker_data'][0, 0].flatten()
                    marker_times = round_data['ParadigmMarker_time'][0, 0].flatten()
                    
                    print(f"  Total markers: {len(marker_data)}")
                    print(f"  Unique markers: {np.unique(marker_data)}")
                    print(f"  First 20 markers: {marker_data[:20]}")
                    
                    # Count specific markers
                    letter_markers = marker_data[marker_data >= 100]
                    phase_markers = marker_data[(marker_data >= 1) & (marker_data <= 5)]
                    
                    print(f"  Letter markers (>=100): {len(letter_markers)} found")
                    print(f"    Unique letters: {np.unique(letter_markers)}")
                    print(f"  Phase markers (1-5): {len(phase_markers)} found")
                    print(f"    Phase marker counts: {np.bincount(phase_markers, minlength=6)[1:6]}")
                    
                    # Check timing
                    if len(marker_times) > 0:
                        duration = marker_times[-1] - marker_times[0]
                        print(f"  Recording duration: {duration:.2f} seconds")
                
                # Check EEG data
                if hasattr(round_data, 'dtype') and 'BrainVisionRDA_data' in round_data.dtype.names:
                    eeg_data = round_data['BrainVisionRDA_data'][0, 0]
                    eeg_times = round_data['BrainVisionRDA_time'][0, 0].flatten()
                    
                    print(f"  EEG data shape: {eeg_data.shape}")
                    print(f"  EEG duration: {eeg_times[-1] - eeg_times[0]:.2f} seconds")
                    print(f"  EEG sampling rate: ~{len(eeg_times) / (eeg_times[-1] - eeg_times[0]):.1f} Hz")
        
    except Exception as e:
        print(f"Error debugging Crell file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_crell_structure("datasets/S01.mat")
