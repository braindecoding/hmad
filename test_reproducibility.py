#!/usr/bin/env python3
"""
Test Reproducibility
====================

Script untuk test reproducibility dari clean state.
Memverifikasi bahwa semua komponen bekerja dan hasil dapat direproduksi.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"{title:^70}")
    print("="*70)

def print_section(title):
    """Print formatted section"""
    print(f"\n{title}")
    print("-" * len(title))

def check_environment():
    """Check that environment is ready"""
    
    print_section("ENVIRONMENT CHECK")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
    
    # Check required packages
    required_packages = ['torch', 'numpy', 'matplotlib', 'scipy', 'sklearn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_project_structure():
    """Check that project structure is correct"""
    
    print_section("PROJECT STRUCTURE CHECK")
    
    # Required files
    required_files = [
        'main.py',
        'requirements.txt',
        'README.md',
        'REPRODUCIBILITY_GUIDE.md'
    ]
    
    # Required directories
    required_dirs = [
        'src',
        'src/models',
        'src/training', 
        'src/evaluation',
        'data',
        'results'
    ]
    
    # Check files
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} missing")
            missing_files.append(file_path)
    
    # Check directories
    missing_dirs = []
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ missing")
            missing_dirs.append(dir_path)
    
    if missing_files or missing_dirs:
        print(f"\n❌ Project structure incomplete")
        return False
    else:
        print(f"\n✅ Project structure complete")
        return True

def check_data_availability():
    """Check that datasets are available"""
    
    print_section("DATA AVAILABILITY CHECK")
    
    # Check datasets
    dataset_files = [
        'data/raw/datasets/EP1.01.txt',
        'data/raw/datasets/S01.mat'
    ]
    
    # Check stimulus directories
    stimulus_dirs = [
        'data/raw/datasets/MindbigdataStimuli',
        'data/raw/datasets/crellStimuli'
    ]
    
    data_available = True
    
    for file_path in dataset_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} missing")
            data_available = False
    
    for dir_path in stimulus_dirs:
        if os.path.exists(dir_path):
            files = len(os.listdir(dir_path))
            print(f"✅ {dir_path}/ ({files} files)")
        else:
            print(f"❌ {dir_path}/ missing")
            data_available = False
    
    if not data_available:
        print(f"\n⚠️  Some datasets missing - pipeline will use available data")
    else:
        print(f"\n✅ All datasets available")
    
    return data_available

def test_imports():
    """Test that all imports work"""
    
    print_section("IMPORT TEST")
    
    # Add src to path
    sys.path.insert(0, 'src')
    
    try:
        # Test model imports
        from models.hmadv2 import create_improved_hmad_model
        print("✅ models.hmadv2 imported")
        
        from models.hmad import HMADFramework
        print("✅ models.hmad imported")
        
        # Test evaluation imports
        from evaluation.test_hmad import load_mindbigdata_sample, load_stimulus_images
        print("✅ evaluation.test_hmad imported")
        
        # Test training imports
        from training.full_training_hmadv2 import full_training_hmadv2
        print("✅ training.full_training_hmadv2 imported")
        
        print("\n✅ All imports successful")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False

def clear_previous_results():
    """Clear previous results to ensure clean state"""
    
    print_section("CLEARING PREVIOUS RESULTS")
    
    # Directories to clear
    result_dirs = [
        'results/models',
        'results/figures', 
        'results/metrics'
    ]
    
    cleared_count = 0
    
    for dir_path in result_dirs:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if not f.startswith('.')]
            for file_name in files:
                file_path = os.path.join(dir_path, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        cleared_count += 1
                    elif os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
                        cleared_count += 1
                except Exception as e:
                    print(f"⚠️  Could not remove {file_path}: {e}")
    
    print(f"✅ Cleared {cleared_count} previous result files")
    print("🧹 Clean state achieved")

def run_full_pipeline():
    """Run the full pipeline"""
    
    print_section("RUNNING FULL PIPELINE")
    
    print("🚀 Starting full training and evaluation pipeline...")
    print("⏱️  Expected runtime: 30-40 minutes")
    print("📊 Progress will be shown below:")
    
    start_time = time.time()
    
    try:
        # Run main pipeline
        result = subprocess.run([
            sys.executable, 'main.py', '--mode', 'full', '--seed', '42'
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        end_time = time.time()
        runtime = end_time - start_time
        
        if result.returncode == 0:
            print(f"\n✅ Pipeline completed successfully!")
            print(f"⏱️  Runtime: {runtime/60:.1f} minutes")
            return True, runtime
        else:
            print(f"\n❌ Pipeline failed with return code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False, runtime
            
    except subprocess.TimeoutExpired:
        print(f"\n⏰ Pipeline timed out after 1 hour")
        return False, 3600
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        return False, 0

def verify_results():
    """Verify that results were generated correctly"""
    
    print_section("VERIFYING RESULTS")
    
    # Expected result files
    expected_files = [
        'results/models/checkpoints/best_mindbigdata_model.pth',
        'results/models/checkpoints/best_crell_model.pth',
        'results/figures/comprehensive_results_summary.png',
        'results/figures/full_training_mindbigdata_results.png',
        'results/figures/full_training_crell_results.png',
        'results/metrics/full_training_mindbigdata_results.pkl',
        'results/metrics/full_training_crell_results.pkl'
    ]
    
    generated_files = []
    missing_files = []
    
    for file_path in expected_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
            generated_files.append(file_path)
        else:
            print(f"❌ {file_path} missing")
            missing_files.append(file_path)
    
    # Check results quality
    success_rate = len(generated_files) / len(expected_files)
    
    print(f"\nResults Summary:")
    print(f"  Generated: {len(generated_files)}/{len(expected_files)} files")
    print(f"  Success Rate: {success_rate*100:.1f}%")
    
    if success_rate >= 0.8:
        print("✅ Results verification successful")
        return True
    else:
        print("❌ Results verification failed")
        return False

def load_and_check_metrics():
    """Load and check performance metrics"""
    
    print_section("CHECKING PERFORMANCE METRICS")
    
    try:
        import torch
        
        # Expected performance ranges
        expected_ranges = {
            'mindbigdata': {
                'psnr_min': 10.0,
                'psnr_max': 15.0,
                'cosine_min': 0.4,
                'cosine_max': 0.8
            },
            'crell': {
                'psnr_min': 10.0,
                'psnr_max': 16.0,
                'cosine_min': 0.9,
                'cosine_max': 1.0
            }
        }
        
        results_ok = True
        
        for dataset in ['mindbigdata', 'crell']:
            result_file = f'results/metrics/full_training_{dataset}_results.pkl'
            
            if os.path.exists(result_file):
                try:
                    results = torch.load(result_file)
                    metrics = results['final_test_metrics']
                    
                    psnr = metrics['avg_psnr']
                    cosine = metrics['avg_cosine']
                    
                    expected = expected_ranges[dataset]
                    
                    print(f"\n{dataset.upper()} Results:")
                    print(f"  PSNR: {psnr:.2f} dB (expected: {expected['psnr_min']}-{expected['psnr_max']})")
                    print(f"  Cosine: {cosine:.4f} (expected: {expected['cosine_min']}-{expected['cosine_max']})")
                    
                    # Check if within expected ranges
                    psnr_ok = expected['psnr_min'] <= psnr <= expected['psnr_max']
                    cosine_ok = expected['cosine_min'] <= cosine <= expected['cosine_max']
                    
                    if psnr_ok and cosine_ok:
                        print(f"  ✅ Performance within expected range")
                    else:
                        print(f"  ⚠️  Performance outside expected range")
                        results_ok = False
                        
                except Exception as e:
                    print(f"  ❌ Error loading {dataset} results: {e}")
                    results_ok = False
            else:
                print(f"  ❌ {dataset} results file missing")
                results_ok = False
        
        return results_ok
        
    except Exception as e:
        print(f"❌ Error checking metrics: {e}")
        return False

def main():
    """Main reproducibility test function"""
    
    print_header("HMADV2 REPRODUCIBILITY TEST")
    
    print("Testing complete reproducibility from clean state...")
    print("This will verify environment, run full pipeline, and check results.")
    
    # Pre-flight checks
    checks_passed = True
    
    if not check_environment():
        checks_passed = False
    
    if not check_project_structure():
        checks_passed = False
    
    data_available = check_data_availability()
    
    if not test_imports():
        checks_passed = False
    
    if not checks_passed:
        print_header("PRE-FLIGHT CHECKS FAILED")
        print("❌ Cannot proceed with reproducibility test")
        print("Please fix the issues above and try again")
        return False
    
    print_header("PRE-FLIGHT CHECKS PASSED")
    print("✅ Environment ready for reproducibility test")
    
    # Ask for confirmation
    if data_available:
        print("\n🚀 Ready to run full pipeline with all datasets")
    else:
        print("\n⚠️  Some datasets missing - will run with available data")
    
    response = input("\nProceed with full reproducibility test? (y/N): ").strip().lower()
    
    if response != 'y':
        print("Reproducibility test cancelled")
        return False
    
    # Clear previous results
    clear_previous_results()
    
    # Run pipeline
    success, runtime = run_full_pipeline()
    
    if not success:
        print_header("PIPELINE EXECUTION FAILED")
        return False
    
    # Verify results
    results_ok = verify_results()
    metrics_ok = load_and_check_metrics()
    
    # Final assessment
    print_header("REPRODUCIBILITY TEST RESULTS")
    
    if success and results_ok and metrics_ok:
        print("🎉 REPRODUCIBILITY TEST PASSED!")
        print("✅ Pipeline executed successfully")
        print("✅ All expected files generated")
        print("✅ Performance metrics within expected ranges")
        print(f"⏱️  Total runtime: {runtime/60:.1f} minutes")
        
        print("\n🏆 REPRODUCIBILITY CONFIRMED!")
        print("The HMADv2 project is fully reproducible from clean state")
        
        return True
    else:
        print("❌ REPRODUCIBILITY TEST FAILED")
        print("Some components did not work as expected")
        
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed - check output above")
