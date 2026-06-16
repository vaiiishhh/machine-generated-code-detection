#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
System Verification Script for CoDET-M4 Authorship Analysis
Run this before training to verify your setup is correct.
"""

import sys
import subprocess

def check_python_version():
    """Check Python version"""
    print("🐍 Python Version Check")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ⚠️  WARNING: Python 3.8+ recommended")
    else:
        print("   ✅ OK")
    print()

def check_cuda():
    """Check CUDA availability"""
    print("🔥 CUDA Check")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA Available")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("   ⚠️  CUDA not available - will use CPU (very slow)")
    except ImportError:
        print("   ❌ PyTorch not installed")
    print()

def check_packages():
    """Check required packages"""
    print("📦 Package Check")
    
    required = {
        'torch': '2.0.0',
        'transformers': '4.30.0',
        'datasets': '2.14.0',
        'sklearn': '1.3.0',
    }
    
    for package, min_version in required.items():
        try:
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                mod = __import__(package)
                version = mod.__version__
            
            print(f"   ✅ {package:20s} {version}")
        except ImportError:
            print(f"   ❌ {package:20s} NOT INSTALLED")
    print()

def check_disk_space():
    """Check available disk space"""
    print("💾 Disk Space Check")
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        print(f"   Total: {total / 1e9:.2f} GB")
        print(f"   Free:  {free / 1e9:.2f} GB")
        if free / 1e9 < 10:
            print("   ⚠️  WARNING: Less than 10GB free space")
        else:
            print("   ✅ OK")
    except Exception as e:
        print(f"   ⚠️  Could not check disk space: {e}")
    print()

def check_model_access():
    """Check if models can be downloaded"""
    print("🤖 Model Access Check")
    try:
        from transformers import AutoTokenizer
        print("   Testing UniXcoder access...")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
        print("   ✅ UniXcoder accessible")
    except Exception as e:
        print(f"   ⚠️  Issue accessing models: {e}")
    print()

def check_dataset_access():
    """Check if dataset can be loaded"""
    print("📚 Dataset Access Check")
    try:
        from datasets import load_dataset
        print("   Testing CoDET-M4 access...")
        # Just load dataset info, don't download
        dataset_info = load_dataset("DaniilOr/CoDET-M4", split="train", streaming=True)
        print("   ✅ CoDET-M4 accessible")
    except Exception as e:
        print(f"   ⚠️  Issue accessing dataset: {e}")
    print()

def estimate_training_time():
    """Estimate training time"""
    print("⏱️  Training Time Estimate")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            
            # Rough estimates based on GPU
            estimates = {
                "T4": "15-20 min/model (3k samples), 2-3 hrs/model (full)",
                "A100": "5-8 min/model (3k samples), 30-45 min/model (full)",
                "H100": "3-5 min/model (3k samples), 20-30 min/model (full)",
                "V100": "10-15 min/model (3k samples), 1-2 hrs/model (full)",
            }
            
            found = False
            for gpu_type, estimate in estimates.items():
                if gpu_type in gpu_name:
                    print(f"   GPU: {gpu_name}")
                    print(f"   Estimate: {estimate}")
                    found = True
                    break
            
            if not found:
                print(f"   GPU: {gpu_name}")
                print(f"   Estimate: Unknown (GPU not in database)")
        else:
            print("   ⚠️  No GPU - CPU training will be very slow (hours to days)")
    except:
        pass
    print()

def recommend_config():
    """Recommend configuration based on hardware"""
    print("⚙️  Recommended Configuration")
    try:
        import torch
        if torch.cuda.is_available():
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            if mem_gb < 12:
                print("   ⚠️  Low GPU memory detected")
                print("   Recommended config:")
                print("      BATCH_SIZE = 4")
                print("      GRAD_ACCUM_STEPS = 64")
                print("      MAX_LENGTH = 128")
                print("      MAX_SAMPLES = 1000 (for testing)")
            elif mem_gb < 16:
                print("   Recommended config:")
                print("      BATCH_SIZE = 8")
                print("      GRAD_ACCUM_STEPS = 32")
                print("      MAX_LENGTH = 256")
                print("      MAX_SAMPLES = 3000 (for testing)")
            else:
                print("   ✅ Good GPU memory")
                print("   Recommended config:")
                print("      BATCH_SIZE = 16")
                print("      GRAD_ACCUM_STEPS = 16")
                print("      MAX_LENGTH = 256")
                print("      MAX_SAMPLES = None (full dataset)")
        else:
            print("   ⚠️  No GPU available")
            print("   Training on CPU is not recommended")
    except:
        pass
    print()

def main():
    print("\n" + "="*70)
    print("  CoDET-M4 Authorship Analysis - System Verification")
    print("="*70 + "\n")
    
    check_python_version()
    check_cuda()
    check_packages()
    check_disk_space()
    check_model_access()
    check_dataset_access()
    estimate_training_time()
    recommend_config()
    
    print("="*70)
    print("  Verification Complete!")
    print("="*70)
    print("\nIf all checks passed, you're ready to run:")
    print("  python codet_m4_authorship.py")
    print("\nFor quick test on limited data:")
    print("  1. Edit Config in codet_m4_authorship.py")
    print("  2. Set MAX_SAMPLES = 3000")
    print("  3. Set MODEL_TO_TRAIN = 'unixcoder'")
    print("  4. Set NUM_EPOCHS = 2")
    print()

if __name__ == "__main__":
    main()
