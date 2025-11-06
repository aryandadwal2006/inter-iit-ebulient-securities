import torch
import numpy as np
import sys
from pathlib import Path

def test_imports():
    packages = {
        'torch': None,
        'numpy': None,
        'optuna': None,
        'pandas': None,
        'dask': None,
    }
    
    for package in packages:
        try:
            mod = __import__(package)
            packages[package] = getattr(mod, '__version__', 'OK')
            print(f"{package:20s}: {packages[package]}")
        except ImportError:
            packages[package] = 'NOT FOUND'
            print(f"{package:20s}: NOT FOUND")
    
    try:
        import cudf
        print(f"{'cudf':20s}: {cudf.__version__}")
    except ImportError:
        print(f"{'cudf':20s}: NOT FOUND")
    
    return all(v != 'NOT FOUND' for v in packages.values())


def test_cuda():
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"Memory: {props.total_memory / 1e9:.2f} GB")
        return True
    else:
        print("CUDA Available: No")
        return False


def test_config():
    try:
        from config import Config
        config = Config()
        print("Configuration loaded")
        print(f"Hidden Size: {config.HIDDEN_SIZE}")
        print(f"LSTM Layers: {config.LSTM_LAYERS}")
        print(f"Attention Heads: {config.ATTENTION_HEADS}")
        print(f"Batch Size: {config.BATCH_SIZE}")
        print(f"Features: {config.get_num_features()}")
        return True
    except Exception as e:
        print(f"Configuration error: {e}")
        return False


def test_model():
    try:
        from config import Config
        from tft_model_enhanced import TemporalFusionTransformer
        
        config = Config()
        model = TemporalFusionTransformer(
            num_features=config.get_num_features(),
            hidden_size=config.HIDDEN_SIZE,
            lstm_layers=config.LSTM_LAYERS,
            num_attention_heads=config.ATTENTION_HEADS,
            dropout=config.DROPOUT,
            ffn_hidden_size=config.FFN_HIDDEN_SIZE
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("Model created successfully")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        batch = torch.randn(4, config.LOOKBACK_WINDOW, config.get_num_features()).to(device)
        output = model(batch)
        
        print(f"Input shape: {batch.shape}")
        print(f"Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"Model error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_paths():
    from config import Config
    config = Config()
    
    paths = {
        'DATA_DIR': config.DATA_DIR,
        'PARQUET_DIR': config.PARQUET_DIR,
        'MODEL_DIR': config.MODEL_DIR,
        'RESULTS_DIR': config.RESULTS_DIR,
        'OPTUNA_DIR': config.OPTUNA_DIR,
        'CACHE_DIR': config.CACHE_DIR,
    }
    
    all_exist = True
    for name, path in paths.items():
        exists = Path(path).exists()
        print(f"{name:20s}: {path} {'(exists)' if exists else '(will be created)'}")
        if name == 'DATA_DIR' and not exists:
            all_exist = False
    
    data_dir = Path(config.DATA_DIR)
    if data_dir.exists():
        csv_files = list(data_dir.glob("day*.csv"))
        print(f"Found {len(csv_files)} CSV files in {config.DATA_DIR}")
        if csv_files:
            print(f"Example: {csv_files[0].name}")
    else:
        print(f"Data directory not found. Please create {config.DATA_DIR} and add CSV files.")
    
    return all_exist


def test_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1e9
            
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            
            print(f"GPU {i}:")
            print(f"Total: {total_memory:.2f} GB")
            print(f"Allocated: {allocated:.2f} GB")
            print(f"Reserved: {reserved:.2f} GB")
            print(f"Free: {total_memory - reserved:.2f} GB")
    
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"System RAM Total: {ram.total / 1e9:.2f} GB")
        print(f"Available: {ram.available / 1e9:.2f} GB")
        print(f"Used: {ram.percent}%")
    except ImportError:
        print("psutil not installed, cannot check system RAM")
    
    return True


def main():
    results = {}
    
    results['imports'] = test_imports()
    results['cuda'] = test_cuda()
    results['config'] = test_config()
    results['model'] = test_model()
    results['paths'] = test_data_paths()
    results['memory'] = test_memory()
    
    print("TEST SUMMARY")
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test.upper():12s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("All tests passed. System ready.")
        print("Ensure data files are in ./EBY/ directory and run main_enhanced.py")
    else:
        print("Some tests failed. Fix issues before proceeding.")
        print("Check requirements, CUDA, and data directories.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
