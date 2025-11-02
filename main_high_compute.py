import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_processor import DataProcessor
from tft_model import TemporalFusionTransformer, TradingDataset, TFTTrainer
from trading_strategy import TFTTradingStrategy, PerformanceEvaluator

def main():
    config = Config()
    
    if config.USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    print("STEP 1: DATA PREPARATION")
    
    data_processor = DataProcessor(config)
    
    print("Loading and processing training data (days 0-199)...")
    X_train, y_train = data_processor.prepare_training_data(start_day=0, end_day=200)
    
    print(f"Training sequences shape: {X_train.shape}")
    print(f"Training targets shape: {y_train.shape}")
    print(f"Target distribution: {np.bincount(y_train.astype(int) + 1)}")
    
    dataset = TradingDataset(X_train, y_train)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    print("STEP 2: MODEL TRAINING")
    
    num_features = X_train.shape[2]
    model = TemporalFusionTransformer(
        num_features=num_features,
        hidden_size=config.HIDDEN_SIZE,
        num_attention_heads=config.ATTENTION_HEADS,
        dropout=config.DROPOUT
    ).to(device)
    
    print(f"Input features: {num_features}")
    print(f"Hidden size: {config.HIDDEN_SIZE}")
    print(f"Attention heads: {config.ATTENTION_HEADS}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    trainer = TFTTrainer(model, config, device)
    
    print(f"Starting training for {config.MAX_EPOCHS} epochs")
    
    trainer.train(train_loader, val_loader, config.MAX_EPOCHS)
    
    model.load_state_dict(torch.load(f"{config.MODEL_DIR}/best_tft_model.pt"))
    print("Best model loaded")
    
    print("STEP 3: STRATEGY BACKTESTING")
    
    print("Testing on days 200-278")
    
    strategy = TFTTradingStrategy(model, config, device)
    evaluator = PerformanceEvaluator(config)
    
    all_trades = []
    test_days = list(range(200, 279))
    
    for day_num in test_days:
        print(f"Processing day {day_num}")
        df = data_processor.load_day_file(day_num)
        if df is not None:
            df = data_processor.filter_stable_period(df)
            df = data_processor.handle_missing_values(df)
            df = data_processor.normalize_features(df, fit=False)
            day_trades = strategy.run_day(df)
            all_trades.extend(day_trades)
    
    print("Backtesting complete")
    
    results = evaluator.evaluate(all_trades, days_traded=len(test_days))
    evaluator.print_results(results)
    
    results_file = f"{config.RESULTS_DIR}/strategy_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to: {results_file}")
    
    trades_df = pd.DataFrame(all_trades)
    trades_file = f"{config.RESULTS_DIR}/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    trades_df.to_csv(trades_file, index=False)
    
    print(f"Trades saved to: {trades_file}")
    print("EXECUTION COMPLETE")

if __name__ == "__main__":
    main()
