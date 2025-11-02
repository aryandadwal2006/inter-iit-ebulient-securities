import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime

class TFTTradingStrategy:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()
        self.position = 0
        self.entry_price = 0
        self.trades = []
        
    def predict(self, sequence: np.ndarray) -> Tuple[int, float]:
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            output = self.model(seq_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities) - 1
            confidence = probabilities[np.argmax(probabilities)]
            return predicted_class, confidence
    
    def should_enter_trade(self, prediction: int, confidence: float, current_price: float) -> bool:
        if confidence < self.config.MIN_PREDICTION_CONFIDENCE:
            return False
        if self.position != 0:
            return False
        if prediction == 0:
            return False
        return True
    
    def should_exit_trade(self, current_price: float) -> bool:
        if self.position == 0:
            return False
        pnl_pct = (current_price - self.entry_price) / self.entry_price * self.position
        if pnl_pct >= self.config.PROFIT_THRESHOLD:
            return True
        if pnl_pct <= -self.config.STOP_LOSS:
            return True
        return False
    
    def execute_trade(self, action: str, price: float, timestamp: datetime):
        if action == 'BUY':
            self.position = 1
            self.entry_price = price
        elif action == 'SELL':
            self.position = -1
            self.entry_price = price
        elif action == 'CLOSE':
            if self.position != 0:
                pnl = (price - self.entry_price) * self.position
                pnl_pct = pnl / self.entry_price
                self.trades.append({
                    'entry_time': self.entry_time,
                    'exit_time': timestamp,
                    'entry_price': self.entry_price,
                    'exit_price': price,
                    'position': self.position,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
                self.position = 0
                self.entry_price = 0
        if action in ['BUY', 'SELL']:
            self.entry_time = timestamp
    
    def run_day(self, df: pd.DataFrame) -> List[Dict]:
        lookback = self.config.LOOKBACK_WINDOW
        feature_cols = [col for col in df.columns if col not in ['Time', 'day', 'target_return', 'target_direction'] and not col.startswith('future_return')]
        for i in range(lookback, len(df)):
            sequence = df[feature_cols].iloc[i-lookback:i].values
            current_price = df['Price'].iloc[i]
            current_time = df['Time'].iloc[i]
            prediction, confidence = self.predict(sequence)
            if self.should_exit_trade(current_price):
                self.execute_trade('CLOSE', current_price, current_time)
            elif self.should_enter_trade(prediction, confidence, current_price):
                if prediction == 1:
                    self.execute_trade('BUY', current_price, current_time)
                elif prediction == -1:
                    self.execute_trade('SELL', current_price, current_time)
        if self.position != 0:
            final_price = df['Price'].iloc[-1]
            final_time = df['Time'].iloc[-1]
            self.execute_trade('CLOSE', final_price, final_time)
        return self.trades


class PerformanceEvaluator:
    def __init__(self, config):
        self.config = config
    
    def calculate_returns(self, trades: List[Dict]) -> pd.Series:
        if not trades:
            return pd.Series([0])
        returns = [trade['pnl_pct'] for trade in trades]
        cumulative_returns = (1 + pd.Series(returns)).cumprod() - 1
        return cumulative_returns
    
    def calculate_drawdown(self, cumulative_returns: pd.Series) -> Tuple[float, pd.Series]:
        cumulative_wealth = 1 + cumulative_returns
        running_max = cumulative_wealth.expanding().max()
        drawdown = (cumulative_wealth - running_max) / running_max
        max_drawdown = drawdown.min()
        return abs(max_drawdown), drawdown
    
    def calculate_sharpe_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        if len(returns) == 0:
            return 0
        mean_return = returns.mean()
        std_return = returns.std()
        if std_return == 0:
            return 0
        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        return sharpe
    
    def calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> float:
        if max_drawdown == 0:
            return 0
        return annual_return / max_drawdown
    
    def evaluate(self, trades: List[Dict], days_traded: int) -> Dict:
        if not trades:
            return {
                'total_trades': 0,
                'annual_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'calmar_ratio': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
        cumulative_returns = self.calculate_returns(trades)
        total_return = cumulative_returns.iloc[-1]
        annual_return = (1 + total_return) ** (279 / days_traded) - 1
        max_drawdown, _ = self.calculate_drawdown(cumulative_returns)
        returns = pd.Series([trade['pnl_pct'] for trade in trades])
        sharpe_ratio = self.calculate_sharpe_ratio(returns, periods_per_year=279)
        calmar_ratio = self.calculate_calmar_ratio(annual_return, max_drawdown)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0
        results = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades else float('inf')
        }
        return results
    
    def print_results(self, results: Dict):
        print("TRADING STRATEGY PERFORMANCE RESULTS")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Winning Trades: {results['winning_trades']}")
        print(f"Losing Trades: {results['losing_trades']}")
        print(f"Win Rate: {results['win_rate']*100:.2f}%")
        print(f"Total Return: {results['total_return']*100:.2f}%")
        print(f"Annualized Return: {results['annual_return']*100:.2f}%")
        print(f"Maximum Drawdown: {results['max_drawdown']*100:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"Calmar Ratio: {results['calmar_ratio']:.3f}")
        print(f"Average Win: {results['avg_win']*100:.2f}%")
        print(f"Average Loss: {results['avg_loss']*100:.2f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        meets_return = results['annual_return'] >= self.config.MIN_ANNUAL_RETURN
        meets_drawdown = results['max_drawdown'] <= self.config.MAX_DRAWDOWN
        print("COMPETITION REQUIREMENTS:")
        print(f"Annual Return >= 20%: {'PASS' if meets_return else 'FAIL'}")
        print(f"Max Drawdown <= 10%: {'PASS' if meets_drawdown else 'FAIL'}")
        if meets_return and meets_drawdown:
            print("Strategy qualifies for competition letsgoo")
        else:
            print("Strategy doesnot meet minimum requirements")
