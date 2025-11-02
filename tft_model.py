import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple

class TemporalFusionTransformer(nn.Module):
    def __init__(self, num_features: int, hidden_size: int = 32, num_attention_heads: int = 2, dropout: float = 0.2, num_classes: int = 3):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.input_embedding = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        self.vsn = VariableSelectionNetwork(num_features, hidden_size, dropout)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout if dropout > 0 else 0.0,
            batch_first=True
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        self.grn = GatedResidualNetwork(hidden_size, hidden_size, dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param, gain=0.1)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        if torch.isnan(x).any():
            print("WARNING: NaN detected in input!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        x_selected = self.vsn(x)
        lstm_out, (hidden, cell) = self.lstm(x_selected)
        if torch.isnan(lstm_out).any():
            print("WARNING: NaN detected after LSTM!")
            lstm_out = torch.nan_to_num(lstm_out, nan=0.0)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.dropout(attn_out) + lstm_out
        last_output = attn_out[:, -1, :]
        grn_out = self.grn(last_output)
        output = self.output_layer(grn_out)
        if torch.isnan(output).any():
            print("WARNING: NaN detected in output!")
            output = torch.nan_to_num(output, nan=0.0)
        return output


class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_features, hidden_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.feature_transform = nn.Linear(num_features, hidden_size)
        self.context = nn.Linear(num_features, hidden_size)
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        transformed = self.feature_transform(x)
        transformed = self.layer_norm(transformed)
        context = self.context(x)
        scores = self.attention(torch.tanh(context))
        scores = torch.clamp(scores, -10, 10)
        weights = torch.softmax(scores, dim=1)
        output = transformed * weights
        return self.dropout(output)


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.elu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        gate = torch.sigmoid(self.gate(x))
        output = gate * out + (1 - gate) * x
        return self.layer_norm(output)


class TradingDataset(Dataset):
    def __init__(self, sequences, targets):
        sequences = np.nan_to_num(sequences, nan=0.0, posinf=1e6, neginf=-1e6)
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.LongTensor(targets + 1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class TFTTrainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.0001,
            weight_decay=1e-5
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
        self.model_saved = False

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        nan_batches = 0
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            if torch.isnan(outputs).any():
                nan_batches += 1
                print(f"NaN detected in batch {batch_idx}, skipping...")
                continue
            loss = self.criterion(outputs, targets)
            if torch.isnan(loss):
                nan_batches += 1
                print(f"NaN loss in batch {batch_idx}, skipping...")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        if nan_batches > 0:
            print(f"WARNING: {nan_batches} batches had NaN values!")
        avg_loss = total_loss / max(len(train_loader) - nan_batches, 1)
        accuracy = 100 * correct / total if total > 0 else 0
        return avg_loss, accuracy

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        valid_batches = 0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(sequences)
                if torch.isnan(outputs).any():
                    continue
                loss = self.criterion(outputs, targets)
                if torch.isnan(loss):
                    continue
                total_loss += loss.item()
                valid_batches += 1
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        avg_loss = total_loss / max(valid_batches, 1)
        accuracy = 100 * correct / total if total > 0 else 0
        return avg_loss, accuracy

    def train(self, train_loader, val_loader, epochs):
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 5
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print("-" * 50)
            if np.isnan(train_loss) or np.isnan(val_loss):
                print("ERROR: Training producing NaN! Stopping...")
                break
            if val_loss < best_val_loss and not np.isnan(val_loss):
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f"{self.config.MODEL_DIR}/best_tft_model.pt")
                self.model_saved = True
                print("Model saved!")
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
            self.scheduler.step(val_loss)
        if not self.model_saved:
            print("WARNING: No improvement during training, saving final model anyway...")
            torch.save(self.model.state_dict(), f"{self.config.MODEL_DIR}/best_tft_model.pt")
            self.model_saved = True
