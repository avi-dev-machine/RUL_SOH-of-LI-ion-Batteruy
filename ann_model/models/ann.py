"""
Artificial Neural Network Architecture for Battery SOH & RUL Prediction
=========================================================================
Built with PyTorch (supports Python 3.14+).

Two architectures:

1. StandardANN  ─  Feed-forward MLP
   Input → [Linear → BN → ReLU → Dropout] × N → Linear(2)

2. MultiTaskANN  ─  Shared trunk + separate SOH/RUL heads
   Input → Shared [Linear-BN-ReLU-Drop] × m
         ↙                    ↘
   SOH Head [Linear×k]    RUL Head [Linear×k]
         ↓                         ↓
   Linear(1)                  Linear(1)
         ↘                    ↙
         cat → output(2)

Regularisation:
  • Batch Normalization  — stable training
  • Dropout             — prevents overfitting
  • L2 (weight_decay)   — parameter shrinkage via Adam

Weight init: Kaiming uniform (default for Linear layers in PyTorch)
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ══════════════════════════════════════════════════════════════════════════════
#  Shared building block
# ══════════════════════════════════════════════════════════════════════════════

def _dense_block(in_features: int,
                  out_features: int,
                  dropout: float,
                  name_prefix: str = '') -> nn.Sequential:
    """Linear → BatchNorm → ReLU → Dropout block."""
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=False),  # bias in BN
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(p=dropout),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Architecture 1 — Standard Feed-Forward ANN
# ══════════════════════════════════════════════════════════════════════════════

class StandardANN(nn.Module):
    """
    Standard fully-connected ANN for multi-output regression.

    Args:
        input_dim    : Number of input features
        hidden_units : List of neurons per hidden layer
        output_dim   : 2 (SOH + RUL)
        dropout_rate : Dropout probability
    """

    def __init__(self,
                 input_dim:    int,
                 hidden_units: list  = (256, 128, 64, 32),
                 output_dim:   int   = 2,
                 dropout_rate: float = 0.20):
        super().__init__()

        layers = []
        prev   = input_dim
        for units in hidden_units:
            layers.append(_dense_block(prev, units, dropout_rate))
            prev = units

        self.encoder = nn.Sequential(*layers)
        self.output  = nn.Linear(prev, output_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.encoder(x))


# ══════════════════════════════════════════════════════════════════════════════
#  Architecture 2 — Multi-task ANN
# ══════════════════════════════════════════════════════════════════════════════

class MultiTaskANN(nn.Module):
    """
    Multi-task ANN with a shared encoder trunk and separate output heads.

    Args:
        input_dim      : Input feature count
        shared_units   : Neurons in each shared-trunk layer
        head_units     : Neurons in each task-head layer
        dropout_rate   : Dropout probability
    """

    def __init__(self,
                 input_dim:    int,
                 shared_units: list  = (128, 64),
                 head_units:   list  = (32,),
                 dropout_rate: float = 0.20):
        super().__init__()

        # Shared trunk
        trunk = []
        prev  = input_dim
        for units in shared_units:
            trunk.append(_dense_block(prev, units, dropout_rate))
            prev = units
        self.trunk = nn.Sequential(*trunk)

        # SOH head
        soh_layers, rul_layers = [], []
        for units in head_units:
            soh_layers.append(_dense_block(prev, units, 0.0))
            rul_layers.append(_dense_block(prev, units, 0.0))
            prev = units
        self.soh_head = nn.Sequential(*soh_layers, nn.Linear(prev, 1))
        self.rul_head = nn.Sequential(*rul_layers, nn.Linear(prev, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared   = self.trunk(x)
        soh_pred = self.soh_head(shared)   # (N, 1)
        rul_pred = self.rul_head(shared)   # (N, 1)
        return torch.cat([soh_pred, rul_pred], dim=1)   # (N, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  High-level wrapper  BatteryANN
# ══════════════════════════════════════════════════════════════════════════════

class BatteryANN:
    """
    Unified wrapper for StandardANN and MultiTaskANN.
    Handles training loop, early stopping, LR scheduling, save/load.
    """

    ARCHITECTURES = ('standard', 'multitask')

    def __init__(self,
                 input_dim:     int,
                 architecture:  str   = 'standard',
                 hidden_units:  list  = None,
                 dropout_rate:  float = 0.20,
                 weight_decay:  float = 1e-4,
                 learning_rate: float = 1e-3):

        if architecture not in self.ARCHITECTURES:
            raise ValueError(f"architecture must be one of {self.ARCHITECTURES}")

        self.architecture  = architecture
        self.input_dim     = input_dim
        self.hidden_units  = hidden_units or [256, 128, 64, 32]
        self.dropout_rate  = dropout_rate
        self.weight_decay  = weight_decay
        self.learning_rate = learning_rate
        self.history       = {'train_loss': [], 'val_loss': [],
                               'train_mae' : [], 'val_mae' : []}
        self.best_val_loss = math.inf
        self.device        = torch.device('cpu')   # CPU-only for compatibility
        self.model         = self._build()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n✅ Built [{architecture}] BatteryANN — "
              f"{input_dim} inputs, "
              f"{total_params:,} parameters")

    # ── Private ────────────────────────────────────────────────────
    def _build(self) -> nn.Module:
        if self.architecture == 'multitask':
            return MultiTaskANN(
                input_dim    = self.input_dim,
                shared_units = self.hidden_units[:2],
                head_units   = self.hidden_units[2:] or [32],
                dropout_rate = self.dropout_rate,
            ).to(self.device)
        else:
            return StandardANN(
                input_dim    = self.input_dim,
                hidden_units = self.hidden_units,
                output_dim   = 2,
                dropout_rate = self.dropout_rate,
            ).to(self.device)

    @staticmethod
    def _to_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32)

    # ── Training ───────────────────────────────────────────────────
    def fit(self,
            X_train:    np.ndarray,
            y_train:    np.ndarray,
            X_val:      np.ndarray,
            y_val:      np.ndarray,
            epochs:     int = 300,
            batch_size: int = 32,
            patience:   int = 30,
            checkpoint_dir: str = None) -> dict:
        """
        Train with:
          • Adam optimiser + weight_decay (L2)
          • ReduceLROnPlateau  (factor=0.5, patience=12)
          • EarlyStopping      (patience=30, restore best weights)

        Returns: history dict
        """
        # Data loaders
        train_ds = TensorDataset(self._to_tensor(X_train), self._to_tensor(y_train))
        val_ds   = TensorDataset(self._to_tensor(X_val),   self._to_tensor(y_val))
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

        optimiser  = torch.optim.Adam(self.model.parameters(),
                                       lr=self.learning_rate,
                                       weight_decay=self.weight_decay)
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode='min', factor=0.5, patience=12, min_lr=1e-7
        )
        criterion  = nn.MSELoss()
        best_state = None
        no_improve = 0

        print(f"\n🚀 Training [{self.architecture}]: "
              f"epochs={epochs}, batch={batch_size}, "
              f"patience={patience}")

        for epoch in range(1, epochs + 1):
            # ── Train ───────────────────────────────────────────────
            self.model.train()
            t_loss, t_mae = 0.0, 0.0
            for Xb, yb in train_dl:
                Xb, yb   = Xb.to(self.device), yb.to(self.device)
                optimiser.zero_grad()
                pred     = self.model(Xb)
                loss     = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimiser.step()
                t_loss += loss.item() * len(Xb)
                t_mae  += torch.mean(torch.abs(pred - yb)).item() * len(Xb)
            t_loss /= len(train_ds)
            t_mae  /= len(train_ds)

            # ── Validate ────────────────────────────────────────────
            self.model.eval()
            v_loss, v_mae = 0.0, 0.0
            with torch.no_grad():
                for Xb, yb in val_dl:
                    Xb, yb  = Xb.to(self.device), yb.to(self.device)
                    pred     = self.model(Xb)
                    v_loss  += criterion(pred, yb).item() * len(Xb)
                    v_mae   += torch.mean(torch.abs(pred - yb)).item() * len(Xb)
            v_loss /= len(val_ds)
            v_mae  /= len(val_ds)

            scheduler.step(v_loss)

            self.history['train_loss'].append(t_loss)
            self.history['val_loss'].append(v_loss)
            self.history['train_mae'].append(t_mae)
            self.history['val_mae'].append(v_mae)

            if epoch % 20 == 0 or epoch == 1:
                lr_now = optimiser.param_groups[0]['lr']
                print(f"  Epoch {epoch:4d}/{epochs} | "
                      f"train_loss: {t_loss:.5f}  val_loss: {v_loss:.5f}  "
                      f"lr: {lr_now:.2e}")

            # ── Early stopping ──────────────────────────────────────
            if v_loss < self.best_val_loss - 1e-7:
                self.best_val_loss = v_loss
                best_state         = {k: v.clone() for k, v in
                                      self.model.state_dict().items()}
                no_improve         = 0
                if checkpoint_dir:
                    self.save(checkpoint_dir, tag='best')
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\n  ⏹ Early stopping at epoch {epoch} "
                          f"(no improvement for {patience} epochs)")
                    break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"  ✅ Restored best weights (val_loss={self.best_val_loss:.5f})")

        return self.history

    # ── Inference ──────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions as numpy array (N, 2)."""
        self.model.eval()
        with torch.no_grad():
            t   = self._to_tensor(X).to(self.device)
            out = self.model(t)
        return out.cpu().numpy()

    # ── Persistence ────────────────────────────────────────────────
    def save(self, directory: str, tag: str = 'final') -> None:
        os.makedirs(directory, exist_ok=True)
        fname = f'battery_ann_{self.architecture}_{tag}.pt'
        path  = os.path.join(directory, fname)
        torch.save({
            'architecture' : self.architecture,
            'input_dim'    : self.input_dim,
            'hidden_units' : self.hidden_units,
            'dropout_rate' : self.dropout_rate,
            'model_state'  : self.model.state_dict(),
        }, path)
        print(f"  💾 Model saved → {path}")

    def load(self, directory: str, tag: str = 'final') -> 'BatteryANN':
        fname = f'battery_ann_{self.architecture}_{tag}.pt'
        path  = os.path.join(directory, fname)
        ckpt  = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()
        print(f"  ✅ Model loaded ← {path}")
        return self
