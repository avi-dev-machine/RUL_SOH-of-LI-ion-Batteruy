"""
Physics-Informed ANN (Multi-task)
===================================
Implements the "Physics-Informed ANN (Multi-task)" block from Fig 12.1.

Architecture:
    Input(19)
        → Shared Encoder [Dense-BN-ReLU-Drop] × 3
        ↙                              ↘
    SOH Head [Dense×2 → Linear(1)]    RUL Head [Dense×2 → Linear(1)]
        ↘                              ↙
            Concatenate → output [SOH, RUL]

Physics-Informed Loss (dashed orange arrow in Fig 12.1):
    L_total = λ_data · L_data + λ_phys · L_phys + λ_mono · L_mono

    L_data  = MSE(ŷ, y)                     — data fidelity
    L_phys  = MSE(SOH_pred, SOH_phys_target) — physics adherence
    L_mono  = mean(ReLU(SOH_i+1 − SOH_i))   — monotonicity (sorted by cycle)

The physics regularisation signal (SOH_phys) is passed as an additional
target alongside the ground truth, matching the dashed orange arrow in Fig 12.1.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ══════════════════════════════════════════════════════════════════════════════
#  Building blocks
# ══════════════════════════════════════════════════════════════════════════════

def _block(in_f: int, out_f: int, dropout: float) -> nn.Sequential:
    """Linear → BN → ReLU → Dropout"""
    return nn.Sequential(
        nn.Linear(in_f, out_f, bias=False),
        nn.BatchNorm1d(out_f),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Physics-Informed Multi-Task ANN
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsInformedMultiTaskANN(nn.Module):
    """
    Shared-trunk, dual-head ANN optimised with a physics-informed loss.

    Shared trunk → SOH head (linear output)
                 → RUL head (linear output)
    Concatenated output shape: (N, 2)  →  [SOH_pred, RUL_pred]
    """

    def __init__(self,
                 input_dim:    int,
                 trunk_units:  list  = (256, 128, 64),
                 head_units:   list  = (32, 16),
                 dropout_rate: float = 0.20):
        super().__init__()
        self.input_dim = input_dim

        # ── Shared trunk ───────────────────────────────────────────
        trunk, prev = [], input_dim
        for u in trunk_units:
            trunk.append(_block(prev, u, dropout_rate))
            prev = u
        self.trunk = nn.Sequential(*trunk)

        # ── SOH head ───────────────────────────────────────────────
        soh, rul = [], []
        p = prev
        for u in head_units:
            soh.append(_block(p, u, 0.0))
            rul.append(_block(p, u, 0.0))
            p = u
        self.soh_head = nn.Sequential(*soh, nn.Linear(p, 1))
        self.rul_head = nn.Sequential(*rul, nn.Linear(p, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.trunk(x)
        soh_p  = self.soh_head(shared)      # (N, 1)
        rul_p  = self.rul_head(shared)      # (N, 1)
        return torch.cat([soh_p, rul_p], dim=1)   # (N, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  Physics-Informed Loss Function
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsInformedLoss(nn.Module):
    """
    Combined loss for Physics-Informed ANN training.

    L_total = λ_data · MSE(pred, true)
            + λ_phys · MSE(SOH_pred, SOH_phys_target)   ← physics signal
            + λ_mono · monotonicity_penalty(SOH_pred)    ← physics constraint

    Args:
        lambda_data  : Weight for standard data-fidelity loss (default 1.0)
        lambda_phys  : Weight for physics adherence loss      (default 0.3)
        lambda_mono  : Weight for monotonicity constraint      (default 0.1)
    """

    def __init__(self,
                 lambda_data: float = 1.0,
                 lambda_phys: float = 0.3,
                 lambda_mono: float = 0.1):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_phys = lambda_phys
        self.lambda_mono = lambda_mono

    def forward(self,
                pred:        torch.Tensor,   # (N, 2)
                target:      torch.Tensor,   # (N, 2)  [SOH_true, RUL_true]
                phys_target: torch.Tensor    # (N, ?)  [SOH_phys_scaled, ...]
                ) -> tuple:
        """
        Returns (total_loss, data_loss, phys_loss, mono_loss) — all scalar tensors.
        """
        # ── Data loss ──────────────────────────────────────────────
        L_data = F.mse_loss(pred, target)

        # ── Physics adherence loss (SOH only) ─────────────────────
        # phys_target[:, 0] = SOH_phys (scaled), same scale as pred[:, 0]
        if phys_target.shape[1] > 0:
            L_phys = F.mse_loss(pred[:, 0], phys_target[:, 0])
        else:
            L_phys = torch.tensor(0.0)

        # ── Monotonicity penalty (SOH_pred should not increase with cycle) ─
        # Operates on the batch: penalise positive differences in SOH
        soh_pred = pred[:, 0]
        if len(soh_pred) > 1:
            diffs  = soh_pred[1:] - soh_pred[:-1]          # SOH_{i+1} - SOH_i
            L_mono = F.relu(diffs).mean()                    # penalise increases
        else:
            L_mono = torch.tensor(0.0)

        L_total = (self.lambda_data * L_data
                   + self.lambda_phys * L_phys
                   + self.lambda_mono * L_mono)

        return L_total, L_data, L_phys, L_mono


# ══════════════════════════════════════════════════════════════════════════════
#  High-level wrapper
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsInformedBatteryANN:
    """
    Wrapper that bundles the model, physics-informed loss, training loop,
    early stopping, LR scheduling and checkpointing.
    """

    def __init__(self,
                 input_dim:    int,
                 trunk_units:  list  = None,
                 head_units:   list  = None,
                 dropout_rate: float = 0.20,
                 weight_decay: float = 1e-4,
                 learning_rate: float = 1e-3,
                 lambda_data:  float = 1.0,
                 lambda_phys:  float = 0.30,
                 lambda_mono:  float = 0.10):
        self.input_dim     = input_dim
        self.trunk_units   = trunk_units  or [256, 128, 64]
        self.head_units    = head_units   or [32, 16]
        self.dropout_rate  = dropout_rate
        self.weight_decay  = weight_decay
        self.learning_rate = learning_rate
        self.lambda_data   = lambda_data
        self.lambda_phys   = lambda_phys
        self.lambda_mono   = lambda_mono
        self.device        = torch.device('cpu')
        self.history       = {k: [] for k in
                              ('train_loss', 'val_loss', 'train_data', 'train_phys',
                               'train_mono', 'val_data', 'val_phys')}
        self.best_val_loss = math.inf

        self.model = PhysicsInformedMultiTaskANN(
            input_dim    = input_dim,
            trunk_units  = self.trunk_units,
            head_units   = self.head_units,
            dropout_rate = self.dropout_rate,
        ).to(self.device)

        params = sum(p.numel() for p in self.model.parameters())
        print(f"\n[PhysicsInformedANN] Built | input={input_dim}d | "
              f"trunk={self.trunk_units} | heads={self.head_units} | "
              f"params={params:,}")
        print(f"  Loss weights: λ_data={lambda_data}  "
              f"λ_phys={lambda_phys}  λ_mono={lambda_mono}")

    @staticmethod
    def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32)

    def fit(self,
            X_train:       np.ndarray,
            y_train:       np.ndarray,
            y_phys_train:  np.ndarray,
            X_val:         np.ndarray,
            y_val:         np.ndarray,
            y_phys_val:    np.ndarray,
            epochs:        int   = 300,
            batch_size:    int   = 32,
            patience:      int   = 30,
            checkpoint_dir: str  = None) -> dict:
        """
        Train the Physics-Informed ANN.

        Args:
            y_phys_train : (N_train, ≥1) physics targets — SOH_phys_scaled, ...
            y_phys_val   : (N_val,   ≥1) physics targets for val monitoring
        """
        criterion = PhysicsInformedLoss(self.lambda_data,
                                         self.lambda_phys,
                                         self.lambda_mono)
        optimiser = torch.optim.Adam(self.model.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode='min', factor=0.5, patience=12, min_lr=1e-7
        )

        train_ds = TensorDataset(self._t(X_train), self._t(y_train), self._t(y_phys_train))
        val_ds   = TensorDataset(self._t(X_val),   self._t(y_val),   self._t(y_phys_val))
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

        best_state = None
        no_improve = 0
        print(f"\n[PhysicsInformedANN] Training: epochs={epochs} "
              f"batch={batch_size} patience={patience}")

        for epoch in range(1, epochs + 1):
            # ── Train ───────────────────────────────────────────────
            self.model.train()
            tl = td = tp = tm = 0.0
            for Xb, yb, pb in train_dl:
                optimiser.zero_grad()
                pred = self.model(Xb.to(self.device))
                loss, ld, lp, lm = criterion(pred, yb.to(self.device), pb.to(self.device))
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimiser.step()
                n = len(Xb)
                tl += loss.item() * n
                td += ld.item()   * n
                tp += lp.item()   * n
                tm += lm.item()   * n
            N = len(train_ds)
            tl /= N; td /= N; tp /= N; tm /= N

            # ── Validate ────────────────────────────────────────────
            self.model.eval()
            vl = vd = vp = 0.0
            with torch.no_grad():
                for Xb, yb, pb in val_dl:
                    pred = self.model(Xb.to(self.device))
                    loss, ld, lp, _ = criterion(pred, yb.to(self.device), pb.to(self.device))
                    n = len(Xb)
                    vl += loss.item() * n
                    vd += ld.item()   * n
                    vp += lp.item()   * n
            Nv = len(val_ds)
            vl /= Nv; vd /= Nv; vp /= Nv

            scheduler.step(vl)
            self.history['train_loss'].append(tl)
            self.history['val_loss'].append(vl)
            self.history['train_data'].append(td)
            self.history['train_phys'].append(tp)
            self.history['train_mono'].append(tm)
            self.history['val_data'].append(vd)
            self.history['val_phys'].append(vp)

            if epoch % 20 == 0 or epoch == 1:
                lr_now = optimiser.param_groups[0]['lr']
                print(f"  Epoch {epoch:4d}/{epochs} | "
                      f"train: {tl:.5f} (data={td:.4f} phys={tp:.4f} mono={tm:.4f}) | "
                      f"val: {vl:.5f} | lr: {lr_now:.2e}")

            if vl < self.best_val_loss - 1e-7:
                self.best_val_loss = vl
                best_state         = {k: v.clone() for k, v in
                                      self.model.state_dict().items()}
                no_improve         = 0
                if checkpoint_dir:
                    self._save_state(best_state, checkpoint_dir, 'best')
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\n  Early stopping at epoch {epoch} "
                          f"(no improvement for {patience} epochs)")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
            print(f"  Restored best weights (val_loss={self.best_val_loss:.5f})")

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model(self._t(X).to(self.device)).cpu().numpy()

    def _save_state(self, state: dict, directory: str, tag: str) -> None:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f'hybrid_ann_{tag}.pt')
        torch.save({'model_state': state,
                    'input_dim': self.input_dim,
                    'trunk_units': self.trunk_units,
                    'head_units': self.head_units}, path)

    def save(self, directory: str) -> None:
        self._save_state(self.model.state_dict(), directory, 'final')
        print(f"  [PhysicsInformedANN] Model saved → {directory}")

    def load(self, directory: str, tag: str = 'final') -> 'PhysicsInformedBatteryANN':
        path = os.path.join(directory, f'hybrid_ann_{tag}.pt')
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt['model_state'])
        return self
