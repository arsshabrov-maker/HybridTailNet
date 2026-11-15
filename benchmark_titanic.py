import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

from model import HybridTailNet, StandardMLP


def load_dataset():
  """
    Loads Spaceship Titanic if present locally/on Kaggle;
    otherwise automatically fetches the open Titanic dataset via OpenML
    so the benchmark runs out-of-the-box anywhere.
    """
  local_candidates = [
      'train.csv',
      'data/train.csv',
      '/kaggle/input/spaceship-titanic/train.csv',
      '/kaggle/input/competitions/spaceship-titanic/train.csv',
  ]
  for path in local_candidates:
    if os.path.exists(path):
      print(f'Loading dataset from: {path}')
      df = pd.read_csv(path)
      if 'Transported' in df.columns:
        num_cols = [
            'Age',
            'RoomService',
            'FoodCourt',
            'ShoppingMall',
            'Spa',
            'VRDeck',
        ]
        cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
        target_col = 'Transported'
        return df, num_cols, cat_cols, target_col

  print('Fetching open Titanic benchmark dataset via OpenML...')
  titanic = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
  df = titanic.frame.copy()
  num_cols = ['pclass', 'age', 'sibsp', 'parch', 'fare']
  cat_cols = ['sex', 'embarked']
  df['survived'] = df['survived'].astype(int)
  return df, num_cols, cat_cols, 'survived'


def preprocess(df, num_cols, cat_cols, target_col):
  # Presence mask for the dynamic tail (1.0 = observed, 0.0 = missing NaN)
  presence_mask = (~df[num_cols + cat_cols].isna()).astype(np.float32).values

  df_clean = df[num_cols + cat_cols].copy()
  for col in num_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
  for col in cat_cols:
    df_clean[col] = df_clean[col].astype(str).fillna('Missing')

  encoded = pd.get_dummies(df_clean, drop_first=True).astype(np.float32)
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(encoded)
  y = df[target_col].astype(np.float32).values
  return X_scaled, presence_mask, y


def train_and_eval(
    model, X_tr, M_tr, y_tr, X_va, M_va, y_va, is_hybrid=False, seed=42, epochs=350
):
  torch.manual_seed(seed)
  for m in model.modules():
    if isinstance(m, nn.Linear):
      nn.init.xavier_uniform_(m.weight)
      nn.init.zeros_(m.bias)

  optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-3)
  criterion = nn.BCEWithLogitsLoss()

  best_loss = float('inf')
  best_state = None

  for _ in range(epochs):
    model.train()
    optimizer.zero_grad()
    logits = model(X_tr, M_tr) if is_hybrid else model(X_tr)
    loss = criterion(logits, y_tr)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
      val_logits = model(X_va, M_va) if is_hybrid else model(X_va)
      val_loss = criterion(val_logits, y_va).item()
      if val_loss < best_loss:
        best_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

  model.load_state_dict(best_state)
  model.eval()
  with torch.no_grad():
    val_probs = (
        torch.sigmoid(model(X_va, M_va) if is_hybrid else model(X_va))
        .numpy()
        .ravel()
    )
  return val_probs


def main():
  df, num_cols, cat_cols, target_col = load_dataset()
  X_scaled, presence_mask, y = preprocess(df, num_cols, cat_cols, target_col)

  X_tr, X_va, M_tr, M_va, y_tr, y_va = train_test_split(
      X_scaled, presence_mask, y, test_size=0.2, random_state=42, stratify=y
  )

  X_tr_t = torch.tensor(X_tr)
  M_tr_t = torch.tensor(M_tr)
  y_tr_t = torch.tensor(y_tr).unsqueeze(1)
  X_va_t = torch.tensor(X_va)
  M_va_t = torch.tensor(M_va)
  y_va_t = torch.tensor(y_va).unsqueeze(1)

  in_dim = X_tr.shape[1]
  seeds = [100 + i * 37 for i in range(10)]

  h_accs, h_lls = [], []
  m_accs, m_lls = [], []

  print('=' * 78)
  print('10-SEED HEAD-TO-HEAD BENCHMARK: HybridTailNet vs StandardMLP')
  print('=' * 78)

  for idx, s in enumerate(seeds):
    hybrid_model = HybridTailNet(
        in_dim=in_dim,
        hidden_dims=(24, 24, 16),
        non_neuron_prob=0.35,
        max_len=10,
        mini_hidden=12,
        seed=42,
    )
    mlp_model = StandardMLP(in_dim=in_dim, hidden_dims=(28, 28, 20))

    if idx == 0:
      h_params = sum(p.numel() for p in hybrid_model.parameters())
      m_params = sum(p.numel() for p in mlp_model.parameters())

    h_vp = train_and_eval(
        hybrid_model, X_tr_t, M_tr_t, y_tr_t, X_va_t, M_va_t, y_va_t,
        is_hybrid=True, seed=s
    )
    m_vp = train_and_eval(
        mlp_model, X_tr_t, M_tr_t, y_tr_t, X_va_t, M_va_t, y_va_t,
        is_hybrid=False, seed=s
    )

    h_acc = accuracy_score(y_va, h_vp > 0.5) * 100
    h_ll = log_loss(y_va, h_vp)
    m_acc = accuracy_score(y_va, m_vp > 0.5) * 100
    m_ll = log_loss(y_va, m_vp)

    h_accs.append(h_acc)
    h_lls.append(h_ll)
    m_accs.append(m_acc)
    m_lls.append(m_ll)

    print(
        f'Seed {s:<4} | Hybrid Acc: {h_acc:.2f}% (LL: {h_ll:.4f})  vs  MLP Acc:'
        f' {m_acc:.2f}% (LL: {m_ll:.4f})'
    )

  print('=' * 78)
  print(
      f'HybridTailNet | Params: {h_params:<4} | Mean Acc: {np.mean(h_accs):.2f}%'
      f' ± {np.std(h_accs):.2f}% | Mean LogLoss: {np.mean(h_lls):.4f} ±'
      f' {np.std(h_lls):.4f}'
  )
  print(
      f'StandardMLP   | Params: {m_params:<4} | Mean Acc: {np.mean(m_accs):.2f}%'
      f' ± {np.std(m_accs):.2f}% | Mean LogLoss: {np.mean(m_lls):.4f} ±'
      f' {np.std(m_lls):.4f}'
  )
  print('=' * 78)


if __name__ == '__main__':
  main()
