# =============================================================================
# House Price Prediction using Neural Networks (PyTorch)
# Dataset: Bengaluru House Prices
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

print("=" * 60)
print("  House Price Prediction using Neural Networks (PyTorch)")
print("=" * 60)
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device    : {device}\n")

# =============================================================================
# 1. LOAD DATA
# =============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path   = os.path.join(script_dir, "bengaluru_house_prices.csv")

df = pd.read_csv(csv_path)
print(f"Loaded dataset  : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns         : {df.columns.tolist()}\n")

# =============================================================================
# 2. DATA CLEANING & FEATURE ENGINEERING
# =============================================================================

# --- Drop columns that are mostly missing or not useful ---
df.drop(columns=["society", "availability"], inplace=True)

# --- Extract number of bedrooms from 'size' (e.g. '2 BHK', '3 Bedroom') ---
def parse_bedrooms(s):
    try:
        return int(str(s).split()[0])
    except:
        return np.nan

df["bedrooms"] = df["size"].apply(parse_bedrooms)
df.drop(columns=["size"], inplace=True)

# --- Parse total_sqft: handle ranges like '1200-1500' by taking the mean ---
def parse_sqft(val):
    val = str(val).strip()
    if "-" in val:
        try:
            lo, hi = val.split("-")
            return (float(lo) + float(hi)) / 2
        except:
            return np.nan
    try:
        return float(val)
    except:
        return np.nan

df["total_sqft"] = df["total_sqft"].apply(parse_sqft)

# --- Drop rows with missing values ---
df.dropna(inplace=True)
print(f"After cleaning  : {df.shape[0]} rows remaining")

# --- Remove extreme outliers in price (top 1%) ---
upper = df["price"].quantile(0.99)
df = df[df["price"] <= upper]
print(f"After outlier   : {df.shape[0]} rows remaining\n")

# --- Encode categorical columns ---
# area_type → label encode (4 categories)
le_area = LabelEncoder()
df["area_type"] = le_area.fit_transform(df["area_type"])

# location → group rare locations (< 10 listings) as 'other', then label encode
location_counts = df["location"].value_counts()
df["location"] = df["location"].apply(
    lambda x: x if location_counts[x] >= 10 else "other"
)
le_loc = LabelEncoder()
df["location"] = le_loc.fit_transform(df["location"])

# --- price_per_sqft as a sanity feature ---
df["price_per_sqft"] = df["price"] * 100000 / df["total_sqft"]

# Keep only rows with reasonable price_per_sqft (remove obvious data errors)
df = df[(df["price_per_sqft"] >= 500) & (df["price_per_sqft"] <= 100000)]
df.drop(columns=["price_per_sqft"], inplace=True)

print(f"Final dataset   : {df.shape[0]} rows × {df.shape[1]} columns")
print(df.head(3).to_string())
print()

# =============================================================================
# 3. PREPARE FEATURES AND TARGET
# =============================================================================

feature_cols = ["area_type", "location", "total_sqft", "bath", "balcony", "bedrooms"]
target_col   = "price"

X = df[feature_cols].values.astype(np.float32)
y = df[target_col].values.astype(np.float32).reshape(-1, 1)

# Log-transform the target so the network doesn't chase large outlier values
y_log = np.log1p(y)

# Train / validation / test split  (70 / 15 / 15)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_log, test_size=0.30, random_state=42)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print(f"Train samples   : {len(X_train)}")
print(f"Val   samples   : {len(X_val)}")
print(f"Test  samples   : {len(X_test)}\n")

# --- Scale features ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# --- Convert to PyTorch tensors ---
def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32).to(device)

X_train_t = to_tensor(X_train);  y_train_t = to_tensor(y_train)
X_val_t   = to_tensor(X_val);    y_val_t   = to_tensor(y_val)
X_test_t  = to_tensor(X_test);   y_test_t  = to_tensor(y_test)

train_ds     = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

# =============================================================================
# 4. NEURAL NETWORK MODEL
# =============================================================================

class HousePriceNet(nn.Module):
    """
    Feed-forward network with 4 hidden layers and batch normalisation.
    Input  : 6 features
    Output : 1 (log price in lakhs)
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


model     = HousePriceNet(input_dim=X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable parameters: {total_params:,}\n")

# =============================================================================
# 5. TRAINING LOOP
# =============================================================================

EPOCHS      = 200
best_val    = float("inf")
best_state  = None
train_losses, val_losses = [], []

print(f"{'Epoch':>6}  {'Train RMSE':>12}  {'Val RMSE':>10}  {'LR':>10}")
print("-" * 48)

for epoch in range(1, EPOCHS + 1):
    # -- train --
    model.train()
    batch_losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    train_loss = np.mean(batch_losses)

    # -- validate --
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t).item()

    scheduler.step(val_loss)

    train_losses.append(np.sqrt(train_loss))
    val_losses.append(np.sqrt(val_loss))

    # save best model
    if val_loss < best_val:
        best_val   = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 20 == 0 or epoch == 1:
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>6}  {np.sqrt(train_loss):>12.4f}  {np.sqrt(val_loss):>10.4f}  {lr_now:>10.6f}")

print("\nTraining complete!")
model.load_state_dict(best_state)
print(f"Best val RMSE (log scale): {np.sqrt(best_val):.4f}\n")

# =============================================================================
# 6. EVALUATION ON TEST SET
# =============================================================================

model.eval()
with torch.no_grad():
    test_pred_log = model(X_test_t).cpu().numpy()
    test_true_log = y_test_t.cpu().numpy()

# Reverse log transform
test_pred = np.expm1(test_pred_log)
test_true = np.expm1(test_true_log)

mae  = np.mean(np.abs(test_pred - test_true))
rmse = np.sqrt(np.mean((test_pred - test_true) ** 2))
ss_res = np.sum((test_true - test_pred) ** 2)
ss_tot = np.sum((test_true - np.mean(test_true)) ** 2)
r2   = 1 - ss_res / ss_tot

print("=" * 40)
print("  TEST SET RESULTS (price in lakhs)")
print("=" * 40)
print(f"  MAE  : ₹ {mae:.2f} L")
print(f"  RMSE : ₹ {rmse:.2f} L")
print(f"  R²   : {r2:.4f}")
print("=" * 40)

# =============================================================================
# 7. PLOTS
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("House Price Prediction — Neural Network Results", fontsize=14, fontweight="bold")

# --- Plot 1: Training & Validation loss curve ---
ax1 = axes[0]
ax1.plot(train_losses, label="Train RMSE", color="steelblue")
ax1.plot(val_losses,   label="Val RMSE",   color="tomato")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("RMSE (log scale)")
ax1.set_title("Training vs Validation Loss")
ax1.legend()
ax1.grid(alpha=0.3)

# --- Plot 2: Actual vs Predicted scatter ---
ax2 = axes[1]
ax2.scatter(test_true, test_pred, alpha=0.3, s=10, color="steelblue")
min_val = min(test_true.min(), test_pred.min())
max_val = max(test_true.max(), test_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect prediction")
ax2.set_xlabel("Actual Price (Lakhs)")
ax2.set_ylabel("Predicted Price (Lakhs)")
ax2.set_title(f"Actual vs Predicted  (R² = {r2:.3f})")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(script_dir, "results.png")
plt.savefig(plot_path, dpi=150)
print(f"\nPlot saved to: {plot_path}")
plt.show()
