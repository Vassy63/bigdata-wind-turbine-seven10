"""
Phase 2: Random Forest Regression -- Huan luyen va danh gia
===========================================================
Nguoi phu trach: Nguyen Quoc An (scikit-learn)
Muc tieu: Du doan power_output tu du lieu turbine gio.

Cach chay (tu bat ky thu muc nao):
    python source/phase2_random_forest.py

Duong dan duoc tinh tuong doi tu vi tri file nay -> thu muc goc du an.
"""

import os
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. DUONG DAN
# ---------------------------------------------------------------------------
THIS_DIR   = Path(__file__).resolve().parent
ROOT       = THIS_DIR.parent
DATA_DIR   = ROOT / "data"
PICS_DIR   = ROOT / "pics" / "phase2"   # anh Phase 2 luu rieng, khong lan vao phase1
SOURCE_DIR = THIS_DIR
OUT_DIR    = ROOT / "output_phase2"

OUT_DIR.mkdir(exist_ok=True)
PICS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. DAC TRUNG 
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "wind_speed",
    "wind_direction",
    "air_density",
    "temperature",
    "blade_angle",
    "rotor_speed",
    "generator_efficiency",
    "altitude",
    "humidity",
]

CATEGORICAL_FEATURES = [
    "region",
    "turbine_id",
    "maintenance_status",
    "technician_note",
]

TARGET = "power_output"
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# 2. TAI DU LIEU
# ---------------------------------------------------------------------------
print("=" * 65)
print("PHASE 2 -- Random Forest Regression (scikit-learn)")
print("=" * 65)

print("\n[1/6] Dang tai du lieu ...")
t0 = time.time()

train = pd.read_csv(DATA_DIR / "train.csv")
val   = pd.read_csv(DATA_DIR / "validation.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")

train_ids = pd.read_csv(DATA_DIR / "train_record_ids.csv")
val_ids   = pd.read_csv(DATA_DIR / "validation_record_ids.csv")
test_ids  = pd.read_csv(DATA_DIR / "test_record_ids.csv")

print(f"  train     : {train.shape[0]:>7,} dong x {train.shape[1]} cot")
print(f"  validation: {val.shape[0]:>7,} dong x {val.shape[1]} cot")
print(f"  test      : {test.shape[0]:>7,} dong x {test.shape[1]} cot")

for name, df in [("train", train), ("validation", val), ("test", test)]:
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"  [WARN] {name} co {missing} gia tri thieu!")
    if TARGET not in df.columns:
        sys.exit(f"[ERROR] Khong tim thay cot '{TARGET}' trong {name}.csv")

print(f"  Tai xong trong {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# 3. TACH X / y
# ---------------------------------------------------------------------------
X_train = train[ALL_FEATURES]
y_train = train[TARGET]

X_val = val[ALL_FEATURES]
y_val = val[TARGET]

X_test = test[ALL_FEATURES]
y_test = test[TARGET]

# ---------------------------------------------------------------------------
# 4. TIEN XU LY
# ---------------------------------------------------------------------------
print("\n[2/6] Chuan bi preprocessor ...")
PKL_PATH = ROOT / "output_phase1" / "preprocessor.pkl"
PKL_NEW  = OUT_DIR / "preprocessor_phase2.pkl"

preprocessor_ok = False
if PKL_PATH.exists():
    try:
        preprocessor = joblib.load(PKL_PATH)
        preprocessor.transform(X_train.head(5))
        print(f"  OK: Tai preprocessor.pkl thanh cong")
        preprocessor_ok = True
    except Exception as e:
        print(f"  FAIL: preprocessor.pkl khong tuong thich: {e}")
        print("    -> Tai tao cung cau hinh, chi fit tren train.")

if not preprocessor_ok:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, PKL_NEW)
    print(f"  Da tao lai va luu tai: {PKL_NEW}")

print("  Transforming ...")
t0 = time.time()
X_train_proc = preprocessor.transform(X_train)
X_val_proc   = preprocessor.transform(X_val)
X_test_proc  = preprocessor.transform(X_test)
print(f"  Xong trong {time.time()-t0:.1f}s | Shape sau transform: {X_train_proc.shape}")

if hasattr(X_train_proc, "toarray"):
    X_train_proc = X_train_proc.toarray()
    X_val_proc   = X_val_proc.toarray()
    X_test_proc  = X_test_proc.toarray()

# ---------------------------------------------------------------------------
# 5. TIM KIEM THAM SO -- chon theo RMSE validation
# ---------------------------------------------------------------------------
print("\n[3/6] Tim kiem tham so (8 configs, chon theo RMSE val) ...")

PARAM_GRID = [
    (100,  None, 1),
    (100,  20,   1),
    (100,  30,   2),
    (200,  None, 1),
    (200,  20,   2),
    (200,  30,   4),
    (300,  None, 2),
    (300,  30,   2),
]

results = []
best_rmse   = float("inf")
best_params = None
best_model  = None

for n_est, max_d, min_leaf in PARAM_GRID:
    label = f"n={n_est}, depth={str(max_d):>4s}, leaf={min_leaf}"
    print(f"  Thu {label} ...", end=" ", flush=True)

    t_start = time.time()
    rf = RandomForestRegressor(
        n_estimators=n_est,
        max_depth=max_d,
        min_samples_leaf=min_leaf,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train_proc, y_train)
    t_train = time.time() - t_start

    t_pred = time.time()
    y_val_pred = rf.predict(X_val_proc)
    t_pred = time.time() - t_pred

    rmse_val = root_mean_squared_error(y_val, y_val_pred)
    mae_val  = mean_absolute_error(y_val, y_val_pred)
    r2_val   = r2_score(y_val, y_val_pred)

    print(f"RMSE_val={rmse_val:,.2f}  MAE_val={mae_val:,.2f}  R2={r2_val:.4f}"
          f"  (train {t_train:.0f}s)")

    row = {
        "n_estimators": n_est,
        "max_depth": max_d,
        "min_samples_leaf": min_leaf,
        "rmse_val": rmse_val,
        "mae_val": mae_val,
        "r2_val": r2_val,
        "train_time_s": round(t_train, 2),
        "pred_time_s": round(t_pred, 4),
    }
    results.append(row)

    if rmse_val < best_rmse:
        best_rmse   = rmse_val
        best_params = (n_est, max_d, min_leaf)
        best_model  = rf

search_df = pd.DataFrame(results).sort_values("rmse_val")
search_df.to_csv(OUT_DIR / "param_search_results.csv", index=False)
print(f"\n  -> Config tot nhat: n_estimators={best_params[0]}, "
      f"max_depth={best_params[1]}, min_samples_leaf={best_params[2]}")
print(f"     RMSE validation = {best_rmse:,.4f}")

# ---------------------------------------------------------------------------
# 6. DANH GIA TREN TEST (chi 1 lan)
# ---------------------------------------------------------------------------
print("\n[4/6] Danh gia tren tat ca cac tap ...")


def evaluate(model, X, y_true, label):
    t0 = time.time()
    y_pred = model.predict(X)
    elapsed = time.time() - t0
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {label:>12s} | MAE={mae:>10,.4f}  RMSE={rmse:>10,.4f}  R2={r2:.6f}  ({elapsed:.2f}s)")
    return y_pred, mae, rmse, r2, elapsed


y_train_pred, mae_tr, rmse_tr, r2_tr, dt_tr = evaluate(
    best_model, X_train_proc, y_train, "Train")
y_val_pred,   mae_v,  rmse_v,  r2_v,  dt_v  = evaluate(
    best_model, X_val_proc,   y_val,   "Validation")
y_test_pred,  mae_ts, rmse_ts, r2_ts, dt_ts = evaluate(
    best_model, X_test_proc,  y_test,  "Test")

# ---------------------------------------------------------------------------
# 7. LUU KET QUA
# ---------------------------------------------------------------------------
print("\n[5/6] Luu ket qua ...")

model_path = SOURCE_DIR / "rf_best_model.pkl"
joblib.dump(best_model, model_path)
print(f"  Mo hinh -> {model_path}")

metrics_df = pd.DataFrame([
    {"split": "train",      "MAE": mae_tr, "RMSE": rmse_tr, "R2": r2_tr,
     "pred_time_s": dt_tr,  "n_estimators": best_params[0],
     "max_depth": best_params[1], "min_samples_leaf": best_params[2]},
    {"split": "validation", "MAE": mae_v,  "RMSE": rmse_v,  "R2": r2_v,
     "pred_time_s": dt_v,   "n_estimators": best_params[0],
     "max_depth": best_params[1], "min_samples_leaf": best_params[2]},
    {"split": "test",       "MAE": mae_ts, "RMSE": rmse_ts, "R2": r2_ts,
     "pred_time_s": dt_ts,  "n_estimators": best_params[0],
     "max_depth": best_params[1], "min_samples_leaf": best_params[2]},
])
metrics_df.to_csv(OUT_DIR / "metrics.csv", index=False)
print(f"  Chi so -> {OUT_DIR / 'metrics.csv'}")

test_preds = pd.concat([
    test_ids.reset_index(drop=True),
    pd.Series(y_test.values, name="actual_power_output"),
    pd.Series(y_test_pred,   name="predicted_power_output"),
], axis=1)
test_preds.to_csv(OUT_DIR / "test_predictions.csv", index=False)
print(f"  Du doan test -> {OUT_DIR / 'test_predictions.csv'}")

# ---------------------------------------------------------------------------
# 8. BIEU DO
# ---------------------------------------------------------------------------
print("\n[6/6] Ve bieu do ...")

# 8a. Param search RMSE
fig, ax = plt.subplots(figsize=(10, 5))
x_labels = [
    f"n={int(r['n_estimators'])}\nd={str(r['max_depth'])}\nl={int(r['min_samples_leaf'])}"
    for _, r in search_df.iterrows()
]
bar_colors = ["#4C72B0"] * len(search_df)
best_pos = list(search_df["rmse_val"]).index(min(search_df["rmse_val"]))
bar_colors[best_pos] = "#2ca02c"
bars = ax.bar(range(len(search_df)), search_df["rmse_val"],
              color=bar_colors, edgecolor="white", width=0.6)
ax.set_xticks(range(len(search_df)))
ax.set_xticklabels(x_labels, fontsize=8)
ax.set_ylabel("RMSE (Validation)")
ax.set_title("Random Forest -- Validation RMSE by Hyperparameter Config",
             fontsize=12, fontweight="bold")
ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#2ca02c", label="Best config"),
                   Patch(color="#4C72B0", label="Other")], loc="upper right")
plt.tight_layout()
p1 = PICS_DIR / "rf_param_search.png"
fig.savefig(p1, dpi=150)
plt.close(fig)
print(f"  -> {p1}")

# 8b. Actual vs Predicted + residuals
rng = np.random.default_rng(42)
idx = rng.choice(len(y_test), size=min(3000, len(y_test)), replace=False)
actual_s  = np.array(y_test)[idx]
pred_s    = y_test_pred[idx]
residuals = actual_s - pred_s

fig = plt.figure(figsize=(14, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig)

ax1 = fig.add_subplot(gs[0])
ax1.scatter(actual_s, pred_s, alpha=0.3, s=6, color="#4C72B0")
lim = [min(actual_s.min(), pred_s.min()), max(actual_s.max(), pred_s.max())]
ax1.plot(lim, lim, "r--", linewidth=1.5)
ax1.set_xlabel("Actual")
ax1.set_ylabel("Predicted")
ax1.set_title("Actual vs Predicted\n(Test sample n=3000)")

ax2 = fig.add_subplot(gs[1])
ax2.hist(residuals, bins=60, color="#4C72B0", edgecolor="none", alpha=0.8)
ax2.axvline(0, color="red", linewidth=1.5, linestyle="--")
ax2.set_xlabel("Residual (Actual - Predicted)")
ax2.set_ylabel("Count")
ax2.set_title("Residual Distribution")

ax3 = fig.add_subplot(gs[2])
ax3.scatter(actual_s, residuals, alpha=0.3, s=6, color="#dd8452")
ax3.axhline(0, color="red", linewidth=1.5, linestyle="--")
ax3.set_xlabel("Actual")
ax3.set_ylabel("Residual")
ax3.set_title("Residual vs Actual")

fig.suptitle(
    f"Random Forest -- Test  |  RMSE={rmse_ts:,.2f}  MAE={mae_ts:,.2f}  R2={r2_ts:.4f}",
    fontsize=12, fontweight="bold", y=1.01
)
plt.tight_layout()
p2 = PICS_DIR / "rf_test_evaluation.png"
fig.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  -> {p2}")

# 8c. Feature importance top-20
feature_names = preprocessor.get_feature_names_out()
importances   = best_model.feature_importances_
fi_df = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .head(20)
)
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="#4C72B0")
ax.set_xlabel("Feature Importance (Mean Decrease Impurity)")
ax.set_title("Top-20 Feature Importances -- Random Forest",
             fontsize=12, fontweight="bold")
plt.tight_layout()
p3 = PICS_DIR / "rf_feature_importance.png"
fig.savefig(p3, dpi=150)
plt.close(fig)
print(f"  -> {p3}")

fi_df.to_csv(OUT_DIR / "feature_importance.csv", index=False)

# ---------------------------------------------------------------------------
# 9. TOM TAT
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print("  KET QUA PHASE 2 -- RANDOM FOREST REGRESSION")
print("=" * 65)
print(f"  Best config : n_estimators={best_params[0]}, "
      f"max_depth={best_params[1]}, min_samples_leaf={best_params[2]}")
print()
print(f"  {'Split':<12} {'MAE':>12} {'RMSE':>12} {'R2':>10}")
print(f"  {'-'*48}")
print(f"  {'Train':<12} {mae_tr:>12,.4f} {rmse_tr:>12,.4f} {r2_tr:>10.6f}")
print(f"  {'Validation':<12} {mae_v:>12,.4f} {rmse_v:>12,.4f} {r2_v:>10.6f}")
print(f"  {'Test':<12} {mae_ts:>12,.4f} {rmse_ts:>12,.4f} {r2_ts:>10.6f}")
print()
print("  Files da tao:")
print("    source/rf_best_model.pkl")
print("    output_phase2/metrics.csv")
print("    output_phase2/param_search_results.csv")
print("    output_phase2/test_predictions.csv")
print("    output_phase2/feature_importance.csv")
print("    pics/rf_param_search.png")
print("    pics/rf_test_evaluation.png")
print("    pics/rf_feature_importance.png")
print("=" * 65)
