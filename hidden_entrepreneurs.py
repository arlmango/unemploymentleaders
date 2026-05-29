"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         MASTERCARD DATA QUEST — Hidden Entrepreneurs Detection              ║
║   One-Class подход: IsolationForest + LightGBM scoring pipeline            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Pipeline:
  1. Load & Merge
  2. Feature Engineering (единые признаки для bus и con)
  3. One-Class обучение на business (IsolationForest)
  4. Scoring consumer-карт по схожести с business-паттерном
  5. LightGBM для калибровки вероятностей
  6. Metrics, SHAP, графики
"""

import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lightgbm as lgb

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)

DATA_DIR = Path(".")
BUSINESS_FILE  = DATA_DIR / "business_cards_MDQ.parquet"
CONSUMER_FILE  = DATA_DIR / "consumer_cards_MDQ.parquet"
MERCHANTS_FILE = DATA_DIR / "merchants_reference.parquet"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

HIDDEN_THRESHOLD = 0.85
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ═══════════════════════════════════════════════════════════════════════════════
#  1. LOAD
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("  MASTERCARD DATA QUEST — Hidden Entrepreneurs Detection")
print("=" * 72)
print(f"\n[1/6] Загрузка parquet-файлов …")

bus = pd.read_parquet(BUSINESS_FILE)
con = pd.read_parquet(CONSUMER_FILE)
mer = pd.read_parquet(MERCHANTS_FILE)

print(f"  business_cards : {bus.shape[0]:>8_} rows, {bus.shape[1]} cols")
print(f"  consumer_cards : {con.shape[0]:>8_} rows, {con.shape[1]} cols")
print(f"  merchants_ref  : {mer.shape[0]:>8_} rows, {mer.shape[1]} cols")

# ─── Ключ слияния ─────────────────────────────────────────────────────────────
MERGE_KEY = None
for candidate in ['merchant_id', 'merchant', 'merchant_number', 'mid']:
    if candidate in bus.columns and candidate in mer.columns:
        MERGE_KEY = candidate
        break
if MERGE_KEY is None:
    intersect = list(set(bus.columns) & set(mer.columns))
    MERGE_KEY = intersect[0] if intersect else "merchant"
print(f"  ключ слияния: '{MERGE_KEY}'")

# ═══════════════════════════════════════════════════════════════════════════════
#  2. MERGE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[2/6] Слияние с merchants_reference …")

bus = bus.merge(mer, on=MERGE_KEY, how="left", suffixes=("_bus", "_mer"))
con = con.merge(mer, on=MERGE_KEY, how="left", suffixes=("_con", "_mer"))

print(f"  business после merge: {bus.shape[0]:>8_} rows, {bus.shape[1]} cols")
print(f"  consumer после merge: {con.shape[0]:>8_} rows, {con.shape[1]} cols")

# ═══════════════════════════════════════════════════════════════════════════════
#  3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[3/6] Feature engineering …")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Единые агрегированные признаки по card_number."""
    df = df.dropna(subset=["card_number"]).copy()
    df["card_number"] = df["card_number"].astype(int)

    exclude = {"card_number", "card_identifier", "transaction_id", "txn_id", MERGE_KEY}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    if date_cols:
        dc = date_cols[0]
        df[dc] = pd.to_datetime(df[dc], errors="coerce")
        df["hour"] = df[dc].dt.hour
        df["dayofweek"] = df[dc].dt.dayofweek
        df["month"] = df[dc].dt.month
        df["quarter"] = df[dc].dt.quarter
        numeric_cols.extend(["hour", "dayofweek", "month", "quarter"])

    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = ["mean", "std", "min", "max", "count"]

    grp = df.groupby("card_number")[numeric_cols].agg(agg_dict)
    grp.columns = [f"{col}_{agg}" for col, agg in grp.columns]
    grp = grp.reset_index()

    if MERGE_KEY in df.columns:
        nunique = df.groupby("card_number")[MERGE_KEY].nunique().reset_index()
        nunique.columns = ["card_number", "nunique_merchants"]
        grp = grp.merge(nunique, on="card_number", how="left")

    amt_col = [c for c in numeric_cols if "amount" in c.lower() or "sum" in c.lower() or "amt" in c.lower()]
    if amt_col:
        amt = amt_col[0]
        total_amt = df.groupby("card_number")[amt].sum().reset_index()
        total_amt.columns = ["card_number", "total_amount"]
        grp = grp.merge(total_amt, on="card_number", how="left")
        cnt_col = f"{amt}_count"
        if cnt_col in grp.columns:
            grp["avg_check"] = grp["total_amount"] / grp[cnt_col].replace(0, np.nan)
        grp["amt_to_mean_ratio"] = grp["total_amount"] / grp[f"{amt}_mean"].replace(0, np.nan)

    return grp

bus_feat = build_features(bus)
con_feat = build_features(con)

print(f"  business features: {bus_feat.shape[1]-1} признаков, {bus_feat.shape[0]} card_number")
print(f"  consumer features: {con_feat.shape[1]-1} признаков, {con_feat.shape[0]} card_number")

common_feat_cols = [c for c in bus_feat.columns if c != "card_number" and c in con_feat.columns]
print(f"  общих признаков: {len(common_feat_cols)}")

# ═══════════════════════════════════════════════════════════════════════════════
#  4. ONE-CLASS LEARNING (IsolationForest на business-картах)
#  + LightGBM для вероятностей с синтетическими негативами
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[4/6] One-Class обучение + LightGBM …")

# --- IsolationForest на business-картах ---
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

X_bus = bus_feat[common_feat_cols].fillna(0).values
scaler = StandardScaler()
X_bus_scaled = scaler.fit_transform(X_bus)

iso_forest.fit(X_bus_scaled)
print(f"  IsolationForest обучен на {len(X_bus_scaled)} business-картах")

# --- Скоринг consumer-карт через IsolationForest ---
X_con = con_feat[common_feat_cols].fillna(0).values
X_con_scaled = scaler.transform(X_con)

# IsolationForest возвращает anomaly score (чем меньше, тем больше похоже на business)
iso_scores = iso_forest.score_samples(X_con_scaled)
# Нормализуем в 0-1 (чем ближе к 1, тем больше похоже на предпринимателя)
iso_probs = 1 - (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())

print(f"  Медиана score: {np.median(iso_scores):.4f}")
print(f"  Медиана prob : {np.median(iso_probs):.4f}")

# --- LightGBM для калибровки вероятностей ---
# Создаём синтетический датасет:
#   positive: business-карты (таргет=1)
#   negative: consumer-карты с низким iso_score (таргет=0)
neg_threshold = np.percentile(iso_probs, 20)  # bottom 20% - явно не предприниматели
con_neg_mask = iso_probs <= neg_threshold

X_train_pos = X_bus_scaled
y_train_pos = np.ones(len(X_train_pos))

X_train_neg = X_con_scaled[con_neg_mask]
y_train_neg = np.zeros(len(X_train_neg))

# Сбалансируем классы (возьмём равное количество негативов)
n_pos = len(X_train_pos)
np.random.seed(RANDOM_STATE)
neg_idx = np.random.choice(len(X_train_neg), size=min(n_pos, len(X_train_neg)), replace=False)
X_train_neg = X_train_neg[neg_idx]
y_train_neg = y_train_neg[neg_idx]

X_train = np.vstack([X_train_pos, X_train_neg])
y_train = np.hstack([y_train_pos, y_train_neg])

print(f"  LightGBM train shape: {X_train.shape}")
print(f"    Positive: {y_train.sum():.0f}, Negative: {(1-y_train).sum():.0f}")

# --- LightGBM ---
X_train_df = pd.DataFrame(X_train, columns=common_feat_cols)
y_train_arr = y_train

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

cv_scores = []
models = []

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_df, y_train_arr)):
    X_tr, X_vl = X_train_df.iloc[train_idx], X_train_df.iloc[val_idx]
    y_tr, y_vl = y_train_arr[train_idx], y_train_arr[val_idx]

    lgb_tr = lgb.Dataset(X_tr, y_tr)
    lgb_vl = lgb.Dataset(X_vl, y_vl, reference=lgb_tr)

    model = lgb.train(
        lgb_params,
        lgb_tr,
        num_boost_round=300,
        valid_sets=[lgb_vl],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    models.append(model)

    y_pred = model.predict(X_vl)
    auc = roc_auc_score(y_vl, y_pred)
    cv_scores.append(auc)
    print(f"  Fold {fold+1}: ROC-AUC = {auc:.4f}")

    if fold == 0:
        RocCurveDisplay.from_predictions(y_vl, y_pred, ax=axes[0],
                                          name=f"Fold {fold+1} (AUC={auc:.3f})",
                                          alpha=0.6)
    else:
        RocCurveDisplay.from_predictions(y_vl, y_pred, ax=axes[0],
                                          name=f"Fold {fold+1} (AUC={auc:.3f})",
                                          alpha=0.4)

mean_auc = np.mean(cv_scores)
std_auc = np.std(cv_scores)
print(f"\n  CV ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")

best_model = models[-1]
y_pred_val = best_model.predict(X_train_df.iloc[val_idx])
y_pred_bin = (y_pred_val >= 0.5).astype(int)
cm = confusion_matrix(y_train_arr[val_idx], y_pred_bin)
ConfusionMatrixDisplay(cm).plot(ax=axes[1], cmap="Blues", colorbar=False)
axes[1].set_title("Confusion Matrix (last fold)")

plt.suptitle("LightGBM — Model Evaluation", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_evaluation.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  -> {OUTPUT_DIR / 'model_evaluation.png'} сохранён")

# ═══════════════════════════════════════════════════════════════════════════════
#  5. PREDICT ON CONSUMER
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[5/6] Поиск скрытых предпринимателей среди consumer-карт …")

# Ансамбль моделей LightGBM
probas_lgb = np.zeros(len(X_con_scaled))
for model in models:
    probas_lgb += model.predict(pd.DataFrame(X_con_scaled, columns=common_feat_cols))
probas_lgb /= len(models)

# Финальная вероятность = средневзвешенное: 0.3 * iso_score + 0.7 * lgb_score
final_probs = 0.3 * iso_probs + 0.7 * probas_lgb

con_feat["iso_probability"] = iso_probs
con_feat["lgb_probability"] = probas_lgb
con_feat["probability"] = final_probs
con_feat["is_hidden_entrepreneur"] = (final_probs >= HIDDEN_THRESHOLD).astype(int)

n_hidden = con_feat["is_hidden_entrepreneur"].sum()
n_total = len(con_feat)
pct_hidden = 100.0 * n_hidden / n_total

print(f"\n  Всего consumer-карт: {n_total}")
print(f"  Из них скрытых предпринимателей (>= {HIDDEN_THRESHOLD}): {n_hidden} ({pct_hidden:.2f}%)")
print(f"  Средняя вероятность: {final_probs.mean():.4f}")

# Сохраняем CSV
out_cols = ["card_number", "probability", "is_hidden_entrepreneur",
            "iso_probability", "lgb_probability"]
out_df = con_feat[out_cols].sort_values("probability", ascending=False).reset_index(drop=True)
out_df.to_csv(OUTPUT_DIR / "hidden_entrepreneurs.csv", index=False)
print(f"  -> {OUTPUT_DIR / 'hidden_entrepreneurs.csv'} сохранён ({len(out_df):,} строк)")

# ═══════════════════════════════════════════════════════════════════════════════
#  6. SHAP
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[6/6] SHAP analysis …")

explainer = shap.TreeExplainer(best_model)
sample_idx = np.random.choice(len(X_con_scaled), size=min(2000, len(X_con_scaled)), replace=False)
X_sample = pd.DataFrame(X_con_scaled[sample_idx], columns=common_feat_cols)
shap_values = explainer.shap_values(X_sample)

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_sample, plot_size=(10, 7), show=False)
plt.title("SHAP Summary — Consumer Cards", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  -> {OUTPUT_DIR / 'shap_summary.png'} сохранён")

plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values, X_sample, plot_type="bar", plot_size=(8, 5), show=False)
plt.title("SHAP Feature Importance — Consumer Cards", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_importance.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  -> {OUTPUT_DIR / 'shap_importance.png'} сохранён")

# ═══════════════════════════════════════════════════════════════════════════════
#  7. PROBABILITY DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

plt.figure(figsize=(10, 5))
sns.histplot(final_probs, bins=50, kde=True, color="steelblue")
plt.axvline(HIDDEN_THRESHOLD, color="red", linestyle="--", label=f"Threshold = {HIDDEN_THRESHOLD}")
plt.title("Probability Distribution — Consumer Cards", fontsize=13, fontweight="bold")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "probability_distribution.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  -> {OUTPUT_DIR / 'probability_distribution.png'} сохранён")

# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  📊  SUMMARY REPORT  📊")
print("=" * 72)
print(f"  CV ROC-AUC:                {mean_auc:.4f} ± {std_auc:.4f}")
print(f"  Hidden entrepreneurs:      {n_hidden} / {n_total} ({pct_hidden:.2f}%)")
print(f"  Threshold:                 {HIDDEN_THRESHOLD}")
print(f"  Features used:             {len(common_feat_cols)}")
print(f"  Consumer cards analyzed:   {n_total}")
print(f"  Business cards (train):    {len(bus_feat)}")
print(f"  Consumer negatives (train): {len(y_train_neg)}")
print("=" * 72)
print("  Файлы результатов:")
print(f"    {OUTPUT_DIR / 'hidden_entrepreneurs.csv'}")
print(f"    {OUTPUT_DIR / 'model_evaluation.png'}")
print(f"    {OUTPUT_DIR / 'probability_distribution.png'}")
print(f"    {OUTPUT_DIR / 'shap_summary.png'}")
print(f"    {OUTPUT_DIR / 'shap_importance.png'}")
print("=" * 72)

gc.collect()
print("\n✅ Pipeline complete.\n")