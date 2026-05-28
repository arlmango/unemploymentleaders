"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         MASTERCARD DATA QUEST — Hidden Entrepreneurs Detection              ║
║         Поиск "скрытых предпринимателей" среди физических лиц               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Pipeline:
  1. Load & Merge        — загрузка parquet-файлов, join с merchants
  2. Feature Engineering — 40+ признаков поведения по card_number
  3. Model Training      — LightGBM с кросс-валидацией
  4. Hidden Business     — predict_proba на consumer-картах, порог 0.85
  5. Metrics & SHAP      — ROC-AUC, Confusion Matrix, SHAP summary plot

Зависимости: pandas, numpy, lightgbm, scikit-learn, shap, matplotlib, seaborn
"""

import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (нет Tkinter)
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lightgbm as lgb

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)

# ─── Пути к файлам (измените при необходимости) ───────────────────────────────
DATA_DIR = Path(".")                              # папка с parquet-файлами
BUSINESS_FILE  = DATA_DIR / "business_cards_MDQ.parquet"
CONSUMER_FILE  = DATA_DIR / "consumer_cards_MDQ.parquet"
MERCHANTS_FILE = DATA_DIR / "merchants_reference.parquet"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Порог вероятности для отнесения к "скрытым предпринимателям"
HIDDEN_THRESHOLD = 0.85

# Seed для воспроизводимости
RANDOM_STATE = 42

print("✅ Imports OK | LightGBM version:", lgb.__version__)


# ==============================================================================
# 1. LOAD & MERGE
# ==============================================================================
print("\n" + "="*70)
print("📂 STEP 1 — LOADING DATA")
print("="*70)

# ---------- Оптимизация памяти при чтении parquet ----------
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Понижает битность числовых колонок для экономии RAM."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique() / len(df) < 0.5:          # высокая повторяемость → category
            df[col] = df[col].astype("category")
    return df


def load_transactions(filepath: Path, target: int, label: str) -> pd.DataFrame:
    """Загружает файл транзакций, добавляет колонку target."""
    print(f"  Loading {label} → {filepath.name}...")
    df = pd.read_parquet(filepath)
    df = optimize_dtypes(df)
    df["target"] = target
    print(f"    Rows: {len(df):,} | Cards: {df['card_number'].nunique():,}")
    return df


# Загружаем справочник мерчантов
merchants = pd.read_parquet(MERCHANTS_FILE)
merchants = optimize_dtypes(merchants)
# Оставляем только нужные колонки из справочника
merchants = merchants[["merchant_id", "merchant_name", "mcc", "merchant_country", "recurring_capable"]].copy()
print(f"  Merchants reference: {len(merchants):,} rows")

# Загружаем транзакции
df_business = load_transactions(BUSINESS_FILE, target=1, label="Business cards")
df_consumer = load_transactions(CONSUMER_FILE, target=0, label="Consumer cards")

# Объединяем в единый датасет
df_all = pd.concat([df_business, df_consumer], ignore_index=True)
del df_business, df_consumer   # освобождаем RAM
gc.collect()

# Left join с мерчантами (обогащение данными)
df_all = df_all.merge(
    merchants.rename(columns={"mcc": "merchant_mcc_ref", "merchant_country": "merchant_country_ref"}),
    on="merchant_id",
    how="left"
)
del merchants
gc.collect()

# Приводим дату к datetime если ещё не приведена
df_all["transaction_date"] = pd.to_datetime(df_all["transaction_date"])
df_all["transaction_timestamp"] = pd.to_datetime(df_all["transaction_timestamp"])

print(f"\n  ✅ Combined dataset: {len(df_all):,} rows | {df_all['card_number'].nunique():,} unique cards")
print(f"  RAM usage: {df_all.memory_usage(deep=True).sum() / 1e9:.2f} GB")


# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================
print("\n" + "="*70)
print("⚙️  STEP 2 — FEATURE ENGINEERING")
print("="*70)

# ---------- B2B MCC коды (бизнес-ориентированные категории) ----------
# Логика: коды, характерные для b2b-платежей, оптовой торговли,
# профессиональных услуг, транспортных и логистических операций.
B2B_MCC = {
    # Оптовая торговля
    5040, 5045, 5046, 5047, 5051, 5065, 5072, 5074, 5085, 5094, 5099,
    # Бизнес-сервисы / IT
    7372, 7371, 7374, 7379, 5045,
    # Реклама и медиа
    7311, 7319,
    # Аренда оборудования и транспорта
    7359, 7513, 7519,
    # Строительство и подрядчики
    1520, 1711, 1731, 1740, 1750, 1761, 1771,
    # Профессиональные услуги (юридические, бухгалтерия)
    8111, 8742, 8743, 8999,
    # Транспорт и логистика
    4215, 4411, 4511, 4812, 4814,
    # Офисные расходы
    5111, 5112, 5943,
    # Поставщики продуктов питания (HoReCa)
    5141, 5149,
    # Телекоммуникации (корпоративные)
    4813, 4899,
    # Финансовые услуги
    6012, 6051, 6099,
}

# ---------- MCC ночной активности (нетипично для обычных физлиц) ----------
NIGHT_HOURS = range(0, 6)   # 00:00–05:59 считаем «ночью»

# ---------- Вспомогательные функции ----------

def safe_entropy(series: pd.Series) -> float:
    """Энтропия Шеннона распределения значений (мера разнообразия)."""
    counts = series.value_counts(normalize=True)
    return float(-(counts * np.log2(counts + 1e-10)).sum())


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует транзакции по card_number и создаёт признаковую таблицу.
    Возвращает DataFrame с одной строкой на карту.
    """
    grp = df.groupby("card_number")

    print("  [1/7] Basic transaction stats...")
    feat = grp["transaction_amount_kzt"].agg(
        txn_count        = "count",
        txn_sum          = "sum",
        txn_mean         = "mean",
        txn_median       = "median",
        txn_std          = "std",
        txn_min          = "min",
        txn_max          = "max",
        txn_q25          = lambda x: x.quantile(0.25),
        txn_q75          = lambda x: x.quantile(0.75),
    ).reset_index()

    # IQR и коэффициент вариации
    feat["txn_iqr"] = feat["txn_q75"] - feat["txn_q25"]
    feat["txn_cv"]  = feat["txn_std"] / (feat["txn_mean"] + 1e-6)

    print("  [2/7] Time-based features...")
    df["hour"]      = df["transaction_timestamp"].dt.hour
    df["dayofweek"] = df["transaction_timestamp"].dt.dayofweek   # 0=Mon
    df["month"]     = df["transaction_timestamp"].dt.month

    time_feat = grp.agg(
        night_txn_ratio   = ("hour", lambda x: (x.isin(NIGHT_HOURS)).mean()),
        weekend_txn_ratio = ("dayofweek", lambda x: (x >= 5).mean()),
        active_days       = ("transaction_date", "nunique"),
        active_months     = ("month", "nunique"),
    ).reset_index()

    # Средний интервал между транзакциями (дни) — показатель регулярности
    df_sorted = df.sort_values(["card_number", "transaction_timestamp"])
    df_sorted["prev_ts"] = df_sorted.groupby("card_number")["transaction_timestamp"].shift(1)
    df_sorted["gap_days"] = (
        (df_sorted["transaction_timestamp"] - df_sorted["prev_ts"])
        .dt.total_seconds() / 86400
    )
    gap_feat = df_sorted.groupby("card_number")["gap_days"].agg(
        avg_txn_gap_days = "mean",
        std_txn_gap_days = "std",
    ).reset_index()
    del df_sorted
    gc.collect()

    print("  [3/7] Merchant diversity features...")
    merch_feat = grp.agg(
        unique_merchants          = ("merchant_id", "nunique"),
        unique_mcc                = ("mcc", "nunique"),
        unique_banks              = ("bank_name", "nunique"),
    ).reset_index()

    # Доля B2B MCC транзакций
    df["is_b2b_mcc"] = df["mcc"].isin(B2B_MCC)
    b2b_feat = grp["is_b2b_mcc"].mean().reset_index().rename(
        columns={"is_b2b_mcc": "b2b_mcc_ratio"}
    )

    # Энтропия распределения по MCC (разнообразие категорий)
    mcc_entropy = df.groupby("card_number")["mcc"].apply(safe_entropy).reset_index()
    mcc_entropy.columns = ["card_number", "mcc_entropy"]

    print("  [4/7] Channel & geo features...")
    channel_feat = grp.agg(
        online_ratio        = ("channel", lambda x: (x.str.lower() == "online").mean()),
        cross_border_ratio  = ("country", lambda x: (x != "KZ").mean()),   # нерезидентные транзакции
    ).reset_index()

    print("  [5/7] Recurring & tokenized features...")
    recurring_feat = grp.agg(
        recurring_ratio  = ("is_recurring", "mean"),
        tokenized_ratio  = ("tokenized", "mean"),
    ).reset_index()

    print("  [6/7] Merchant reference features...")
    # Доля транзакций у мерчантов с recurring_capable=True
    df["recurring_capable"] = df["recurring_capable"].fillna(False)
    rc_feat = grp["recurring_capable"].mean().reset_index().rename(
        columns={"recurring_capable": "recurring_capable_merch_ratio"}
    )

    # Концентрация трат у топ-1 мерчанта (Herfindahl-Hirschman Index упрощённый)
    top1_share = (
        df.groupby(["card_number", "merchant_id"])["transaction_amount_kzt"]
        .sum()
        .reset_index()
    )
    total_per_card = df.groupby("card_number")["transaction_amount_kzt"].sum().reset_index()
    top1_share = top1_share.merge(total_per_card, on="card_number", suffixes=("_merch", "_total"))
    top1_share["share"] = top1_share["transaction_amount_kzt_merch"] / (
        top1_share["transaction_amount_kzt_total"] + 1e-6
    )
    hhi_feat = (
        top1_share.groupby("card_number")["share"]
        .apply(lambda x: (x**2).sum())
        .reset_index()
        .rename(columns={"share": "merchant_hhi"})
    )
    del top1_share, total_per_card
    gc.collect()

    print("  [7/7] Card tier & label features...")
    # Уровень карты (ordinal encoding)
    tier_map = {t: i for i, t in enumerate(
        ["standard", "classic", "silver", "gold", "platinum", "infinite"],
        start=1
    )}
    df["card_tier_enc"] = df["card_tier"].str.lower().map(tier_map).fillna(0).astype(int)
    tier_feat = grp["card_tier_enc"].max().reset_index().rename(
        columns={"card_tier_enc": "card_tier_max"}
    )

    # Target (берём из первой строки каждой карты)
    target_feat = grp["target"].first().reset_index()

    print("  Merging all feature groups...")
    # Последовательный merge всех групп
    result = feat.copy()
    for other in [
        time_feat, gap_feat, merch_feat, b2b_feat, mcc_entropy,
        channel_feat, recurring_feat, rc_feat, hhi_feat,
        tier_feat, target_feat
    ]:
        result = result.merge(other, on="card_number", how="left")

    # Добавляем флаг: является ли карта изначально consumer (для финального predict)
    result["is_consumer_original"] = result["target"] == 0

    return result


# Строим признаки
features_df = build_features(df_all)
del df_all
gc.collect()

# Финальная очистка: заполнение NaN
num_cols = features_df.select_dtypes(include=["number"]).columns.tolist()
features_df[num_cols] = features_df[num_cols].fillna(0)

print(f"\n  ✅ Feature matrix: {features_df.shape[0]:,} cards × {features_df.shape[1]} columns")
print(f"  Class distribution:\n{features_df['target'].value_counts()}")


# ==============================================================================
# 3. MODEL TRAINING (LightGBM + StratifiedKFold)
# ==============================================================================
print("\n" + "="*70)
print("🤖 STEP 3 — MODEL TRAINING")
print("="*70)

FEATURE_COLS = [
    c for c in features_df.columns
    if c not in ("card_number", "target", "is_consumer_original")
]

X = features_df[FEATURE_COLS].values
y = features_df["target"].values

# ─── LightGBM параметры ───────────────────────────────────────────────────────
LGB_PARAMS = {
    "objective":       "binary",
    "metric":          "auc",
    "learning_rate":   0.05,
    "num_leaves":      63,
    "max_depth":       -1,
    "min_child_samples": 50,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "lambda_l1":         0.1,
    "lambda_l2":         0.1,
    "n_estimators":      1000,
    "early_stopping_rounds": 50,
    "verbose":          -1,
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
    # Балансировка классов (бизнес-карт значительно меньше)
    "is_unbalance":     True,
}

SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

oof_preds   = np.zeros(len(X))     # out-of-fold предсказания
feature_imp = np.zeros(len(FEATURE_COLS))
models      = []

print(f"  Features: {len(FEATURE_COLS)} | Folds: 5")

for fold, (train_idx, val_idx) in enumerate(SKF.split(X, y), start=1):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(period=100)],
    )

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    feature_imp += model.feature_importances_ / 5
    models.append(model)

    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"  Fold {fold} — AUC: {fold_auc:.4f}")

overall_auc = roc_auc_score(y, oof_preds)
print(f"\n  ✅ Overall OOF AUC: {overall_auc:.4f}")


# ==============================================================================
# 4. DETECTING HIDDEN ENTREPRENEURS
# ==============================================================================
print("\n" + "="*70)
print("🔍 STEP 4 — IDENTIFYING HIDDEN ENTREPRENEURS")
print("="*70)

# Усредняем предсказания всех 5 фолдов для consumer-карт
consumer_mask = features_df["is_consumer_original"].values

X_consumer = features_df.loc[consumer_mask, FEATURE_COLS].values

# Ensemble predict: среднее из 5 моделей
ensemble_proba = np.mean(
    [m.predict_proba(X_consumer)[:, 1] for m in models],
    axis=0
)

# Сохраняем результат
consumer_results = features_df[consumer_mask][["card_number"]].copy()
consumer_results["hidden_business_proba"] = ensemble_proba
consumer_results["is_hidden_entrepreneur"] = ensemble_proba >= HIDDEN_THRESHOLD

hidden_count  = consumer_results["is_hidden_entrepreneur"].sum()
total_consumer = len(consumer_results)

print(f"  Consumer cards analyzed:    {total_consumer:,}")
print(f"  Hidden entrepreneurs found: {hidden_count:,}  ({hidden_count/total_consumer*100:.2f}%)")
print(f"  Probability threshold used: {HIDDEN_THRESHOLD}")

# Сохраняем в CSV
out_path = OUTPUT_DIR / "hidden_entrepreneurs.csv"
consumer_results.sort_values("hidden_business_proba", ascending=False).to_csv(
    out_path, index=False
)
print(f"\n  ✅ Results saved → {out_path}")


# ==============================================================================
# 5. METRICS & VISUALIZATIONS
# ==============================================================================
print("\n" + "="*70)
print("📊 STEP 5 — METRICS & EXPLAINABILITY")
print("="*70)

# ─── 5a. Classification Report ───────────────────────────────────────────────
oof_binary = (oof_preds >= 0.5).astype(int)
print("\n📋 Classification Report (OOF, threshold=0.5):")
print(classification_report(y, oof_binary, target_names=["Consumer", "Business"]))

# ─── 5b. ROC-AUC Curve ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Mastercard Data Quest — Model Evaluation", fontsize=14, fontweight="bold")

# ROC Curve
RocCurveDisplay.from_predictions(y, oof_preds, ax=axes[0], name=f"LightGBM (AUC={overall_auc:.3f})")
axes[0].set_title("ROC Curve (OOF)")
axes[0].plot([0,1],[0,1],"k--", linewidth=0.8)

# ─── 5c. Confusion Matrix ────────────────────────────────────────────────────
cm = confusion_matrix(y, oof_binary)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Consumer","Business"])
disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title("Confusion Matrix (OOF, threshold=0.5)")

# ─── 5d. Feature Importance (top-20) ─────────────────────────────────────────
fi_df = pd.DataFrame({
    "feature":    FEATURE_COLS,
    "importance": feature_imp
}).sort_values("importance", ascending=False).head(20)

axes[2].barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="steelblue")
axes[2].set_title("Top-20 Feature Importance (LightGBM)")
axes[2].set_xlabel("Importance (avg. over folds)")

plt.tight_layout()
fig_path = OUTPUT_DIR / "model_evaluation.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  ✅ Evaluation plots saved → {fig_path}")

# ─── 5e. Probability Distribution ────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(10, 4))
ax.hist(oof_preds[y == 0], bins=80, alpha=0.6, label="Consumer (true)", color="steelblue", density=True)
ax.hist(oof_preds[y == 1], bins=80, alpha=0.6, label="Business (true)", color="tomato",    density=True)
ax.axvline(HIDDEN_THRESHOLD, color="black", linestyle="--", linewidth=1.2,
           label=f"Hidden threshold = {HIDDEN_THRESHOLD}")
ax.set_xlabel("Predicted Probability of Business")
ax.set_ylabel("Density")
ax.set_title("OOF Probability Distribution by True Class")
ax.legend()
plt.tight_layout()
fig2_path = OUTPUT_DIR / "probability_distribution.png"
plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  ✅ Probability distribution saved → {fig2_path}")

# ─── 5f. SHAP Values ─────────────────────────────────────────────────────────
print("\n  Computing SHAP values (on sample of 3,000 cards)...")

# Берём balanced sample для SHAP (дорогостоящий расчёт)
shap_sample_idx = (
    np.concatenate([
        np.random.choice(np.where(y == 1)[0], min(1500, (y==1).sum()), replace=False),
        np.random.choice(np.where(y == 0)[0], min(1500, (y==0).sum()), replace=False),
    ])
)
X_shap = X[shap_sample_idx]

# Используем первую модель для SHAP (все 5 очень похожи)
explainer   = shap.TreeExplainer(models[0])
shap_values = explainer.shap_values(X_shap)

# LightGBM бинарная классификация: shap_values — список [class0, class1]
# Берём класс 1 (business)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

fig3, ax3 = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    sv,
    X_shap,
    feature_names=FEATURE_COLS,
    max_display=20,
    show=False,
    plot_type="dot"
)
plt.title("SHAP Summary Plot — Why the model predicts 'Business'", fontsize=12, pad=15)
plt.tight_layout()
fig3_path = OUTPUT_DIR / "shap_summary.png"
plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  ✅ SHAP summary saved → {fig3_path}")

# ─── SHAP bar plot (mean |SHAP|) ──────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(9, 7))
shap.summary_plot(
    sv,
    X_shap,
    feature_names=FEATURE_COLS,
    max_display=20,
    show=False,
    plot_type="bar"
)
plt.title("SHAP Feature Importance — Mean |SHAP value|", fontsize=12, pad=15)
plt.tight_layout()
fig4_path = OUTPUT_DIR / "shap_importance.png"
plt.savefig(fig4_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  ✅ SHAP importance saved → {fig4_path}")


# ==============================================================================
# 6. SUMMARY REPORT
# ==============================================================================
print("\n" + "="*70)
print("📝 FINAL SUMMARY")
print("="*70)

summary = f"""
╔═══════════════════════════════════════════════════════════════╗
║           HIDDEN ENTREPRENEURS — FINAL REPORT                 ║
╠═══════════════════════════════════════════════════════════════╣
║  Model        : LightGBM (5-fold StratifiedKFold ensemble)    ║
║  OOF ROC-AUC  : {overall_auc:.4f}                                   ║
║─────────────────────────────────────────────────────────────  ║
║  Consumer cards analyzed  : {total_consumer:>10,}                   ║
║  Hidden entrepreneurs     : {hidden_count:>10,}  (p ≥ {HIDDEN_THRESHOLD})      ║
║  Detection rate           : {hidden_count/total_consumer*100:>9.2f}%                   ║
╠═══════════════════════════════════════════════════════════════╣
║  Output files:                                                 ║
║    outputs/hidden_entrepreneurs.csv  — scored consumer cards  ║
║    outputs/model_evaluation.png      — ROC, CM, FI            ║
║    outputs/probability_distribution.png                       ║
║    outputs/shap_summary.png          — SHAP dot plot          ║
║    outputs/shap_importance.png       — SHAP bar plot          ║
╚═══════════════════════════════════════════════════════════════╝
"""
print(summary)

# Топ-10 скрытых предпринимателей по вероятности
print("Top-10 hidden entrepreneurs by probability:")
print(
    consumer_results[consumer_results["is_hidden_entrepreneur"]]
    .sort_values("hidden_business_proba", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\n🏁 Pipeline complete!")
