import json

cells = []

def md(source):
    """Add a Markdown cell."""
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    })

def code(source):
    """Add a Code cell."""
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source]
    })

# ===== CELL 1: IMPORTS & SETTINGS =====
md("""# 🏆 Mastercard Data Quest — Поиск скрытых предпринимателей

**Hidden Entrepreneurs Detection** — ML-пайплайн для выявления скрытой коммерческой активности среди физических лиц на основе анализа транзакционных данных банковских карт.

### Постановка задачи

По транзакционным данным необходимо определить держателей consumer-карт (физические лица), которые на самом деле используют свои карты для предпринимательской деятельности («скрытые предприниматели»).

**План решения:**
1. Загрузка и объединение данных (business и consumer карты + справочник мерчантов)
2. Feature Engineering — 29 признаков поведения по каждой карте
3. Обучение LightGBM с 5-кратной стратифицированной кросс-валидацией
4. Предсказание вероятности скрытой предпринимательской активности для consumer-карт
5. Визуализация метрик и SHAP-анализ""")

md("""## 1️⃣ Импорт библиотек и глобальные настройки

В этом блоке мы подключаем все необходимые библиотеки:
- **pandas, numpy** — обработка данных
- **pyarrow** — чтение Parquet-файлов
- **lightgbm** — градиентный бустинг (основная модель)
- **scikit-learn** — кросс-валидация и метрики
- **shap** — интерпретация модели
- **matplotlib, seaborn** — визуализация

Также задаём seed для воспроизводимости и порог вероятности для отнесения карты к «скрытым предпринимателям».""")

code(r'''"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         MASTERCARD DATA QUEST — Hidden Entrepreneurs Detection              ║
║         Поиск "скрытых предпринимателей" среди физических лиц               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (для серверов без дисплея)
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

# ─── Пути к файлам ───────────────────────────────────────────────
DATA_DIR = Path(".")
BUSINESS_FILE  = DATA_DIR / "business_cards.parquet"
CONSUMER_FILE  = DATA_DIR / "consumer_cards.parquet"
MERCHANTS_FILE = DATA_DIR / "merchants_reference.parquet"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Порог вероятности для отнесения к "скрытым предпринимателям"
HIDDEN_THRESHOLD = 0.85

# Seed для воспроизводимости
RANDOM_STATE = 42

print("✅ Imports OK | LightGBM version:", lgb.__version__)
''')

# ===== CELL 2: LOAD & MERGE =====
md("""## 2️⃣ Загрузка и объединение данных

В этом блоке мы:
1. Загружаем **справочник мерчантов** (`merchants_reference.parquet`) и оставляем только нужные колонки: `merchant_id`, `merchant_name`, `mcc`, `merchant_country`, `recurring_capable`.
2. Загружаем **транзакции бизнес-карт** (`business_cards.parquet`) — это карты, заведомо используемые для предпринимательской деятельности (target = 1).
3. Загружаем **транзакции consumer-карт** (`consumer_cards.parquet`) — физические лица (target = 0).
4. Объединяем оба датасета и делаем left join со справочником мерчантов.

> **Важно:** все файлы должны лежать в корневой папке проекта. Если названия файлов отличаются (например, с суффиксом `_MDQ`), раскомментируйте соответствующую строку в коде.""")

code(r'''def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Понижает битность числовых колонок для экономии RAM."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique() / len(df) < 0.5:
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


# ─── Загружаем справочник мерчантов ──────────────────────────────
merchants = pd.read_parquet(MERCHANTS_FILE)
merchants = optimize_dtypes(merchants)
merchants = merchants[["merchant_id", "merchant_name", "mcc", "merchant_country", "recurring_capable"]].copy()
print(f"  Merchants reference: {len(merchants):,} rows")

# ─── Загружаем транзакции ─────────────────────────────────────────
df_business = load_transactions(BUSINESS_FILE, target=1, label="Business cards")
df_consumer = load_transactions(CONSUMER_FILE, target=0, label="Consumer cards")

# ─── Объединяем в единый датасет ──────────────────────────────────
df_all = pd.concat([df_business, df_consumer], ignore_index=True)
del df_business, df_consumer
gc.collect()

# ─── Left join с мерчантами ───────────────────────────────────────
df_all = df_all.merge(
    merchants.rename(columns={"mcc": "merchant_mcc_ref", "merchant_country": "merchant_country_ref"}),
    on="merchant_id",
    how="left"
)
del merchants
gc.collect()

# Приводим дату к datetime
df_all["transaction_date"] = pd.to_datetime(df_all["transaction_date"])
df_all["transaction_timestamp"] = pd.to_datetime(df_all["transaction_timestamp"])

print(f"\n  ✅ Combined dataset: {len(df_all):,} rows | {df_all['card_number'].nunique():,} unique cards")
print(f"  RAM usage: {df_all.memory_usage(deep=True).sum() / 1e9:.2f} GB")
''')

# ===== CELL 3: FEATURE ENGINEERING =====
md("""## 3️⃣ Feature Engineering — построение признаков

В этом блоке мы на основе сырых транзакций создаём **29 признаков** для каждой карты. Это самый важный этап пайплайна.

**Какие признаки мы строим:**

1. **Базовые статистики по суммам транзакций:** количество, сумма, среднее, медиана, стандартное отклонение, мин/макс, квартили. Дополнительно — межквартильный размах (IQR) и коэффициент вариации (CV). Чем выше вариация сумм, тем менее предсказуемо поведение — может указывать на предпринимательскую деятельность.

2. **Временные признаки:**
   - Доля ночных транзакций (00:00–05:59) — предприниматели чаще работают ночью.
   - Доля транзакций в выходные — бизнес-активность не привязана к рабочей неделе.
   - Количество активных дней и месяцев — чем их больше, тем регулярнее использование карты.
   - Средний интервал между транзакциями — бизнес-карты часто используются с разной периодичностью.

3. **Мерчант-разнообразие:**
   - Количество уникальных мерчантов и MCC-кодов.
   - Количество уникальных банков-эквайеров.
   - **Доля B2B MCC** — ключевой признак: если человек платит в оптовых магазинах, арендует оборудование или пользуется бизнес-услугами — это сильный сигнал скрытого предпринимательства.
   - **Энтропия MCC** — чем разнообразнее категории трат, тем выше энтропия.

4. **Каналы и гео:**
   - Доля онлайн-транзакций.
   - Доля кросс-бордерных транзакций (не Казахстан).

5. **Recurring и токенизация:**
   - Доля recurring (регулярных) платежей.
   - Доля токенизированных транзакций.

6. **Концентрация трат (HHI):** насколько сильно траты сконцентрированы у одного мерчанта. Бизнес-карты часто имеют высокую концентрацию (один поставщик).

7. **Уровень карты:** ordinal encoding (standard → infinite).""")

code(r'''# ===== Список B2B MCC-кодов =====
# Коды, характерные для b2b-платежей, оптовой торговли,
# профессиональных услуг, транспортных и логистических операций.
B2B_MCC = {
    5040, 5045, 5046, 5047, 5051, 5065, 5072, 5074, 5085, 5094, 5099,
    7372, 7371, 7374, 7379,
    7311, 7319,
    7359, 7513, 7519,
    1520, 1711, 1731, 1740, 1750, 1761, 1771,
    8111, 8742, 8743, 8999,
    4215, 4411, 4511, 4812, 4814,
    5111, 5112, 5943,
    5141, 5149,
    4813, 4899,
    6012, 6051, 6099,
}

NIGHT_HOURS = range(0, 6)  # 00:00–05:59


def safe_entropy(series):
    """Энтропия Шеннона — мера разнообразия категорий."""
    counts = series.value_counts(normalize=True)
    return float(-(counts * np.log2(counts + 1e-10)).sum())


def build_features(df):
    """Агрегирует транзакции по card_number и создаёт признаковую таблицу."""
    grp = df.groupby("card_number")

    print("  [1/7] Basic transaction stats...")
    feat = grp["transaction_amount_kzt"].agg(
        txn_count="count", txn_sum="sum", txn_mean="mean",
        txn_median="median", txn_std="std", txn_min="min", txn_max="max",
        txn_q25=lambda x: x.quantile(0.25),
        txn_q75=lambda x: x.quantile(0.75),
    ).reset_index()
    feat["txn_iqr"] = feat["txn_q75"] - feat["txn_q25"]
    feat["txn_cv"] = feat["txn_std"] / (feat["txn_mean"] + 1e-6)

    print("  [2/7] Time-based features...")
    df["hour"] = df["transaction_timestamp"].dt.hour
    df["dayofweek"] = df["transaction_timestamp"].dt.dayofweek
    df["month"] = df["transaction_timestamp"].dt.month

    time_feat = grp.agg(
        night_txn_ratio=("hour", lambda x: (x.isin(NIGHT_HOURS)).mean()),
        weekend_txn_ratio=("dayofweek", lambda x: (x >= 5).mean()),
        active_days=("transaction_date", "nunique"),
        active_months=("month", "nunique"),
    ).reset_index()

    # Средний интервал между транзакциями (дни)
    df_sorted = df.sort_values(["card_number", "transaction_timestamp"])
    df_sorted["prev_ts"] = df_sorted.groupby("card_number")["transaction_timestamp"].shift(1)
    df_sorted["gap_days"] = (df_sorted["transaction_timestamp"] - df_sorted["prev_ts"]).dt.total_seconds() / 86400
    gap_feat = df_sorted.groupby("card_number")["gap_days"].agg(
        avg_txn_gap_days="mean", std_txn_gap_days="std"
    ).reset_index()
    del df_sorted
    gc.collect()

    print("  [3/7] Merchant diversity features...")
    merch_feat = grp.agg(
        unique_merchants=("merchant_id", "nunique"),
        unique_mcc=("mcc", "nunique"),
        unique_banks=("bank_name", "nunique"),
    ).reset_index()

    df["is_b2b_mcc"] = df["mcc"].isin(B2B_MCC)
    b2b_feat = grp["is_b2b_mcc"].mean().reset_index().rename(columns={"is_b2b_mcc": "b2b_mcc_ratio"})

    mcc_entropy = df.groupby("card_number")["mcc"].apply(safe_entropy).reset_index()
    mcc_entropy.columns = ["card_number", "mcc_entropy"]

    print("  [4/7] Channel & geo features...")
    channel_feat = grp.agg(
        online_ratio=("channel", lambda x: (x.str.lower() == "online").mean()),
        cross_border_ratio=("country", lambda x: (x != "KZ").mean()),
    ).reset_index()

    print("  [5/7] Recurring & tokenized features...")
    recurring_feat = grp.agg(
        recurring_ratio=("is_recurring", "mean"),
        tokenized_ratio=("tokenized", "mean"),
    ).reset_index()

    print("  [6/7] Merchant reference features...")
    df["recurring_capable"] = df["recurring_capable"].fillna(False)
    rc_feat = grp["recurring_capable"].mean().reset_index().rename(
        columns={"recurring_capable": "recurring_capable_merch_ratio"}
    )

    # HHI — концентрация трат у топ-1 мерчанта
    top1_share = df.groupby(["card_number", "merchant_id"])["transaction_amount_kzt"].sum().reset_index()
    total_per_card = df.groupby("card_number")["transaction_amount_kzt"].sum().reset_index()
    top1_share = top1_share.merge(total_per_card, on="card_number", suffixes=("_merch", "_total"))
    top1_share["share"] = top1_share["transaction_amount_kzt_merch"] / (top1_share["transaction_amount_kzt_total"] + 1e-6)
    hhi_feat = top1_share.groupby("card_number")["share"].apply(lambda x: (x**2).sum()).reset_index().rename(columns={"share": "merchant_hhi"})
    del top1_share, total_per_card
    gc.collect()

    print("  [7/7] Card tier & label features...")
    tier_map = {t: i for i, t in enumerate(["standard", "classic", "silver", "gold", "platinum", "infinite"], start=1)}
    df["card_tier_enc"] = df["card_tier"].str.lower().map(tier_map).fillna(0).astype(int)
    tier_feat = grp["card_tier_enc"].max().reset_index().rename(columns={"card_tier_enc": "card_tier_max"})

    target_feat = grp["target"].first().reset_index()

    print("  Merging all feature groups...")
    result = feat.copy()
    for other in [time_feat, gap_feat, merch_feat, b2b_feat, mcc_entropy,
                  channel_feat, recurring_feat, rc_feat, hhi_feat,
                  tier_feat, target_feat]:
        result = result.merge(other, on="card_number", how="left")

    result["is_consumer_original"] = result["target"] == 0
    return result


# ─── Строим признаки ──────────────────────────────────────────────
features_df = build_features(df_all)
del df_all
gc.collect()

# Заполнение NaN
num_cols = features_df.select_dtypes(include=["number"]).columns.tolist()
features_df[num_cols] = features_df[num_cols].fillna(0)

print(f"\n  ✅ Feature matrix: {features_df.shape[0]:,} cards × {features_df.shape[1]} columns")
print(f"  Class distribution:\n{features_df['target'].value_counts()}")
''')

# ===== CELL 4: MODEL TRAINING =====
md("""## 4️⃣ Обучение модели LightGBM

В этом блоке мы обучаем градиентный бустинг LightGBM с **5-кратной стратифицированной кросс-валидацией**.

**Параметры обучения:**
- `is_unbalance=True` — автоматическая балансировка классов (business-карт меньше)
- `num_leaves=63`, `feature_fraction=0.8`, `bagging_fraction=0.8` — регуляризация для борьбы с переобучением
- `early_stopping_rounds=50` — остановка, если AUC не улучшается

Для каждого фолда считается **ROC-AUC** — основная метрика качества. Финальная оценка — **средний OOF AUC** по всем фолдам. Также сохраняются importance признаков и сами модели для ансамблевого предсказания.""")

code(r'''FEATURE_COLS = [
    c for c in features_df.columns
    if c not in ("card_number", "target", "is_consumer_original")
]

X = features_df[FEATURE_COLS].values
y = features_df["target"].values

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
    "verbose": -1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "is_unbalance": True,
}

SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

oof_preds = np.zeros(len(X))
feature_imp = np.zeros(len(FEATURE_COLS))
models = []

print(f"  Features: {len(FEATURE_COLS)} | Folds: 5")

for fold, (train_idx, val_idx) in enumerate(SKF.split(X, y), start=1):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.log_evaluation(period=100)])

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    feature_imp += model.feature_importances_ / 5
    models.append(model)

    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"  Fold {fold} — AUC: {fold_auc:.4f}")

overall_auc = roc_auc_score(y, oof_preds)
print(f"\n  ✅ Overall OOF AUC: {overall_auc:.4f}")
''')

# ===== CELL 5: HIDDEN BUSINESS DETECTION =====
md("""## 5️⃣ Выявление скрытых предпринимателей

Теперь самое интересное! Используем обученную модель для предсказания на **consumer-картах**.

**Процесс:**
1. Берём все consumer-карты (те, у которых `is_consumer_original = True`).
2. Каждая из 5 моделей из кросс-валидации делает предсказание вероятности бизнес-класса.
3. Усредняем предсказания (ансамбль) — это даёт более стабильную оценку.
4. Если средняя вероятность ≥ 0.85 — карта помечается как **«скрытый предприниматель»**.

**Результат сохраняется в CSV-файл:** `outputs/hidden_entrepreneurs.csv`
с колонками: `card_number`, `hidden_business_proba`, `is_hidden_entrepreneur`.""")

code(r'''consumer_mask = features_df["is_consumer_original"].values
X_consumer = features_df.loc[consumer_mask, FEATURE_COLS].values

# Ensemble predict: среднее из 5 моделей
ensemble_proba = np.mean(
    [m.predict_proba(X_consumer)[:, 1] for m in models],
    axis=0
)

consumer_results = features_df[consumer_mask][["card_number"]].copy()
consumer_results["hidden_business_proba"] = ensemble_proba
consumer_results["is_hidden_entrepreneur"] = ensemble_proba >= HIDDEN_THRESHOLD

hidden_count = consumer_results["is_hidden_entrepreneur"].sum()
total_consumer = len(consumer_results)

print(f"  Consumer cards analyzed:    {total_consumer:,}")
print(f"  Hidden entrepreneurs found: {hidden_count:,}  ({hidden_count/total_consumer*100:.2f}%)")
print(f"  Probability threshold used: {HIDDEN_THRESHOLD}")

# Сохраняем в CSV
out_path = OUTPUT_DIR / "hidden_entrepreneurs.csv"
consumer_results.sort_values("hidden_business_proba", ascending=False).to_csv(out_path, index=False)
print(f"\n  ✅ Results saved → {out_path}")
''')

# ===== CELL 6: METRICS & SHAP =====
md("""## 6️⃣ Метрики и SHAP-анализ

В этом финальном блоке мы:
1. Выводим **Classification Report** — precision, recall, f1-score для consumer и business классов.
2. Строим **ROC-кривую** с AUC — показывает способность модели разделять классы.
3. Строим **Confusion Matrix** — сколько карт классифицировано верно / неверно.
4. Выводим **Top-20 Feature Importance** — какие признаки наиболее важны для модели.
5. Строим **гистограмму распределения вероятностей** для обоих классов.
6. Считаем **SHAP values** (на сбалансированной выборке 3,000 карт) — определяем, как каждый признак влияет на итоговое предсказание.

Все графики сохраняются в папку `outputs/`.""")

code(r'''# ─── 6a. Classification Report ─────────────────────────────────
oof_binary = (oof_preds >= 0.5).astype(int)
print("\n📋 Classification Report (OOF, threshold=0.5):")
print(classification_report(y, oof_binary, target_names=["Consumer", "Business"]))

# ─── 6b. ROC-AUC Curve + Confusion Matrix + Feature Importance ───
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Mastercard Data Quest — Model Evaluation", fontsize=14, fontweight="bold")

RocCurveDisplay.from_predictions(y, oof_preds, ax=axes[0], name=f"LightGBM (AUC={overall_auc:.3f})")
axes[0].set_title("ROC Curve (OOF)")
axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)

cm = confusion_matrix(y, oof_binary)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Consumer", "Business"])
disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title("Confusion Matrix (OOF, threshold=0.5)")

fi_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": feature_imp}).sort_values("importance", ascending=False).head(20)
axes[2].barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="steelblue")
axes[2].set_title("Top-20 Feature Importance (LightGBM)")
axes[2].set_xlabel("Importance (avg. over folds)")

plt.tight_layout()
fig_path = OUTPUT_DIR / "model_evaluation.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  ✅ Evaluation plots saved → {fig_path}")

# ─── 6c. Probability Distribution ─────────────────────────────────
fig2, ax = plt.subplots(figsize=(10, 4))
ax.hist(oof_preds[y == 0], bins=80, alpha=0.6, label="Consumer (true)", color="steelblue", density=True)
ax.hist(oof_preds[y == 1], bins=80, alpha=0.6, label="Business (true)", color="tomato", density=True)
ax.axvline(HIDDEN_THRESHOLD, color="black", linestyle="--", linewidth=1.2, label=f"Hidden threshold = {HIDDEN_THRESHOLD}")
ax.set_xlabel("Predicted Probability of Business")
ax.set_ylabel("Density")
ax.set_title("OOF Probability Distribution by True Class")
ax.legend()
plt.tight_layout()
fig2_path = OUTPUT_DIR / "probability_distribution.png"
plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  ✅ Probability distribution saved → {fig2_path}")

# ─── 6d. SHAP Values ──────────────────────────────────────────────
print("\n  Computing SHAP values (on sample of 3,000 cards)...")

shap_sample_idx = np.concatenate([
    np.random.choice(np.where(y == 1)[0], min(1500, (y == 1).sum()), replace=False),
    np.random.choice(np.where(y == 0)[0], min(1500, (y == 0).sum()), replace=False),
])
X_shap = X[shap_sample_idx]

explainer = shap.TreeExplainer(models[0])
shap_values = explainer.shap_values(X_shap)

sv = shap_values[1] if isinstance(shap_values, list) else shap_values

fig3, ax3 = plt.subplots(figsize=(10, 8))
shap.summary_plot(sv, X_shap, feature_names=FEATURE_COLS, max_display=20, show=False, plot_type="dot")
plt.title("SHAP Summary Plot — Why the model predicts 'Business'", fontsize=12, pad=15)
plt.tight_layout()
fig3_path = OUTPUT_DIR / "shap_summary.png"
plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  ✅ SHAP summary saved → {fig3_path}")

fig4, ax4 = plt.subplots(figsize=(9, 7))
shap.summary_plot(sv, X_shap, feature_names=FEATURE_COLS, max_display=20, show=False, plot_type="bar")
plt.title("SHAP Feature Importance — Mean |SHAP value|", fontsize=12, pad=15)
plt.tight_layout()
fig4_path = OUTPUT_DIR / "shap_importance.png"
plt.savefig(fig4_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  ✅ SHAP importance saved → {fig4_path}")

# ─── 6e. Top-10 Hidden Entrepreneurs ──────────────────────────────
print("\n" + "=" * 70)
print("Top-10 hidden entrepreneurs by probability:")
top_hidden = (
    consumer_results[consumer_results["is_hidden_entrepreneur"]]
    .sort_values("hidden_business_proba", ascending=False)
    .head(10)
)
if len(top_hidden) > 0:
    print(top_hidden.to_string(index=False))
else:
    print("No hidden entrepreneurs found at threshold", HIDDEN_THRESHOLD)

print("\n🏁 Pipeline complete!")
''')


# ===== BUILD NOTEBOOK =====
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.14.3"
        }
    },
    "cells": cells
}

with open("solution.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("✅ Notebook 'solution.ipynb' created successfully!")
print(f"   Cells: {len(cells)} (Markdown + Code pairs)")