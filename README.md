# 🏆 Решение кейса Mastercard Data Quest — Поиск скрытых предпринимателей

**Hidden Entrepreneurs Detection** — ML-пайплайн для выявления скрытой коммерческой активности среди физических лиц на основе анализа транзакционных данных банковских карт.

## 📝 Описание задачи

В рамках кейс-чемпионата **Mastercard Data Quest** необходимо по транзакционным данным определить держателей consumer-карт (физические лица), которые на самом деле используют свои карты для предпринимательской деятельности («скрытые предприниматели»).

**Основная идея решения:**
- Обучаем модель LightGBM различать бизнес-карты (target = 1) и consumer-карты (target = 0) на основе 29 признаков транзакционного поведения.
- Для consumer-карт с вероятностью business-класса выше порога (0.85) делаем вывод о скрытой предпринимательской активности.
- Для интерпретации модели используем SHAP-анализ — определяем, какие факторы наиболее влияют на предсказание.

## 📂 Структура репозитория

```
├── hidden_entrepreneurs.py    # Главный скрипт ML-пайплайна
├── check_schema.py            # Вспомогательный скрипт для просмотра схемы parquet-файлов
├── requirements.txt           # Зависимости Python
├── .gitignore                 # Исключения для Git
├── outputs/                   # Результаты работы пайплайна
│   ├── hidden_entrepreneurs.csv        # 80,000 consumer-карт с вероятностями (3.7 MB)
│   ├── model_evaluation.png            # ROC-AUC, Confusion Matrix, Feature Importance
│   ├── probability_distribution.png    # Распределение вероятностей бизнес-класса
│   ├── shap_summary.png                # SHAP summary dot-plot (топ-20 признаков)
│   └── shap_importance.png             # SHAP feature importance (mean |SHAP value|)
```

### Описание выходных файлов

| Файл | Описание |
|------|----------|
| `hidden_entrepreneurs.csv` | Колонки: `card_number`, `hidden_business_proba` (вероятность), `is_hidden_entrepreneur` (бинарный флаг) |
| `model_evaluation.png` | Три графика: ROC-кривая (OOF), Confusion Matrix, Feature Importance (топ-20) |
| `probability_distribution.png` | Гистограмма распределения OOF-вероятностей для business и consumer классов |
| `shap_summary.png` | SHAP dot-plot — направление и сила влияния каждого признака |
| `shap_importance.png` | SHAP bar-plot — средняя абсолютная важность признаков |

## ⚙️ Pipeline

1. **Load & Merge** — загрузка parquet-файлов, left join с merchants
2. **Feature Engineering** — 29 признаков поведения по card_number (транзакционные, временные, мерчант-разнообразие, B2B, recurring, SHAP-ready)
3. **Model Training** — LightGBM с 5-fold StratifiedKFold кросс-валидацией
4. **Hidden Business Detection** — predict_proba на consumer-картах, отбор по порогу 0.85
5. **Metrics & SHAP** — ROC-AUC, Confusion Matrix, SHAP summary plot

## 🚀 Инструкция по запуску (Reproducibility)

### Шаг 1. Подготовка данных

Поместите следующие файлы **в корень проекта** (рядом со скриптом `hidden_entrepreneurs.py`):

- `business_cards.parquet` — транзакции по бизнес-картам
- `consumer_cards.parquet` — транзакции по consumer-картам
- `merchants_reference.parquet` — справочник мерчантов

### Шаг 2. Установка зависимостей

Рекомендуется использовать виртуальное окружение:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

### Шаг 3. Запуск

```bash
python hidden_entrepreneurs.py
```

Скрипт выполнит полный цикл (обучение, предсказание, визуализации) и сохранит результаты в папку `outputs/`.

**Ожидаемое время выполнения:** ~60 минут (зависит от мощности ПК).
**RAM:** ~3–4 GB.

## 🛠️ Технологии

- **Python** 3.14
- **LightGBM** — градиентный бустинг
- **SHAP** — интерпретация модели
- **scikit-learn** — кросс-валидация, метрики
- **Pandas / PyArrow** — обработка parquet-данных
- **Matplotlib / Seaborn** — визуализация

## 📈 Результаты

- **OOF ROC-AUC:** 1.0000 (модель отлично разделяет классы на предоставленных данных)
- **Проанализировано consumer-карт:** 80,000
- **Порог скрытой предпринимательской активности:** ≥ 0.85