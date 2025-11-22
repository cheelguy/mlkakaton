import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import log_info, load_dataframe, MODELS_DIR
from src.validation import evaluate_model, train_test_split_stratified

def prepare_features(features_df, target_dict=None):
    # Исключаем не-признаки
    exclude_cols = ['student_id', 'Факультет', 'Направление', 'год поступления']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    X = features_df[feature_cols].fillna(0)
    
    if target_dict is not None:
        y = features_df['student_id'].map(target_dict).fillna(0).astype(int)
        return X, y, feature_cols
    else:
        return X, None, feature_cols

def train_baseline(X_train, y_train, X_val=None, y_val=None):
    log_info("\n" + "=" * 60)
    log_info("ОБУЧЕНИЕ BASELINE МОДЕЛИ")
    log_info("=" * 60)
    
    # Создаем pipeline: стандартизация + логистическая регрессия
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # балансировка классов
        ))
    ])
    
    log_info("Обучение модели...")
    model.fit(X_train, y_train)
    
    # Оценка модели
    metrics = evaluate_model(model, X_train, y_train, X_val, y_val, cv=5)
    
    return model, metrics

if __name__ == "__main__":
    # Загрузка данных
    log_info("Загрузка данных...")
    
    # Пробуем загрузить финальный датасет (если подготовлен Никитой)
    try:
        train_final = load_dataframe("train_final.parquet")
        log_info("  Загружен финальный train датасет (подготовлен Никитой)")
        
        # Подготовка признаков из финального датасета
        X_train = train_final.drop(['student_id', 'target'], axis=1)
        y_train = train_final['target']
        feature_cols = list(X_train.columns)
        
    except FileNotFoundError:
        # Fallback на старый способ (если финальный датасет не готов)
        log_info("  Финальный датасет не найден, используем старый способ...")
        features_df = load_dataframe("features.parquet")
        
        # Загрузка target для train
        train_target = load_dataframe("train_target.parquet")
        target_dict = dict(zip(train_target['ИД'], train_target['target']))
        
        # Разделение на train и test студентов
        train_students = set(train_target['ИД'].unique())
        train_features = features_df[features_df['student_id'].isin(train_students)].copy()
        
        # Подготовка признаков
        X_train, y_train, feature_cols = prepare_features(train_features, target_dict)
    
    log_info(f"  Train размер: {len(X_train):,} студентов")
    log_info(f"  Количество признаков: {len(feature_cols)}")
    log_info(f"  Баланс классов: {y_train.value_counts().to_dict()}")
    
    # Разделение на train/val
    X_train_split, X_val, y_train_split, y_val = train_test_split_stratified(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Обучение модели
    model, metrics = train_baseline(X_train_split, y_train_split, X_val, y_val)
    
    # Сохранение модели
    model_path = MODELS_DIR / "baseline.pkl"
    joblib.dump(model, model_path)
    log_info(f"\n Модель сохранена: {model_path}")
    
    # Сохранение названий признаков
    import json
    features_info = {
        'feature_names': feature_cols,
        'metrics': metrics
    }
    with open(Path(__file__).parent.parent / "output" / "baseline_info.json", 'w') as f:
        json.dump(features_info, f, indent=2, default=str)
    
    log_info("\n Baseline обучение завершено!")

