import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import log_info, load_dataframe, MODELS_DIR
from src.validation import evaluate_model, train_test_split_stratified

def prepare_features_catboost(features_df, target_dict=None):
    # Исключаем не-признаки
    exclude_cols = ['student_id', 'Факультет', 'Направление', 'год поступления']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    X = features_df[feature_cols].fillna(0)
    
    # Определяем категориальные признаки (если есть)
    cat_features = []
    for i, col in enumerate(feature_cols):
        if 'encoded' in col.lower() or col in ['faculty_encoded', 'direction_encoded']:
            cat_features.append(i)
    
    if target_dict is not None:
        y = features_df['student_id'].map(target_dict).fillna(0).astype(int)
        return X, y, feature_cols, cat_features
    else:
        return X, None, feature_cols, cat_features

def train_catboost(X_train, y_train, X_val=None, y_val=None, cat_features=None):
    log_info("\n" + "=" * 60)
    log_info("ОБУЧЕНИЕ CATBOOST МОДЕЛИ")
    log_info("=" * 60)
    
    # Параметры модели
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=100,
        class_weights=[1, 1]  # можно настроить для балансировки
    )
    
    log_info("Обучение модели...")
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_features if cat_features else None,
            use_best_model=True,
            verbose=100
        )
    else:
        model.fit(
            X_train, y_train,
            cat_features=cat_features if cat_features else None,
            verbose=100
        )
    
    # Оценка модели
    metrics = evaluate_model(model, X_train, y_train, X_val, y_val, cv=5)
    
    # Важность признаков
    log_info("\n Топ-10 важных признаков:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(10).iterrows():
        log_info(f"  {row['feature']}: {row['importance']:.4f}")
    
    metrics['feature_importance'] = feature_importance.to_dict('records')
    
    return model, metrics

if __name__ == "__main__":
    # Загрузка данных
    log_info("Загрузка данных...")
    
    # Пробуем загрузить финальный датасет 
    try:
        train_final = load_dataframe("train_final.parquet")
        log_info("  Загружен финальный train датасет")
        
        # Подготовка признаков из финального датасета
        X_train = train_final.drop(['student_id', 'target'], axis=1)
        y_train = train_final['target']
        feature_cols = list(X_train.columns)
        
        # Определяем категориальные признаки
        cat_features = []
        for i, col in enumerate(feature_cols):
            if 'encoded' in col.lower() or col in ['faculty_encoded', 'direction_encoded']:
                cat_features.append(i)
        
    except FileNotFoundError:
        # Fallback на старый способ
        log_info("  Финальный датасет не найден, используем старый способ...")
        features_df = load_dataframe("features.parquet")
        
        # Загрузка target для train
        train_target = load_dataframe("train_target.parquet")
        target_dict = dict(zip(train_target['ИД'], train_target['target']))
        
        # Разделение на train и test студентов
        train_students = set(train_target['ИД'].unique())
        train_features = features_df[features_df['student_id'].isin(train_students)].copy()
        
        # Подготовка признаков
        X_train, y_train, feature_cols, cat_features = prepare_features_catboost(train_features, target_dict)
    
    log_info(f"  Train размер: {len(X_train):,} студентов")
    log_info(f"  Количество признаков: {len(feature_cols)}")
    log_info(f"  Категориальных признаков: {len(cat_features)}")
    log_info(f"  Баланс классов: {y_train.value_counts().to_dict()}")
    
    # Разделение на train/val
    X_train_split, X_val, y_train_split, y_val = train_test_split_stratified(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Обучение модели
    model, metrics = train_catboost(X_train_split, y_train_split, X_val, y_val, cat_features)
    
    # Сохранение модели
    model_path = MODELS_DIR / "catboost_model.pkl"
    joblib.dump(model, model_path)
    log_info(f"\n Модель сохранена: {model_path}")
    
    # Сохранение информации
    import json
    model_info = {
        'feature_names': feature_cols,
        'cat_features': cat_features,
        'metrics': metrics
    }
    with open(Path(__file__).parent.parent / "output" / "catboost_info.json", 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    
    log_info("\n CatBoost обучение завершено!")

