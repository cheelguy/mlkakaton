"""
LightGBM модель для предсказания результатов обучения
"""
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Настройка окружения для LightGBM перед импортом
from src.utils import log_info, load_dataframe, MODELS_DIR, setup_lightgbm_environment
setup_lightgbm_environment()

# Теперь импортируем LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError as e:
    log_info(f"⚠️  LightGBM не установлен: {e}")
    LIGHTGBM_AVAILABLE = False
except OSError as e:
    log_info(f"⚠️  LightGBM не может загрузиться (требуется libomp): {e}")
    log_info("   Установите libomp: brew install libomp")
    LIGHTGBM_AVAILABLE = False

import joblib
from src.validation import evaluate_model, train_test_split_stratified

# Класс-обертка для LightGBM (должен быть на уровне модуля для pickle)
class LightGBMWrapper:
    """Обертка для LightGBM модели для совместимости с sklearn API"""
    def __init__(self, model):
        self.model = model
        self.feature_importances_ = model.feature_importance(importance_type='gain')
    
    def fit(self, X, y):
        """Для совместимости с sklearn (не используется, модель уже обучена)"""
        return self
    
    def predict_proba(self, X):
        """Предсказание вероятностей"""
        pred = self.model.predict(X)
        return np.column_stack([1 - pred, pred])
    
    def predict(self, X):
        """Предсказание классов"""
        return (self.model.predict(X) > 0.5).astype(int)

def prepare_features_lightgbm(features_df, target_dict=None):
    """
    Подготовка признаков для LightGBM
    
    Args:
        features_df: DataFrame с признаками
        target_dict: словарь {student_id: target} для train
    
    Returns:
        X: признаки
        y: целевая переменная (если target_dict передан)
        feature_names: названия признаков
        cat_features: названия категориальных признаков
    """
    # Исключаем не-признаки
    exclude_cols = ['student_id', 'Факультет', 'Направление', 'год поступления']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    X = features_df[feature_cols].fillna(0)
    
    # Определяем категориальные признаки
    cat_features = []
    for col in feature_cols:
        if 'encoded' in col.lower() or col in ['faculty_encoded', 'direction_encoded']:
            cat_features.append(col)
    
    if target_dict is not None:
        y = features_df['student_id'].map(target_dict).fillna(0).astype(int)
        return X, y, feature_cols, cat_features
    else:
        return X, None, feature_cols, cat_features

def train_lightgbm(X_train, y_train, X_val=None, y_val=None, cat_features=None):
    """
    Обучение LightGBM модели
    
    Args:
        X_train: признаки для обучения
        y_train: целевая переменная для обучения
        X_val: признаки для валидации (опционально)
        y_val: целевая переменная для валидации (опционально)
        cat_features: названия категориальных признаков
    
    Returns:
        model: обученная модель
        metrics: метрики модели
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM недоступен. Установите libomp: brew install libomp")
    
    log_info("\n" + "=" * 60)
    log_info("ОБУЧЕНИЕ LIGHTGBM МОДЕЛИ")
    log_info("=" * 60)
    
    # Параметры модели
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    # Подготовка данных для LightGBM
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features if cat_features else None)
    
    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features if cat_features else None, reference=train_data)
        log_info("Обучение модели с валидацией...")
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100)
            ]
        )
    else:
        log_info("Обучение модели...")
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            callbacks=[lgb.log_evaluation(period=100)]
        )
    
    # Обертка для совместимости с sklearn API
    wrapped_model = LightGBMWrapper(model)
    
    # Оценка модели
    metrics = evaluate_model(wrapped_model, X_train, y_train, X_val, y_val, cv=5)
    
    # Важность признаков
    log_info("\n📊 Топ-10 важных признаков:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': wrapped_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(10).iterrows():
        log_info(f"  {row['feature']}: {row['importance']:.4f}")
    
    metrics['feature_importance'] = feature_importance.to_dict('records')
    
    return wrapped_model, metrics

if __name__ == "__main__":
    # Проверка доступности LightGBM
    if not LIGHTGBM_AVAILABLE:
        log_info("❌ LightGBM недоступен!")
        log_info("   Для установки libomp выполните: brew install libomp")
        log_info("   Или используйте CatBoost как альтернативу: python src/model_catboost.py")
        sys.exit(1)
    
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
        
        # Определяем категориальные признаки
        cat_features = []
        for col in feature_cols:
            if 'encoded' in col.lower() or col in ['faculty_encoded', 'direction_encoded']:
                cat_features.append(col)
        
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
        X_train, y_train, feature_cols, cat_features = prepare_features_lightgbm(train_features, target_dict)
    
    log_info(f"  Train размер: {len(X_train):,} студентов")
    log_info(f"  Количество признаков: {len(feature_cols)}")
    log_info(f"  Категориальных признаков: {len(cat_features)}")
    log_info(f"  Баланс классов: {y_train.value_counts().to_dict()}")
    
    # Разделение на train/val
    X_train_split, X_val, y_train_split, y_val = train_test_split_stratified(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Обучение модели
    try:
        model, metrics = train_lightgbm(X_train_split, y_train_split, X_val, y_val, cat_features)
        
        # Сохранение модели
        model_path = MODELS_DIR / "lightgbm_model.pkl"
        joblib.dump(model, model_path)
        log_info(f"\n✅ Модель сохранена: {model_path}")
        
        # Сохранение информации
        import json
        model_info = {
            'feature_names': feature_cols,
            'cat_features': cat_features,
            'metrics': metrics
        }
        with open(Path(__file__).parent.parent / "output" / "lightgbm_info.json", 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        
        log_info("\n✅ LightGBM обучение завершено!")
    except Exception as e:
        log_info(f"\n❌ Ошибка при обучении LightGBM: {e}")
        log_info("   Убедитесь, что libomp установлен: brew install libomp")
        sys.exit(1)

