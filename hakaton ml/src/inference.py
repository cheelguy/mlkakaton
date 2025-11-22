import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import log_info, load_dataframe, MODELS_DIR, OUTPUT_DIR, setup_lightgbm_environment

# Настройка окружения для LightGBM перед загрузкой моделей
setup_lightgbm_environment()

def load_model(model_name='catboost'):
    model_path = MODELS_DIR / f"{model_name}_model.pkl"
    if model_name == 'baseline':
        model_path = MODELS_DIR / "baseline.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    
    log_info(f"Загрузка модели: {model_path}")
    model = joblib.load(model_path)
    return model

def prepare_test_features(features_df, test_student_ids):
    test_features = features_df[features_df['student_id'].isin(test_student_ids)].copy()
    
    # Проверяем, что все студенты из теста есть в features
    missing_students = set(test_student_ids) - set(test_features['student_id'].unique())
    if missing_students:
        log_info(f" ВНИМАНИЕ: {len(missing_students)} студентов из теста отсутствуют в features")
        log_info(f"  Это может означать, что у них нет данных за первые 2 курса")
    
    # Исключаем не-признаки
    exclude_cols = ['student_id', 'Факультет', 'Направление', 'год поступления']
    feature_cols = [col for col in test_features.columns if col not in exclude_cols]
    
    X_test = test_features[feature_cols].fillna(0)
    test_ids = test_features['student_id'].values
    
    log_info(f"  Тестовая выборка: {len(X_test):,} студентов")
    log_info(f"  Количество признаков: {len(feature_cols)}")
    
    return X_test, test_ids, feature_cols

def predict(model, X_test, model_name='catboost'):
    log_info(f"\nПредсказания моделью {model_name}...")
    
    if model_name == 'lightgbm':
        # LightGBM wrapper
        predictions = model.predict_proba(X_test)[:, 1]
    else:
        # CatBoost и baseline используют стандартный sklearn API
        predictions = model.predict_proba(X_test)[:, 1]
    
    log_info(f"  Предсказано: {len(predictions):,} студентов")
    log_info(f"  Средняя вероятность: {predictions.mean():.4f}")
    log_info(f"  Медианная вероятность: {np.median(predictions):.4f}")
    
    return predictions

def create_submission(test_ids, predictions, output_path=None):
    if output_path is None:
        output_path = OUTPUT_DIR / "submission.csv"
    
    # Преобразуем вероятности в классы (1 - выпустился, 0 - отчислен)
    # Порог 0.5 можно настроить
    classes = (predictions >= 0.5).astype(int)
    
    submission_df = pd.DataFrame({
        'ID': test_ids,
        'выпуск': classes
    })
    
    # Сортируем по ID для соответствия sample_submission
    submission_df = submission_df.sort_values('ID').reset_index(drop=True)
    
    # Сохранение
    submission_df.to_csv(output_path, index=False)
    log_info(f"\n Submission сохранен: {output_path}")
    log_info(f"  Размер: {len(submission_df):,} строк")
    log_info(f"  Распределение предсказаний:")
    log_info(f"    Выпустился (1): {(classes == 1).sum():,} ({(classes == 1).mean()*100:.1f}%)")
    log_info(f"    Отчислен (0): {(classes == 0).sum():,} ({(classes == 0).mean()*100:.1f}%)")
    
    return submission_df

def ensemble_predictions(models_dict, X_test):
    
    log_info("\n" + "=" * 60)
    log_info("АНСАМБЛЬ ПРЕДСКАЗАНИЙ")
    log_info("=" * 60)
    
    all_predictions = []
    
    for model_name, model in models_dict.items():
        log_info(f"\nПредсказания моделью {model_name}...")
        if model_name == 'lightgbm':
            pred = model.predict_proba(X_test)[:, 1]
        else:
            pred = model.predict_proba(X_test)[:, 1]
        all_predictions.append(pred)
        log_info(f"  Средняя вероятность: {pred.mean():.4f}")
    
    # Усреднение
    ensemble_pred = np.mean(all_predictions, axis=0)
    log_info(f"\n Ансамбль: средняя вероятность {ensemble_pred.mean():.4f}")
    
    return ensemble_pred

if __name__ == "__main__":
    # Загрузка данных
    log_info("=" * 60)
    log_info("INFERENCE НА ТЕСТОВОЙ ВЫБОРКЕ")
    log_info("=" * 60)
    
    # Пробуем загрузить финальный test датасет 
    try:
        test_final = load_dataframe("test_final.parquet")
        log_info("  Загружен финальный test датасет ")
        
        X_test = test_final.drop(['student_id'], axis=1)
        test_ids = test_final['student_id'].values
        feature_cols = list(X_test.columns)
        
    except FileNotFoundError:
        # Fallback на старый способ
        log_info("  Финальный test датасет не найден, используем старый способ...")
        features_df = load_dataframe("features.parquet")
        
        # Загрузка sample_submission для получения ID тестовых студентов
        sample_submission = load_dataframe("sample_submission.parquet")
        test_student_ids = sample_submission['ID'].unique().tolist()
        
        log_info(f"\nТестовая выборка: {len(test_student_ids):,} студентов")
        
        # Подготовка признаков
        X_test, test_ids, feature_cols = prepare_test_features(features_df, test_student_ids)
    
    # Выбор модели (можно изменить)
    model_name = 'catboost'  # или 'baseline', 'lightgbm'
    
    # Загрузка модели
    model = load_model(model_name)
    
    # Предсказания
    predictions = predict(model, X_test, model_name)
    
    # Создание submission
    submission_df = create_submission(test_ids, predictions)
    
    log_info("\n Inference завершен!")
    log_info(f"  Файл submission: {OUTPUT_DIR / 'submission.csv'}")

