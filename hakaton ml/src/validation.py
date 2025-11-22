import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import log_info, check_data_leakage

def evaluate_model(model, X_train, y_train, X_val=None, y_val=None, cv=5):

    log_info("\n" + "=" * 60)
    log_info("валидация модели")
    log_info("=" * 60)
    
    metrics = {}
    
    # Кросс-валидация на train (только для sklearn-совместимых моделей)
    log_info(f"\nКросс-валидация (CV={cv})...")
    try:
        # Проверяем, есть ли метод fit (для sklearn моделей)
        if hasattr(model, 'fit') and callable(getattr(model, 'fit')):
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
            log_info(f"  CV ROC-AUC: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
        else:
            log_info(f"    Кросс-валидация пропущена (модель не sklearn-совместима)")
            metrics['cv_mean'] = 0
            metrics['cv_std'] = 0
    except Exception as e:
        log_info(f"    Ошибка кросс-валидации: {e}")
        metrics['cv_mean'] = 0
        metrics['cv_std'] = 0
    
    # Валидация на отдельной выборке
    if X_val is not None and y_val is not None:
        log_info("\nвалидация на отдельной выборке")
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        
        metrics['val_roc_auc'] = roc_auc_score(y_val, y_pred_proba)
        metrics['val_accuracy'] = accuracy_score(y_val, y_pred)
        metrics['val_precision'] = precision_score(y_val, y_pred, zero_division=0)
        metrics['val_recall'] = recall_score(y_val, y_pred, zero_division=0)
        metrics['val_f1'] = f1_score(y_val, y_pred, zero_division=0)
        
        log_info(f"  ROC-AUC: {metrics['val_roc_auc']:.4f}")
        log_info(f"  Accuracy: {metrics['val_accuracy']:.4f}")
        log_info(f"  Precision: {metrics['val_precision']:.4f}")
        log_info(f"  Recall: {metrics['val_recall']:.4f}")
        log_info(f"  F1-score: {metrics['val_f1']:.4f}")
        
        # Детальный отчет
        log_info("\n  Classification Report:")
        report = classification_report(y_val, y_pred, target_names=['отчислен', 'выпустился'])
        log_info(f"\n{report}")
    
    return metrics

def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def validate_features(train_df, test_df, feature_cols):

    log_info("\n" + "=" * 60)
    log_info("проверка признаков на даталики")
    log_info("=" * 60)
    
    warnings = check_data_leakage(train_df, test_df, feature_cols)
    return warnings

