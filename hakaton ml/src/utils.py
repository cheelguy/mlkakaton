import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Настройка окружения для LightGBM (требует libomp на macOS)
def setup_lightgbm_environment():
    """Настройка переменных окружения для LightGBM"""
    libomp_path = Path("/opt/homebrew/opt/libomp")
    if libomp_path.exists():
        os.environ.setdefault("LDFLAGS", "-L/opt/homebrew/opt/libomp/lib")
        os.environ.setdefault("CPPFLAGS", "-I/opt/homebrew/opt/libomp/include")
        # Также для динамической загрузки библиотек
        import sys
        if sys.platform == "darwin":  # macOS
            lib_path = str(libomp_path / "lib")
            if "DYLD_LIBRARY_PATH" in os.environ:
                os.environ["DYLD_LIBRARY_PATH"] = f"{lib_path}:{os.environ['DYLD_LIBRARY_PATH']}"
            else:
                os.environ["DYLD_LIBRARY_PATH"] = lib_path

# Вызываем настройку при импорте модуля
setup_lightgbm_environment()

# Настройка путей
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Создание директорий если их нет
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Настройка логирования
def setup_logging():
    log_file = OUTPUT_DIR / "logs.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def log_info(message):
    #Логирование информации
    logger = setup_logging()
    logger.info(message)

def save_dataframe(df, filename, directory=DATA_PROCESSED):
    #Сохранение DataFrame в parquet
    filepath = directory / filename
    df.to_parquet(filepath, index=False)
    log_info(f"Сохранено: {filepath}")

def load_dataframe(filename, directory=DATA_PROCESSED):
    #Загрузка DataFrame из parquet
    filepath = directory / filename
    if filepath.exists():
        return pd.read_parquet(filepath)
    else:
        raise FileNotFoundError(f"Файл не найден: {filepath}")

def check_data_leakage(train_df, test_df, feature_cols):
    #Проверка признаков на даталик
    warnings = []
    for col in feature_cols:
        if col in train_df.columns and col in test_df.columns:
            train_vals = train_df[col].dropna().unique()
            test_vals = test_df[col].dropna().unique()
            
            # Проверка на уникальные значения в тесте, которых нет в трейне
            unique_in_test = set(test_vals) - set(train_vals)
            if len(unique_in_test) > 0 and len(unique_in_test) / len(test_vals) > 0.1:
                warnings.append(f"  {col}: {len(unique_in_test)} уникальных значений в тесте отсутствуют в трейне")
            
            # Проверка на разные распределения
            if train_df[col].dtype in ['float64', 'int64']:
                train_mean = train_df[col].mean()
                test_mean = test_df[col].mean()
                if abs(train_mean - test_mean) / (abs(train_mean) + 1e-6) > 0.5:
                    warnings.append(f"  {col}: значительное различие средних (train: {train_mean:.2f}, test: {test_mean:.2f})")
    
    if warnings:
        log_info("предупреждения о возможных даталиках:")
        for w in warnings:
            log_info(w)
    else:
        log_info("  проверка на даталики: подозрительных признаков не найдено")
    
    return warnings

