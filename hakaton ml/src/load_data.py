"""
Загрузка и первичный анализ данных
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import DATA_RAW, log_info, save_dataframe

def load_raw_data():
    """
    Загрузка сырых данных из CSV файлов
    
    Returns:
        data_df: DataFrame с данными об оценках
        marking_df: DataFrame с информацией о студентах
        sample_submission_df: DataFrame с форматом сабмита
    """
    log_info("=" * 60)
    log_info("ЗАГРУЗКА ДАННЫХ")
    log_info("=" * 60)
    
    # Загрузка данных об оценках
    # Обработка странного пути с двоеточием
    data_path = str(DATA_RAW).replace("raw", "raw:") + "/data.csv"
    if not Path(data_path).exists():
        # Пробуем альтернативный путь
        alt_path = Path(__file__).parent.parent / "data" / "raw:" / "data.csv"
        if alt_path.exists():
            data_path = alt_path
        else:
            data_path = DATA_RAW / "data.csv"
    log_info(f"Загрузка {data_path}...")
    data_df = pd.read_csv(data_path)
    log_info(f"  Загружено строк: {len(data_df):,}")
    log_info(f"  Колонки: {list(data_df.columns)}")
    
    # Загрузка информации о студентах
    marking_path = str(DATA_RAW).replace("raw", "raw:") + "/marking.csv"
    if not Path(marking_path).exists():
        alt_path = Path(__file__).parent.parent / "data" / "raw:" / "marking.csv"
        if alt_path.exists():
            marking_path = alt_path
        else:
            marking_path = DATA_RAW / "marking.csv"
    log_info(f"Загрузка {marking_path}...")
    marking_df = pd.read_csv(marking_path)
    log_info(f"  Загружено строк: {len(marking_df):,}")
    log_info(f"  Колонки: {list(marking_df.columns)}")
    
    # Загрузка sample submission
    sample_path = str(DATA_RAW).replace("raw", "raw:") + "/sample_submission.csv"
    if not Path(sample_path).exists():
        alt_path = Path(__file__).parent.parent / "data" / "raw:" / "sample_submission.csv"
        if alt_path.exists():
            sample_path = alt_path
        else:
            sample_path = DATA_RAW / "sample_submission.csv"
    log_info(f"Загрузка {sample_path}...")
    sample_submission_df = pd.read_csv(sample_path)
    log_info(f"  Загружено строк: {len(sample_submission_df):,}")
    log_info(f"  Колонки: {list(sample_submission_df.columns)}")
    
    return data_df, marking_df, sample_submission_df

def analyze_data(data_df, marking_df):
    """
    Первичный анализ данных
    
    Args:
        data_df: DataFrame с данными об оценках
        marking_df: DataFrame с информацией о студентах
    """
    log_info("\n" + "=" * 60)
    log_info("АНАЛИЗ ДАННЫХ")
    log_info("=" * 60)
    
    # Анализ data.csv
    log_info("\n📊 Анализ data.csv (оценки):")
    log_info(f"  Уникальных студентов: {data_df['PK'].nunique():,}")
    log_info(f"  Уникальных семестров: {sorted(data_df['SEMESTER'].unique())}")
    log_info(f"  Диапазон семестров: {data_df['SEMESTER'].min()} - {data_df['SEMESTER'].max()}")
    log_info(f"  Уникальных дисциплин: {data_df['DNAME'].nunique():,}")
    log_info(f"  Типы оценок: {data_df['TYPE'].value_counts().to_dict()}")
    
    # Пропуски
    log_info("\n  Пропуски в данных:")
    missing = data_df.isnull().sum()
    for col, count in missing[missing > 0].items():
        pct = count / len(data_df) * 100
        log_info(f"    {col}: {count:,} ({pct:.1f}%)")
    
    # Анализ marking.csv
    log_info("\n📊 Анализ marking.csv (студенты):")
    log_info(f"  Уникальных студентов: {marking_df['ИД'].nunique():,}")
    
    # Целевая переменная
    if 'выпуск' in marking_df.columns:
        log_info(f"\n  Распределение целевой переменной 'выпуск':")
        target_dist = marking_df['выпуск'].value_counts()
        for val, count in target_dist.items():
            pct = count / len(marking_df) * 100
            log_info(f"    '{val}': {count:,} ({pct:.1f}%)")
    
    # Статусы студентов
    if 'статус' in marking_df.columns:
        log_info(f"\n  Распределение статусов:")
        status_dist = marking_df['статус'].value_counts()
        for val, count in status_dist.items():
            pct = count / len(marking_df) * 100
            log_info(f"    {val}: {count:,} ({pct:.1f}%)")
    
    # Пересечение студентов
    students_in_data = set(data_df['PK'].unique())
    students_in_marking = set(marking_df['ИД'].unique())
    common_students = students_in_data & students_in_marking
    log_info(f"\n  Студентов в data.csv: {len(students_in_data):,}")
    log_info(f"  Студентов в marking.csv: {len(students_in_marking):,}")
    log_info(f"  Общих студентов: {len(common_students):,}")
    
    # Анализ по семестрам
    log_info("\n📊 Распределение данных по семестрам:")
    semester_counts = data_df['SEMESTER'].value_counts().sort_index()
    for sem, count in semester_counts.items():
        log_info(f"  Семестр {sem}: {count:,} записей")
    
    # Определение границы 2 курса (семестры 1-4 = первые 2 курса)
    log_info("\n📌 ЛОГИКА РАЗДЕЛЕНИЯ:")
    log_info("  Train: данные за семестры 1-4 (первые 2 курса)")
    log_info("  Test: студенты из sample_submission.csv")
    
    return {
        'students_in_data': len(students_in_data),
        'students_in_marking': len(students_in_marking),
        'common_students': len(common_students),
        'max_semester': data_df['SEMESTER'].max(),
        'min_semester': data_df['SEMESTER'].min()
    }

if __name__ == "__main__":
    # Загрузка данных
    data_df, marking_df, sample_submission_df = load_raw_data()
    
    # Анализ
    stats = analyze_data(data_df, marking_df)
    
    # Сохранение сырых данных в processed для дальнейшей работы
    save_dataframe(data_df, "data_raw.parquet")
    save_dataframe(marking_df, "marking_raw.parquet")
    save_dataframe(sample_submission_df, "sample_submission.parquet")
    
    log_info("\n✅ Загрузка данных завершена!")

