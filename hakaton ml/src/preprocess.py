"""
Предобработка данных: очистка, обработка пропусков, базовые преобразования
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import log_info, save_dataframe, load_dataframe

def preprocess_data(data_df):
    """
    Предобработка данных об оценках
    
    Args:
        data_df: DataFrame с сырыми данными об оценках
    
    Returns:
        processed_df: обработанный DataFrame
    """
    log_info("\n" + "=" * 60)
    log_info("ПРЕДОБРАБОТКА ДАННЫХ")
    log_info("=" * 60)
    
    df = data_df.copy()
    
    # Обработка дат
    if 'EXAM_DATE' in df.columns:
        log_info("Обработка дат...")
        df['EXAM_DATE'] = pd.to_datetime(df['EXAM_DATE'], errors='coerce')
        df['year'] = df['EXAM_DATE'].dt.year
        df['month'] = df['EXAM_DATE'].dt.month
        df['day_of_week'] = df['EXAM_DATE'].dt.dayofweek
    
    # Обработка оценок
    log_info("Обработка оценок...")
    
    # MARK: числовые оценки (5, 4, 3) и текстовые (з, ня)
    # Создаем числовую версию оценки
    def parse_mark(mark):
        if pd.isna(mark):
            return np.nan
        mark_str = str(mark).strip().lower()
        if mark_str in ['5', '4', '3', '2']:
            return int(mark_str)
        elif mark_str in ['з', 'зач']:
            return 5  # зачет = 5
        elif mark_str in ['ня', 'неявка']:
            return 0  # неявка = 0
        else:
            return np.nan
    
    df['mark_numeric'] = df['MARK'].apply(parse_mark)
    
    # BALLS: числовые баллы
    df['balls'] = pd.to_numeric(df['BALLS'], errors='coerce')
    
    # GRADE: буквенные оценки (A, B, C, etc.)
    # Преобразуем в числовую шкалу
    grade_mapping = {
        'A': 5, 'A-': 4.7, 'A+': 5,
        'B': 4, 'B-': 3.7, 'B+': 4.3,
        'C': 3, 'C-': 2.7, 'C+': 3.3,
        'D': 2, 'D-': 1.7, 'D+': 2.3,
        'F': 1
    }
    df['grade_numeric'] = df['GRADE'].map(grade_mapping)
    
    # Используем лучшую доступную оценку
    df['final_score'] = df['balls'].fillna(df['grade_numeric']).fillna(df['mark_numeric'])
    
    # Обработка типа оценки
    log_info("Обработка типов оценок...")
    
    # Определяем название колонки с типом (может быть 'TYPE' или уже переименована)
    type_col = None
    if 'TYPE' in df.columns:
        type_col = 'TYPE'
    elif 'exam_type' in df.columns:
        type_col = 'exam_type'
    else:
        log_info("  ⚠️  Колонка с типом оценки не найдена, пропускаем обработку типов")
        type_col = None
    
    if type_col:
        df[type_col] = df[type_col].astype(str).str.strip().str.lower()
        
        # Создаем бинарные признаки для типов
        df['is_exam'] = (df[type_col] == 'экз').astype(int)
        df['is_credit'] = (df[type_col] == 'зач').astype(int)
        df['is_practice'] = (df[type_col] == 'прак').astype(int)
        df['is_coursework'] = (df[type_col] == 'кп').astype(int)
    else:
        # Если колонки нет, создаем нулевые признаки
        df['is_exam'] = 0
        df['is_credit'] = 0
        df['is_practice'] = 0
        df['is_coursework'] = 0
    
    # Обработка пропусков
    log_info("Обработка пропусков...")
    missing_before = df.isnull().sum().sum()
    
    # Для числовых признаков заполняем медианой
    numeric_cols = ['final_score', 'balls', 'mark_numeric', 'grade_numeric']
    for col in numeric_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    
    missing_after = df.isnull().sum().sum()
    log_info(f"  Пропусков до обработки: {missing_before:,}")
    log_info(f"  Пропусков после обработки: {missing_after:,}")
    
    # Переименование колонок для удобства (только если они существуют)
    rename_dict = {}
    if 'PK' in df.columns:
        rename_dict['PK'] = 'student_id'
    if 'SEMESTER' in df.columns:
        rename_dict['SEMESTER'] = 'semester'
    if 'DNAME' in df.columns:
        rename_dict['DNAME'] = 'discipline_name'
    if 'TYPE' in df.columns:
        rename_dict['TYPE'] = 'exam_type'
    
    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)
    
    log_info("\n✅ Предобработка завершена!")
    log_info(f"  Итоговых колонок: {len(df.columns)}")
    log_info(f"  Итоговых записей: {len(df):,}")
    
    return df

if __name__ == "__main__":
    # Загрузка данных
    data_df = load_dataframe("data_raw.parquet")
    
    # Предобработка
    processed_df = preprocess_data(data_df)
    
    # Сохранение
    save_dataframe(processed_df, "data_processed.parquet")
    
    log_info("\n✅ Предобработка данных завершена!")

