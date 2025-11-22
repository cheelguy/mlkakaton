"""
Разделение данных на train/test по логике "до/после 2 курса"

Логика:
- Train: студенты с данными за семестры 1-4 (первые 2 курса) + есть целевая переменная
- Test: студенты из sample_submission.csv (для них нужно предсказать результат)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import log_info, save_dataframe, load_dataframe

def split_train_test(data_df, marking_df, sample_submission_df, max_semester_train=4):
    """
    Разделение данных на train и test
    
    Args:
        data_df: DataFrame с данными об оценках
        marking_df: DataFrame с информацией о студентах и целевой переменной
        sample_submission_df: DataFrame с ID студентов для теста
        max_semester_train: максимальный семестр для train (по умолчанию 4 = конец 2 курса)
    
    Returns:
        train_data: данные для обучения (только семестры 1-4)
        train_target: целевая переменная для train
        test_data: данные для теста (только семестры 1-4 для студентов из sample_submission)
        test_ids: ID студентов для теста
    """
    log_info("\n" + "=" * 60)
    log_info("РАЗДЕЛЕНИЕ НА TRAIN/TEST")
    log_info("=" * 60)
    
    # Получаем ID студентов для теста
    test_student_ids = set(sample_submission_df['ID'].unique())
    log_info(f"Студентов в тестовой выборке: {len(test_student_ids):,}")
    
    # Получаем студентов с целевой переменной (для train)
    marking_with_target = marking_df[marking_df['выпуск'].notna()].copy()
    
    # Преобразуем целевую переменную в бинарную
    # "выпустился" -> 1, "отчислен" -> 0
    marking_with_target['target'] = (marking_with_target['выпуск'] == 'выпустился').astype(int)
    
    train_student_ids = set(marking_with_target['ИД'].unique())
    # Исключаем студентов из теста из train
    train_student_ids = train_student_ids - test_student_ids
    log_info(f"Студентов в обучающей выборке: {len(train_student_ids):,}")
    
    # Разделяем данные по семестрам
    log_info(f"\nФильтрация данных: только семестры 1-{max_semester_train} (первые 2 курса)")
    
    # Train: данные за семестры 1-4 для студентов из train
    train_data = data_df[
        (data_df['PK'].isin(train_student_ids)) & 
        (data_df['SEMESTER'] <= max_semester_train)
    ].copy()
    
    log_info(f"  Train данных: {len(train_data):,} записей")
    log_info(f"  Train студентов: {train_data['PK'].nunique():,}")
    
    # Test: данные за семестры 1-4 для студентов из теста
    test_data = data_df[
        (data_df['PK'].isin(test_student_ids)) & 
        (data_df['SEMESTER'] <= max_semester_train)
    ].copy()
    
    log_info(f"  Test данных: {len(test_data):,} записей")
    log_info(f"  Test студентов: {test_data['PK'].nunique():,}")
    
    # Получаем целевую переменную для train
    train_target = marking_with_target[
        marking_with_target['ИД'].isin(train_student_ids)
    ][['ИД', 'target']].drop_duplicates(subset=['ИД'])
    
    # Проверяем, что у всех студентов из train есть целевая переменная
    train_students_in_data = set(train_data['PK'].unique())
    train_students_with_target = set(train_target['ИД'].unique())
    missing_target = train_students_in_data - train_students_with_target
    
    if missing_target:
        log_info(f"⚠️  ВНИМАНИЕ: {len(missing_target)} студентов в train данных не имеют целевой переменной")
        # Оставляем только студентов с целевой переменной
        train_data = train_data[train_data['PK'].isin(train_students_with_target)]
        log_info(f"  После фильтрации: {len(train_data):,} записей, {train_data['PK'].nunique():,} студентов")
    
    # Проверяем баланс классов
    log_info(f"\n📊 Баланс классов в train:")
    target_dist = train_target['target'].value_counts().sort_index()
    for target_val, count in target_dist.items():
        pct = count / len(train_target) * 100
        label = "выпустился" if target_val == 1 else "отчислен"
        log_info(f"  {label} (target={target_val}): {count:,} ({pct:.1f}%)")
    
    # Проверяем покрытие семестрами
    log_info(f"\n📊 Покрытие семестрами в train:")
    for sem in range(1, max_semester_train + 1):
        students_in_sem = train_data[train_data['SEMESTER'] == sem]['PK'].nunique()
        log_info(f"  Семестр {sem}: {students_in_sem:,} студентов")
    
    log_info(f"\n📊 Покрытие семестрами в test:")
    for sem in range(1, max_semester_train + 1):
        students_in_sem = test_data[test_data['SEMESTER'] == sem]['PK'].nunique()
        log_info(f"  Семестр {sem}: {students_in_sem:,} студентов")
    
    # Создаем словарь target для быстрого доступа
    target_dict = dict(zip(train_target['ИД'], train_target['target']))
    
    return {
        'train_data': train_data,
        'train_target': train_target,
        'target_dict': target_dict,
        'test_data': test_data,
        'test_ids': list(test_student_ids),
        'train_student_ids': list(train_student_ids)
    }

if __name__ == "__main__":
    # Загрузка данных
    data_df = load_dataframe("data_raw.parquet")
    marking_df = load_dataframe("marking_raw.parquet")
    sample_submission_df = load_dataframe("sample_submission.parquet")
    
    # Разделение
    splits = split_train_test(data_df, marking_df, sample_submission_df, max_semester_train=4)
    
    # Сохранение
    save_dataframe(splits['train_data'], "train_data.parquet")
    save_dataframe(splits['train_target'], "train_target.parquet")
    save_dataframe(splits['test_data'], "test_data.parquet")
    
    # Сохранение метаданных
    import json
    metadata = {
        'train_students': len(splits['train_student_ids']),
        'test_students': len(splits['test_ids']),
        'max_semester_train': 4
    }
    with open(Path(__file__).parent.parent / "output" / "split_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    log_info("\n✅ Разделение данных завершено!")

