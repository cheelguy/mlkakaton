import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import log_info, save_dataframe, load_dataframe, check_data_leakage
from src.load_data import load_raw_data, analyze_data
from src.preprocess import preprocess_data
from src.feature_engineering import create_student_features, merge_with_marking

def step1_analyze_data_csv(data_df):
    log_info("\n" + "=" * 60)
    log_info("АНАЛИЗ data.csv")
    log_info("=" * 60)
    
    log_info(f"\n Основная статистика:")
    log_info(f"  Всего записей: {len(data_df):,}")
    log_info(f"  Уникальных студентов: {data_df['PK'].nunique():,}")
    log_info(f"  Уникальных семестров: {sorted(data_df['SEMESTER'].unique())}")
    log_info(f"  Уникальных дисциплин: {data_df['DNAME'].nunique():,}")
    
    log_info(f"\n Распределение по семестрам:")
    semester_dist = data_df['SEMESTER'].value_counts().sort_index()
    for sem, count in semester_dist.items():
        pct = count / len(data_df) * 100
        log_info(f"  Семестр {sem}: {count:,} ({pct:.1f}%)")
    
    log_info(f"\n Типы оценок:")
    type_dist = data_df['TYPE'].value_counts()
    for typ, count in type_dist.items():
        pct = count / len(data_df) * 100
        log_info(f"  {typ}: {count:,} ({pct:.1f}%)")
    
    return data_df

def step2_join_marking(data_df, marking_df):
    log_info("\n" + "=" * 60)
    log_info("ПРИСОЕДИНЕНИЕ marking.csv")
    log_info("=" * 60)
    
    # Переименовываем для удобства
    marking_renamed = marking_df.rename(columns={'ИД': 'PK'})
    
    # Берем актуальную запись для каждого студента 
    if 'дата изменения' in marking_renamed.columns:
        marking_latest = marking_renamed.sort_values('дата изменения', ascending=False).drop_duplicates(subset=['PK'], keep='first')
    else:
        marking_latest = marking_renamed.drop_duplicates(subset=['PK'], keep='first')
    
    log_info(f"  Записей в marking: {len(marking_df):,}")
    log_info(f"  Уникальных студентов в marking: {marking_latest['PK'].nunique():,}")
    
    # Проверяем пересечение
    students_in_data = set(data_df['PK'].unique())
    students_in_marking = set(marking_latest['PK'].unique())
    common = students_in_data & students_in_marking
    
    log_info(f"\n  Студентов в data.csv: {len(students_in_data):,}")
    log_info(f"  Студентов в marking.csv: {len(students_in_marking):,}")
    log_info(f"  Общих студентов: {len(common):,}")
    
    # Сохраняем marking отдельно для дальнейшего использования
    save_dataframe(marking_latest, "marking_processed.parquet")
    
    return marking_latest

def step3_remove_test_students(data_df, marking_df, sample_submission_df):
    log_info("\n" + "=" * 60)
    log_info("УДАЛЕНИЕ СТУДЕНТОВ ТЕСТА")
    log_info("=" * 60)
    
    # Получаем ID студентов для теста
    test_student_ids = set(sample_submission_df['ID'].unique())
    log_info(f"  Студентов в тесте: {len(test_student_ids):,}")
    
    # Студенты с целевой переменной (для train)
    marking_with_target = marking_df[marking_df['выпуск'].notna()].copy()
    train_student_ids = set(marking_with_target['PK'].unique())
    
    # Исключаем студентов из теста
    train_student_ids = train_student_ids - test_student_ids
    log_info(f"  Студентов с целевой переменной: {len(marking_with_target['PK'].unique()):,}")
    log_info(f"  Студентов для обучения (после исключения теста): {len(train_student_ids):,}")
    
    # Фильтруем данные: только семестры 1-4 для train студентов
    train_data = data_df[
        (data_df['PK'].isin(train_student_ids)) & 
        (data_df['SEMESTER'] <= 4)
    ].copy()
    
    log_info(f"  Записей для обучения: {len(train_data):,}")
    log_info(f"  Уникальных студентов в train данных: {train_data['PK'].nunique():,}")
    
    return train_data, train_student_ids, test_student_ids

def step4_check_missing_data(data_df):
    log_info("\n" + "=" * 60)
    log_info("ПРОВЕРКА ОТСУТСТВУЮЩИХ ДАННЫХ")
    log_info("=" * 60)
    
    missing_summary = {}
    
    for col in data_df.columns:
        missing_count = data_df[col].isnull().sum()
        if missing_count > 0:
            missing_pct = missing_count / len(data_df) * 100
            missing_summary[col] = {
                'count': missing_count,
                'percentage': missing_pct
            }
            log_info(f"  {col}: {missing_count:,} ({missing_pct:.2f}%)")
    
    if not missing_summary:
        log_info(" Пропусков не обнаружено!")
    else:
        log_info(f"\n  Всего колонок с пропусками: {len(missing_summary)}")
    
    return missing_summary

def step5_build_safe_features_list(data_df, marking_df):
    log_info("\n" + "=" * 60)
    log_info("ПОСТРОЕНИЕ СПИСКА БЕЗОПАСНЫХ ФИЧ")
    log_info("=" * 60)
    
    # Проверяем, что используем только семестры 1-4
    max_semester = data_df['SEMESTER'].max()
    min_semester = data_df['SEMESTER'].min()
    
    log_info(f"  Диапазон семестров в данных: {min_semester} - {max_semester}")
    
    if max_semester > 4:
        log_info(f"  ⚠️  ВНИМАНИЕ: Есть данные за семестры > 4, они будут исключены!")
        data_safe = data_df[data_df['SEMESTER'] <= 4].copy()
    else:
        data_safe = data_df.copy()
    
    log_info(f"  Безопасных записей (семестры 1-4): {len(data_safe):,}")
    
    # Безопасный список, который можно вычислить из семестров 1-4)
    safe_features_categories = [
        "Статистики по оценкам (среднее, медиана, мин, макс, std)",
        "Динамика оценок по семестрам 1-4",
        "Статистики по типам оценок (экзамены, зачеты, практики)",
        "Количество дисциплин",
        "Процент успешных оценок",
        "Информация о студенте (факультет, направление, год поступления)"
    ]
    
    log_info(f"\n  Категории безопасных признаков:")
    for i, cat in enumerate(safe_features_categories, 1):
        log_info(f"    {i}. {cat}")
    
    return data_safe

def step6_build_features(data_df, marking_df, student_ids=None):
    log_info("\n" + "=" * 60)
    log_info("ПОСТРОЕНИЕ ПРИЗНАКОВ")
    log_info("=" * 60)
    
    # Предобработка данных
    log_info("\n  Предобработка данных...")
    processed_df = preprocess_data(data_df)
    
    # Создание признаков
    log_info("\n  Создание признаков...")
    features_df = create_student_features(processed_df, student_ids)
    
    # Объединение с marking
    log_info("\n  Объединение с marking...")
    features_df = merge_with_marking(features_df, marking_df)
    
    log_info(f"\n  Итоговое количество признаков: {len(features_df.columns) - 1}")
    log_info(f"  Итоговое количество студентов: {len(features_df):,}")
    
    return features_df, processed_df

def step7_check_leakage(train_features, test_features):
    log_info("\n" + "=" * 60)
    log_info("ПРОВЕРКА НА УТЕЧКИ")
    log_info("=" * 60)
    
    # Исключаем не-признаки
    exclude_cols = ['student_id', 'Факультет', 'Направление', 'год поступления', 'выпуск', 'target']
    feature_cols = [col for col in train_features.columns if col not in exclude_cols]
    
    log_info(f"  Проверка {len(feature_cols)} признаков...")
    
    warnings = check_data_leakage(train_features, test_features, feature_cols)
    
    if warnings:
        log_info(f"\n  Найдено {len(warnings)} предупреждений о возможных утечках!")
        log_info("  Рекомендуется проверить эти признаки вручную.")
    else:
        log_info(f"\n  Все признаки прошли проверку на утечки!")
    
    return warnings, feature_cols

def step8_prepare_final_dataset(train_features, test_features, train_target, feature_cols):
    log_info("\n" + "=" * 60)
    log_info("ШАГ 8: ПОДГОТОВКА ФИНАЛЬНОГО ДАТАСЕТА")
    log_info("=" * 60)
    
    # Подготовка train датасета
    train_final = train_features[['student_id'] + feature_cols].copy()
    train_final = train_final.merge(
        train_target[['ИД', 'target']].rename(columns={'ИД': 'student_id'}),
        on='student_id',
        how='inner'
    )
    
    # Подготовка test датасета
    test_final = test_features[['student_id'] + feature_cols].copy()
    
    log_info(f"\n  Train датасет:")
    log_info(f"    Студентов: {len(train_final):,}")
    log_info(f"    Признаков: {len(feature_cols)}")
    log_info(f"    Баланс классов: {train_final['target'].value_counts().to_dict()}")
    
    log_info(f"\n  Test датасет:")
    log_info(f"    Студентов: {len(test_final):,}")
    log_info(f"    Признаков: {len(feature_cols)}")
    
    # Сохранение финальных датасетов
    save_dataframe(train_final, "train_final.parquet")
    save_dataframe(test_final, "test_final.parquet")
    
    # Сохранение списка признаков
    import json
    features_info = {
        'feature_names': feature_cols,
        'n_features': len(feature_cols),
        'n_train': len(train_final),
        'n_test': len(test_final)
    }
    with open(Path(__file__).parent.parent / "output" / "features_info.json", 'w') as f:
        json.dump(features_info, f, indent=2)
    
    log_info(f"\n  Финальные датасеты сохранены:")
    log_info(f"    - data/processed/train_final.parquet")
    log_info(f"    - data/processed/test_final.parquet")
    log_info(f"    - output/features_info.json")
    
    log_info("\n" + "=" * 60)
    log_info("ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА!")
    log_info("   Данные готовы (Models + Validation)")
    log_info("=" * 60)
    
    return train_final, test_final

def run_full_data_preparation():
    log_info("\n" + "=" * 80)
    log_info("ПОЛНАЯ ПОДГОТОВКА ДАННЫХ")
    log_info("=" * 80)
    
    # Загрузка данных
    data_df, marking_df, sample_submission_df = load_raw_data()
    
    # Анализ data.csv
    data_df = step1_analyze_data_csv(data_df)
    
    # Присоединение marking.csv
    marking_processed = step2_join_marking(data_df, marking_df)
    
    # Удаление студентов теста
    train_data, train_student_ids, test_student_ids = step3_remove_test_students(
        data_df, marking_processed, sample_submission_df
    )
    
    # Проверка отсутствующих данных
    missing_summary = step4_check_missing_data(train_data)
    
    # Список безопасных фич
    data_safe = step5_build_safe_features_list(train_data, marking_processed)
    
    # Построение признаков
    train_features, processed_df = step6_build_features(
        data_safe, marking_processed, train_student_ids
    )
    
    # Создаем признаки для теста тоже
    test_data = data_df[
        (data_df['PK'].isin(test_student_ids)) & 
        (data_df['SEMESTER'] <= 4)
    ].copy()
    test_processed = preprocess_data(test_data)
    test_features, _ = step6_build_features(
        test_processed, marking_processed, test_student_ids
    )
    
    # Проверка на утечки
    warnings, feature_cols = step7_check_leakage(train_features, test_features)
    
    # Финальный датасет
    train_target = marking_processed[
        marking_processed['PK'].isin(train_student_ids)
    ][['PK', 'выпуск']].copy()
    train_target['target'] = (train_target['выпуск'] == 'выпустился').astype(int)
    train_target = train_target.rename(columns={'PK': 'student_id'})
    
    # Переименовываем для совместимости с функцией
    train_target_for_func = train_target.rename(columns={'student_id': 'ИД'})
    
    train_final, test_final = step8_prepare_final_dataset(
        train_features, test_features, train_target_for_func, feature_cols
    )
    
    return train_final, test_final, feature_cols

if __name__ == "__main__":
    train_final, test_final, feature_cols = run_full_data_preparation()
    
    log_info("\n Все шаги подготовки данных выполнены!")
    log_info("   Теперь можно переходить к обучению моделей ")

