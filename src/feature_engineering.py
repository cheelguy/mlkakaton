import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import log_info, save_dataframe, load_dataframe

def create_student_features(data_df, student_ids=None):
    log_info("\n" + "=" * 60)
    log_info("СОЗДАНИЕ ПРИЗНАКОВ")
    log_info("=" * 60)
    
    if student_ids is not None:
        data_df = data_df[data_df['student_id'].isin(student_ids)].copy()
    
    features_list = []
    
    # Группируем по студентам
    grouped = data_df.groupby('student_id')
    log_info(f"Обработка {len(grouped)} студентов...")
    
    for student_id, student_data in grouped:
        features = {'student_id': student_id}
        
        # Базовые статистики по оценкам
        if 'final_score' in student_data.columns:
            scores = student_data['final_score'].dropna()
            if len(scores) > 0:
                features['score_mean'] = scores.mean()
                features['score_median'] = scores.median()
                features['score_min'] = scores.min()
                features['score_max'] = scores.max()
                features['score_std'] = scores.std() if len(scores) > 1 else 0
                features['score_count'] = len(scores)
                features['score_sum'] = scores.sum()
            else:
                features.update({
                    'score_mean': 0, 'score_median': 0, 'score_min': 0,
                    'score_max': 0, 'score_std': 0, 'score_count': 0, 'score_sum': 0
                })
        
        # Статистики по семестрам
        if 'semester' in student_data.columns:
            semesters = sorted(student_data['semester'].unique())
            features['semester_count'] = len(semesters)
            features['semester_min'] = min(semesters) if semesters else 0
            features['semester_max'] = max(semesters) if semesters else 0
            
            # Средние оценки по семестрам
            semester_scores = []
            for sem in semesters:
                sem_data = student_data[student_data['semester'] == sem]
                if 'final_score' in sem_data.columns:
                    sem_scores = sem_data['final_score'].dropna()
                    if len(sem_scores) > 0:
                        semester_scores.append(sem_scores.mean())
            
            if len(semester_scores) > 0:
                features['semester_score_mean'] = np.mean(semester_scores)
                # Динамика: улучшение или ухудшение
                if len(semester_scores) >= 2:
                    features['score_trend'] = semester_scores[-1] - semester_scores[0]
                    features['score_improvement'] = 1 if semester_scores[-1] > semester_scores[0] else 0
                else:
                    features['score_trend'] = 0
                    features['score_improvement'] = 0
            else:
                features['semester_score_mean'] = 0
                features['score_trend'] = 0
                features['score_improvement'] = 0
        
        # Статистики по типам оценок
        if 'is_exam' in student_data.columns:
            features['exams_count'] = student_data['is_exam'].sum()
            features['credits_count'] = student_data['is_credit'].sum()
            features['practices_count'] = student_data['is_practice'].sum()
            features['courseworks_count'] = student_data['is_coursework'].sum()
            
            # Средние оценки по типам
            if features['exams_count'] > 0:
                exam_scores = student_data[student_data['is_exam'] == 1]['final_score'].dropna()
                features['exam_score_mean'] = exam_scores.mean() if len(exam_scores) > 0 else 0
            else:
                features['exam_score_mean'] = 0
            
            if features['credits_count'] > 0:
                credit_scores = student_data[student_data['is_credit'] == 1]['final_score'].dropna()
                features['credit_score_mean'] = credit_scores.mean() if len(credit_scores) > 0 else 0
            else:
                features['credit_score_mean'] = 0
        
        # Статистики по дисциплинам
        if 'discipline_name' in student_data.columns:
            features['disciplines_count'] = student_data['discipline_name'].nunique()
            features['records_count'] = len(student_data)
        
        # Статистики по датам
        if 'year' in student_data.columns:
            years = student_data['year'].dropna()
            if len(years) > 0:
                features['year_min'] = years.min()
                features['year_max'] = years.max()
                features['year_span'] = years.max() - years.min()
            else:
                features['year_min'] = 0
                features['year_max'] = 0
                features['year_span'] = 0
        
        # % по успешным оценкам
        if 'final_score' in student_data.columns:
            scores = student_data['final_score'].dropna()
            if len(scores) > 0:
                # Оценка >= 3 считается успешной
                features['success_rate'] = (scores >= 3).sum() / len(scores)
                features['excellent_rate'] = (scores >= 4.5).sum() / len(scores)
                features['fail_rate'] = (scores < 3).sum() / len(scores)
            else:
                features['success_rate'] = 0
                features['excellent_rate'] = 0
                features['fail_rate'] = 0
        
        # Динамики по семестрам
        if 'semester' in student_data.columns and 'final_score' in student_data.columns:
            # Средние оценки по каждому семестру
            for sem in [1, 2, 3, 4]:
                sem_data = student_data[student_data['semester'] == sem]
                if len(sem_data) > 0:
                    sem_scores = sem_data['final_score'].dropna()
                    features[f'semester_{sem}_score_mean'] = sem_scores.mean() if len(sem_scores) > 0 else 0
                    features[f'semester_{sem}_count'] = len(sem_data)
                else:
                    features[f'semester_{sem}_score_mean'] = 0
                    features[f'semester_{sem}_count'] = 0
        
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    # Заполняем пропуски нулями
    features_df = features_df.fillna(0)
    
    log_info(f"Создано признаков: {len(features_df.columns) - 1}")  # -1 для student_id
    log_info(f"Создано записей: {len(features_df):,}")
    
    return features_df

def merge_with_marking(features_df, marking_df):
    log_info("\nОбъединение с marking.csv...")
    
    # Определяем какая колонка используется для ID студента
    marking_df = marking_df.copy()
    id_col = None
    if 'PK' in marking_df.columns:
        id_col = 'PK'
    elif 'ИД' in marking_df.columns:
        id_col = 'ИД'
    else:
        raise ValueError("Не найдена колонка с ID студента (ожидается 'PK' или 'ИД')")
    
    # Самая актуальная версия для каждого студента 
    if 'дата изменения' in marking_df.columns:
        marking_latest = marking_df.sort_values('дата изменения', ascending=False).drop_duplicates(subset=[id_col], keep='first')
    else:
        marking_latest = marking_df.drop_duplicates(subset=[id_col], keep='first')
    
    # Переименовываем колонки для объединения
    marking_latest = marking_latest.rename(columns={id_col: 'student_id'})
    
    # Объединяем
    merged_df = features_df.merge(
        marking_latest[['student_id', 'Факультет', 'Направление', 'год поступления']],
        on='student_id',
        how='left'
    )
    
    # Кодируем категориальные признаки
    if 'Факультет' in merged_df.columns:
        merged_df['faculty_encoded'] = pd.Categorical(merged_df['Факультет']).codes
        merged_df['faculty_encoded'] = merged_df['faculty_encoded'].fillna(-1).astype(int)
    
    if 'Направление' in merged_df.columns:
        merged_df['direction_encoded'] = pd.Categorical(merged_df['Направление']).codes
        merged_df['direction_encoded'] = merged_df['direction_encoded'].fillna(-1).astype(int)
    
    if 'год поступления' in merged_df.columns:
        merged_df['admission_year'] = pd.to_numeric(merged_df['год поступления'], errors='coerce')
        merged_df['admission_year'] = merged_df['admission_year'].fillna(merged_df['admission_year'].median())
    
    log_info(f" Объединено записей: {len(merged_df):,}")
    
    return merged_df

if __name__ == "__main__":
    # Загрузка данных
    processed_df = load_dataframe("data_processed.parquet")
    marking_df = load_dataframe("marking_raw.parquet")
    
    # Создание признаков для всех студентов
    features_df = create_student_features(processed_df)
    
    # Объединение с marking
    features_df = merge_with_marking(features_df, marking_df)
    
    # Сохранение
    save_dataframe(features_df, "features.parquet")
    
    log_info("\n Feature engineering завершен!")

