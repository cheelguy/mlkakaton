#!/usr/bin/env python3
"""
Скрипт проверки готовности проекта к запуску
"""
import sys
from pathlib import Path

def check_structure():
    """Проверка структуры проекта"""
    print("=" * 60)
    print("ПРОВЕРКА СТРУКТУРЫ ПРОЕКТА")
    print("=" * 60)
    
    required_dirs = ['src', 'data/processed', 'models', 'output']
    all_ok = True
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ - НЕ НАЙДЕНА")
            all_ok = False
    
    return all_ok

def check_files():
    """Проверка необходимых файлов"""
    print("\n" + "=" * 60)
    print("ПРОВЕРКА ФАЙЛОВ")
    print("=" * 60)
    
    required_files = [
        'src/data_preparation.py',
        'src/baseline.py',
        'src/model_catboost.py',
        'src/inference.py',
        'requirements.txt'
    ]
    all_ok = True
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - НЕ НАЙДЕН")
            all_ok = False
    
    return all_ok

def check_data():
    """Проверка наличия данных"""
    print("\n" + "=" * 60)
    print("ПРОВЕРКА ДАННЫХ")
    print("=" * 60)
    
    data_files = [
        'data/raw:/data.csv',
        'data/raw:/marking.csv',
        'data/raw:/sample_submission.csv'
    ]
    all_ok = True
    
    for data_file in data_files:
        path = Path(data_file)
        if path.exists():
            size = path.stat().st_size / 1024  # KB
            print(f"  ✅ {data_file} ({size:.1f} KB)")
        else:
            alt_path = Path(data_file.replace('raw:', 'raw'))
            if alt_path.exists():
                size = alt_path.stat().st_size / 1024
                print(f"  ✅ {data_file} (альтернативный путь, {size:.1f} KB)")
            else:
                print(f"  ❌ {data_file} - НЕ НАЙДЕН")
                all_ok = False
    
    return all_ok

def check_dependencies():
    """Проверка установленных зависимостей"""
    print("\n" + "=" * 60)
    print("ПРОВЕРКА ЗАВИСИМОСТЕЙ")
    print("=" * 60)
    
    dependencies = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'catboost': 'catboost',
        'pyarrow': 'pyarrow',
        'joblib': 'joblib'
    }
    
    all_ok = True
    for module_name, package_name in dependencies.items():
        try:
            if module_name == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                mod = __import__(module_name)
                version = mod.__version__
            print(f"  ✅ {package_name}: {version}")
        except ImportError:
            print(f"  ❌ {package_name} - НЕ УСТАНОВЛЕН")
            all_ok = False
    
    # Проверка LightGBM (опционально)
    try:
        import lightgbm
        print(f"  ✅ lightgbm: {lightgbm.__version__}")
    except ImportError:
        print(f"  ⚠️  lightgbm - не установлен (опционально)")
    except OSError:
        print(f"  ⚠️  lightgbm - требует libomp (brew install libomp)")
    
    return all_ok

def check_processed_data():
    """Проверка обработанных данных"""
    print("\n" + "=" * 60)
    print("ПРОВЕРКА ОБРАБОТАННЫХ ДАННЫХ")
    print("=" * 60)
    
    processed_files = [
        'data/processed/train_final.parquet',
        'data/processed/test_final.parquet'
    ]
    
    all_exist = True
    for file_path in processed_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size / 1024  # KB
            print(f"  ✅ {file_path} ({size:.1f} KB)")
        else:
            print(f"  ⚠️  {file_path} - не найден (запустите data_preparation.py)")
            all_exist = False
    
    return all_exist

def main():
    """Главная функция"""
    print("\n" + "=" * 60)
    print("🚀 ПРОВЕРКА ГОТОВНОСТИ ПРОЕКТА")
    print("=" * 60 + "\n")
    
    checks = [
        ("Структура проекта", check_structure),
        ("Файлы проекта", check_files),
        ("Исходные данные", check_data),
        ("Зависимости", check_dependencies),
        ("Обработанные данные", check_processed_data)
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    # Итог
    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ СТАТУС")
    print("=" * 60)
    
    all_ok = True
    for name, result in results.items():
        status = "✅ ГОТОВО" if result else "❌ ТРЕБУЕТ ВНИМАНИЯ"
        print(f"  {name}: {status}")
        if not result and name != "Обработанные данные":
            all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ ПРОЕКТ ГОТОВ К ЗАПУСКУ!")
        print("\nСледующий шаг:")
        print("  python3 src/data_preparation.py")
    else:
        print("⚠️  ЕСТЬ ПРОБЛЕМЫ - ИСПРАВЬТЕ ИХ ПЕРЕД ЗАПУСКОМ")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()

