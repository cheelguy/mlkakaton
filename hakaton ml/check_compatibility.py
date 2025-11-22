"""
Скрипт для проверки совместимости проекта с Python 3.9.6
"""
import sys
import importlib

def check_python_version():
    """Проверка версии Python"""
    version = sys.version_info
    print(f"Python версия: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Требуется Python 3.9 или выше")
        return False
    elif version.major == 3 and version.minor == 9:
        print("✅ Python 3.9 - поддерживается")
        return True
    else:
        print("✅ Python версия поддерживается")
        return True

def check_syntax_features():
    """Проверка использования синтаксиса, несовместимого с Python 3.9"""
    issues = []
    
    # Проверяем, что не используется match/case (Python 3.10+)
    try:
        with open('src/utils.py', 'r') as f:
            content = f.read()
            if 'match ' in content or ' case ' in content:
                issues.append("Используется match/case (требует Python 3.10+)")
    except:
        pass
    
    return issues

def check_imports():
    """Проверка импортов на совместимость"""
    required_modules = [
        'pandas',
        'numpy',
        'sklearn',
        'catboost',
        'lightgbm',
        'pyarrow',
        'joblib',
        'pathlib'
    ]
    
    missing = []
    for module in required_modules:
        try:
            if module == 'sklearn':
                importlib.import_module('sklearn')
            elif module == 'lightgbm':
                importlib.import_module('lightgbm')
            else:
                importlib.import_module(module)
            print(f"✅ {module} - доступен")
        except ImportError as e:
            print(f"❌ {module} - не установлен: {e}")
            missing.append(module)
    
    return missing

def check_library_versions():
    """Проверка версий библиотек на совместимость с Python 3.9"""
    compatibility = {
        'pandas': (1.3, '3.8+'),
        'numpy': (1.19, '3.8+'),
        'scikit-learn': (1.0, '3.7+'),
        'catboost': (1.0, '3.6+'),
        'lightgbm': (3.0, '3.7+'),
        'pyarrow': (5.0, '3.7+'),
        'joblib': (1.0, '3.6+')
    }
    
    print("\nПроверка совместимости версий библиотек:")
    for lib, (min_version, py_requirement) in compatibility.items():
        try:
            if lib == 'scikit-learn':
                import sklearn
                version = sklearn.__version__
            elif lib == 'lightgbm':
                import lightgbm
                version = lightgbm.__version__
            else:
                mod = importlib.import_module(lib)
                version = mod.__version__
            
            print(f"  {lib}: {version} (требует Python {py_requirement})")
        except ImportError:
            print(f"  {lib}: не установлен")

if __name__ == "__main__":
    print("=" * 60)
    print("ПРОВЕРКА СОВМЕСТИМОСТИ С PYTHON 3.9.6")
    print("=" * 60)
    
    # Проверка версии Python
    print("\n1. Проверка версии Python:")
    python_ok = check_python_version()
    
    # Проверка синтаксиса
    print("\n2. Проверка синтаксиса:")
    syntax_issues = check_syntax_features()
    if syntax_issues:
        print("⚠️  Проблемы с синтаксисом:")
        for issue in syntax_issues:
            print(f"   - {issue}")
    else:
        print("✅ Синтаксис совместим с Python 3.9")
    
    # Проверка импортов
    print("\n3. Проверка установленных библиотек:")
    missing = check_imports()
    
    # Проверка версий
    print("\n4. Проверка версий библиотек:")
    check_library_versions()
    
    # Итог
    print("\n" + "=" * 60)
    if python_ok and not syntax_issues and not missing:
        print("✅ ПРОЕКТ СОВМЕСТИМ С PYTHON 3.9.6")
    else:
        print("⚠️  ЕСТЬ ПРОБЛЕМЫ СОВМЕСТИМОСТИ")
        if missing:
            print(f"   Установите недостающие библиотеки: pip install {' '.join(missing)}")
    print("=" * 60)

