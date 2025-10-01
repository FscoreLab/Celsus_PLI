"""
FVLM Package

Делаем импорты lavis доступными как из fvlm.lavis, так и просто из lavis
для обратной совместимости.
"""

import sys

# Создаем алиас модуля: когда код пытается сделать 'import lavis' или 'from lavis...',
# Python будет использовать 'fvlm.lavis' вместо этого

class LavisModuleAlias:
    """Прокси-модуль, который перенаправляет все импорты на fvlm.lavis"""
    
    def __getattr__(self, name):
        import importlib
        # Динамически импортируем из fvlm.lavis
        try:
            module = importlib.import_module(f'fvlm.lavis.{name}')
            return module
        except ImportError:
            # Пробуем импортировать сам fvlm.lavis
            module = importlib.import_module('fvlm.lavis')
            return getattr(module, name)

# Регистрируем алиас только если 'lavis' еще не существует
if 'lavis' not in sys.modules:
    sys.modules['lavis'] = LavisModuleAlias()
