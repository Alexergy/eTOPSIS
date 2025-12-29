"""
Главный скрипт запуска расширенного метода TOPSIS
"""

# Импорт необходимых для работы библиотек
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from openpyxl.styles import Alignment, Font
    
def main():
    # Название файла с примером
    file_name = "example1"
  
    # Пример использования
    solver = ExtendedTOPSIS(f'data/{file_name}.json')
    results = solver.solve()
   
    # Сохранение результатов в JSON
    with open(f'{file_name}_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 2. Генерация таблиц
    table_generator = TablesGenerator(results)

    # Печать таблиц в консоль
    table_generator.print_all_tables()

    # Сохранение таблиц в Excel
    table_generator.save_all_tables_to_excel(f'{file_name}_all_tables.xlsx')
