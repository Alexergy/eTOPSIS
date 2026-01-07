"""
Главный скрипт запуска расширенного метода TOPSIS
"""

# Импорт необходимых для работы библиотек
import json
import os
import argparse  # модуль для работы с аргументами командной строки
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from openpyxl.styles import Alignment, Font

from extended_topsis import ExtendedTOPSIS
from table_generator import TablesGenerator
    
def main():
    # 1. Настройка парсера аргументов командной строки
    parser = argparse.ArgumentParser(description='Запуск расширенного метода TOPSIS для разных примеров.')
    parser.add_argument('example',
                        type=str,
                        nargs='?',          # Делаем аргумент необязательным (по умолчанию будет 'example1')
                        default='example1', # Значение по умолчанию
                        help='Название примера для запуска (например: example1, example2, example3)')
    
    args = parser.parse_args()
    file_name = args.example

    # Создание папки для сохранения результатов
    os.makedirs(f'research/analysis/{file_name}_solve', exist_ok=True)  # Создаст папку, если её нет
    
    # Пример использования
    print(f"=== ЗАПУСК РАСЧЕТА ДЛЯ ПРИМЕРА: {file_name} ===")
    solver = ExtendedTOPSIS(f'research/testing_data/{file_name}.json')
    results = solver.solve()
   
    # Сохранение результатов в JSON
    with open(f'research/analysis/{file_name}_solve/{file_name}_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Результаты расчета сохранены в 'research/analysis/{file_name}_solve/{file_name}_results.json'")

    # 2. Генерация таблиц
    table_generator = TablesGenerator(results)

    # Печать таблиц в консоль
    table_generator.print_all_tables()

    # Сохранение таблиц в Excel
    table_generator.save_all_tables_to_excel(f'research/analysis/{file_name}_solve/{file_name}_all_tables.xlsx')
    print(f"Все таблицы сохранены в 'research/analysis/{file_name}_solve/{file_name}_all_tables.xlsx'")

    print("\n" + "="*80)
    print("РАСЧЕТ УСПЕШНО ЗАВЕРШЕН!")
    print("="*80)
    
if __name__ == "__main__":
    main()
