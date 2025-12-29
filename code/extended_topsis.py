"""
Класс ExtendedTOPSIS - реализация расширенного метода TOPSIS
"""
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from openpyxl.styles import Alignment, Font

class ExtendedTOPSIS:
    def __init__(self, json_file: str):
        """Инициализация с загрузкой данных из JSON файла"""
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.alternatives = self.data['alternatives']
        self.criteria = self.data['criteria']
        self.dms = self.data['dms']
        self.parameters = self.data.get('parameters', {})

        # Преобразование в numpy массивы
        self.X_list = [np.array(dm['scores']) for dm in self.dms]
        self.weights_list = [np.array(dm['weights']) for dm in self.dms]
        self.dm_ids = [dm['id'] for dm in self.dms]

        self.m = len(self.alternatives)  # количество альтернатив
        self.n = len(self.criteria)      # количество критериев
        self.t = len(self.dms)           # количество экспертов

        print(f"Загружено: {self.m} альтернатив, {self.n} критериев, {self.t} экспертов")

    def normalize_matrix(self, X: np.ndarray) -> np.ndarray:
        """Нормализация матрицы решений"""
        R = np.zeros_like(X, dtype=float)

        for j in range(self.n):
            criterion_type = self.criteria[j]['type']
            column = X[:, j]
            norm = np.sqrt(np.sum(column**2))

            if criterion_type == 'positive':
                # Benefit criterion
                R[:, j] = column / norm if norm != 0 else 0
            elif criterion_type == 'negative':
                # Cost criterion
                R[:, j] = 1 - (column / norm) if norm != 0 else 0
            else:
                raise ValueError(f"Unknown criterion type: {criterion_type}")

        return R

    def calculate_weighted_matrices(self) -> List[np.ndarray]:
        """Расчет взвешенных нормализованных матриц"""
        Y_list = []

        for k in range(self.t):
            # Нормализация
            R_k = self.normalize_matrix(self.X_list[k])
            # Взвешивание
            Y_k = R_k * self.weights_list[k]
            Y_list.append(Y_k)

        return Y_list

    def calculate_ideal_solutions(self, Y_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Расчет идеальных решений: PIS, L-NIS, R-NIS"""
        # PIS - средняя матрица
        Y_star = np.mean(Y_list, axis=0)

        # L-NIS - минимальная матрица
        Y_l = np.min(Y_list, axis=0)

        # R-NIS - максимальная матрига
        Y_r = np.max(Y_list, axis=0)

        return Y_star, Y_l, Y_r

    def calculate_distances(self, Y_list: List[np.ndarray], Y_star: np.ndarray,
                          Y_l: np.ndarray, Y_r: np.ndarray) -> Tuple[List[float], List[float], List[float]]:
        """Расчет расстояний до идеальных решений"""
        S_plus = []   # расстояния до PIS
        S_l_minus = [] # расстояния до L-NIS
        S_r_minus = [] # расстояния до R-NIS

        for Y_k in Y_list:
            # Евклидово расстояние между матрицами
            S_plus.append(np.sqrt(np.sum((Y_k - Y_star)**2)))
            S_l_minus.append(np.sqrt(np.sum((Y_k - Y_l)**2)))
            S_r_minus.append(np.sqrt(np.sum((Y_k - Y_r)**2)))

        return S_plus, S_l_minus, S_r_minus

    def calculate_dm_weights(self, S_plus: List[float], S_l_minus: List[float],
                           S_r_minus: List[float]) -> Tuple[List[float], List[float]]:
        """Расчет весов экспертов"""
        C = []  # коэффициенты близости
        lambda_weights = []  # веса экспертов

        for k in range(self.t):
            # Коэффициент близости
            numerator = S_l_minus[k] + S_r_minus[k]
            denominator = S_plus[k] + S_l_minus[k] + S_r_minus[k]
            C_k = numerator / denominator if denominator != 0 else 0
            C.append(C_k)

        # Нормализация к весам
        total_C = sum(C)
        for C_k in C:
            lambda_k = C_k / total_C if total_C != 0 else 1/self.t
            lambda_weights.append(lambda_k)

        return lambda_weights, C

    def rank_alternatives(self, Y_list: List[np.ndarray], lambda_weights: List[float]) -> pd.DataFrame:
        """Ранжирование альтернатив на основе весов экспертов"""
        # Агрегация группового решения
        Y_group = np.zeros_like(Y_list[0])

        for k in range(self.t):
            Y_group += lambda_weights[k] * Y_list[k]

        # Интеграция оценок по атрибутам
        alternative_scores = np.sum(Y_group, axis=1)

        # Создание DataFrame с результатами
        results = pd.DataFrame({
            'Alternative': self.alternatives,
            'Total_Score': alternative_scores
        })

        # Ранжирование
        results['Rank'] = results['Total_Score'].rank(ascending=False, method='dense').astype(int)
        results = results.sort_values('Rank')

        return results, Y_group

    def solve(self) -> Dict:
        """Основной метод решения"""
        print("=== РАСШИРЕННЫЙ МЕТОД TOPSIS ДЛЯ ГРУППОВОГО РЕШЕНИЯ ===")

        # Шаг 1: Расчет взвешенных матриц
        print("1. Расчет взвешенных нормализованных матриц...")
        Y_list = self.calculate_weighted_matrices()

        # Шаг 2: Расчет идеальных решений
        print("2. Расчет идеальных решений...")
        Y_star, Y_l, Y_r = self.calculate_ideal_solutions(Y_list)

        # Шаг 3: Расчет расстояний
        print("3. Расчет расстояний...")
        S_plus, S_l_minus, S_r_minus = self.calculate_distances(Y_list, Y_star, Y_l, Y_r)

        # Шаг 4: Расчет весов экспертов
        print("4. Расчет весов экспертов...")
        lambda_weights, C = self.calculate_dm_weights(S_plus, S_l_minus, S_r_minus)

        # Шаг 5: Ранжирование альтернатив
        print("5. Ранжирование альтернатив...")
        alternatives_ranking, Y_group = self.rank_alternatives(Y_list, lambda_weights)

        # Формирование результатов
        results = {
            'alternatives': self.alternatives,
            'criteria': self.criteria,
            'dms': self.dms,
            'X_list': [X.tolist() for X in self.X_list],
            'weights_list': [w.tolist() for w in self.weights_list],
            'dm_ids': self.dm_ids,
            'Y_list': [Y.tolist() for Y in Y_list],
            'Y_star': Y_star.tolist(),
            'Y_l': Y_l.tolist(),
            'Y_r': Y_r.tolist(),
            'S_plus': S_plus,
            'S_l_minus': S_l_minus,
            'S_r_minus': S_r_minus,
            'C': C,
            'lambda_weights': lambda_weights,
            'alternatives_ranking': alternatives_ranking.to_dict('records'),
            'Y_group': Y_group.tolist()
        }

        return results
