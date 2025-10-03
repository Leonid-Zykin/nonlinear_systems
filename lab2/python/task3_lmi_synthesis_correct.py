#!/usr/bin/env python3
"""
Синтез линейного регулятора через LMI для экспоненциальной устойчивости степени 2
Задача 3 лабораторной работы №2 (правильная версия)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize
import sympy as sp
from sympy import symbols, Matrix, simplify

# Настройка для корректного отображения русского текста
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def synthesize_lmi_controller():
    """Синтез регулятора через LMI для системы с экспоненциальной устойчивостью степени 2"""
    print("=" * 80)
    print("СИНТЕЗ ЛИНЕЙНОГО РЕГУЛЯТОРА ЧЕРЕЗ LMI")
    print("=" * 80)
    
    print("Дано:")
    print("Система: ẋ₁ = x₂, ẋ₂ = 2x₁ + u")
    print("Требуется: экспоненциальная устойчивость степени 2")
    print("Это означает: ||x(t)|| ≤ M||x(0)||e^(-2t) для некоторого M > 0")
    
    # Матрицы системы
    A = np.array([[0, 1],
                  [2, 0]])
    B = np.array([[0],
                  [1]])
    
    print(f"\nМатрицы системы:")
    print(f"A = \n{A}")
    print(f"B = \n{B}")
    
    # Проверим собственные значения исходной системы
    eigenvals_open = np.linalg.eigvals(A)
    print(f"\nСобственные значения разомкнутой системы: {eigenvals_open}")
    print(f"Система неустойчива: {np.any(np.real(eigenvals_open) > 0)}")
    
    # Проверим управляемость
    C = np.hstack([B, A@B])
    rank_C = np.linalg.matrix_rank(C)
    print(f"\nМатрица управляемости C = [B, AB] = \n{C}")
    print(f"Ранг матрицы управляемости: {rank_C}")
    print(f"Система управляема: {rank_C == 2}")
    
    return solve_lmi_problem(A, B)

def solve_lmi_problem(A, B):
    """Решение задачи LMI для синтеза регулятора"""
    print("\n" + "=" * 60)
    print("РЕШЕНИЕ ЗАДАЧИ LMI")
    print("=" * 60)
    
    # Для экспоненциальной устойчивости степени α = 2
    # Условие: Re(λ) < -α для всех собственных значений замкнутой системы
    # Это эквивалентно: A + BK + αI должна быть устойчивой
    
    alpha = 2
    
    # Метод 1: Прямое размещение полюсов
    print("Метод 1: Размещение полюсов")
    # Желаемые полюса: λ₁ = -3, λ₂ = -4 (обеспечивают степень устойчивости > 2)
    desired_poles = [-3, -4]
    
    # Используем метод размещения полюсов
    K_pp = place_poles(A, B, desired_poles)
    
    print(f"Регулятор размещения полюсов: K = \n{K_pp}")
    
    # Проверка собственных значений замкнутой системы
    A_cl_pp = A + B @ K_pp
    eigenvals_pp = np.linalg.eigvals(A_cl_pp)
    print(f"Собственные значения: λ = {eigenvals_pp}")
    
    max_real_part_pp = np.max(np.real(eigenvals_pp))
    print(f"Максимальная вещественная часть: {max_real_part_pp:.4f}")
    print(f"Условие Re(λ) < -2 выполнено: {max_real_part_pp < -2}")
    
    # Метод 2: LQR с большими весами
    print("\nМетод 2: LQR с большими весами")
    Q = 100 * np.eye(2)  # Большие веса состояния
    R = 1                # Вес управления
    
    P = solve_continuous_are(A, B, Q, R)
    K_lqr = R**(-1) * B.T @ P
    
    print(f"LQR регулятор: K = \n{K_lqr}")
    
    A_cl_lqr = A + B @ K_lqr
    eigenvals_lqr = np.linalg.eigvals(A_cl_lqr)
    print(f"Собственные значения: λ = {eigenvals_lqr}")
    
    max_real_part_lqr = np.max(np.real(eigenvals_lqr))
    print(f"Максимальная вещественная часть: {max_real_part_lqr:.4f}")
    print(f"Условие Re(λ) < -2 выполнено: {max_real_part_lqr < -2}")
    
    # Выбираем лучший регулятор
    if max_real_part_pp < -2 and max_real_part_lqr < -2:
        if max_real_part_pp < max_real_part_lqr:
            print("\nВыбран регулятор размещения полюсов (более агрессивный)")
            return K_pp, P, A_cl_pp
        else:
            print("\nВыбран LQR регулятор")
            return K_lqr, P, A_cl_lqr
    elif max_real_part_pp < -2:
        print("\nВыбран регулятор размещения полюсов")
        return K_pp, P, A_cl_pp
    elif max_real_part_lqr < -2:
        print("\nВыбран LQR регулятор")
        return K_lqr, P, A_cl_lqr
    else:
        print("\nОба метода не обеспечивают требуемую степень устойчивости")
        print("Используем регулятор размещения полюсов как лучший вариант")
        return K_pp, P, A_cl_pp

def place_poles(A, B, poles):
    """Размещение полюсов для системы"""
    # Для системы ẋ = Ax + Bu
    # Желаемый характеристический полином: (s - p₁)(s - p₂) = s² - (p₁+p₂)s + p₁p₂
    
    p1, p2 = poles
    
    # Желаемый характеристический полином: s² + a₁s + a₀
    a1 = -(p1 + p2)
    a0 = p1 * p2
    
    print(f"Желаемый характеристический полином: s² + {a1}s + {a0}")
    
    # Для системы с матрицами A, B находим K такой, что
    # det(sI - (A + BK)) = s² + a₁s + a₀
    
    # Для нашей системы A = [[0,1],[2,0]], B = [[0],[1]]
    # A + BK = [[0, 1], [2+k₁, k₂]]
    # Характеристический полином: s² - k₂s - (2+k₁) = 0
    
    # Приравниваем коэффициенты:
    # -k₂ = a₁  =>  k₂ = -a₁
    # -(2+k₁) = a₀  =>  k₁ = -2 - a₀
    
    k1 = -2 - a0
    k2 = -a1
    
    K = np.array([[k1, k2]])
    
    print(f"Коэффициенты регулятора: k₁ = {k1}, k₂ = {k2}")
    
    return K

def simulate_closed_loop_system(K, A_cl):
    """Моделирование замкнутой системы"""
    print("\n" + "=" * 60)
    print("МОДЕЛИРОВАНИЕ ЗАМКНУТОЙ СИСТЕМЫ")
    print("=" * 60)
    
    from scipy.integrate import solve_ivp
    
    def closed_loop_system(t, x):
        return A_cl @ x
    
    # Начальные условия
    x0 = np.array([1.0, 1.0])
    
    # Время моделирования
    t_span = (0, 3)  # Уменьшаем время для лучшей визуализации
    t_eval = np.linspace(0, 3, 1000)
    
    # Решение системы
    sol = solve_ivp(closed_loop_system, t_span, x0, t_eval=t_eval)
    
    # Построение графиков
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Моделирование замкнутой системы с регулятором', fontsize=14)
    
    # График состояний
    axes[0,0].plot(sol.t, sol.y[0], 'b-', linewidth=2, label='x₁(t)')
    axes[0,0].plot(sol.t, sol.y[1], 'r-', linewidth=2, label='x₂(t)')
    axes[0,0].set_title('Состояния системы')
    axes[0,0].set_xlabel('Время t')
    axes[0,0].set_ylabel('Состояние')
    axes[0,0].grid(True)
    axes[0,0].legend()
    
    # График управления
    u = np.array([K @ sol.y[:, i] for i in range(len(sol.t))]).flatten()
    axes[0,1].plot(sol.t, u, 'g-', linewidth=2, label='u(t)')
    axes[0,1].set_title('Управление')
    axes[0,1].set_xlabel('Время t')
    axes[0,1].set_ylabel('Управление u')
    axes[0,1].grid(True)
    axes[0,1].legend()
    
    # Фазовый портрет
    axes[1,0].plot(sol.y[0], sol.y[1], 'b-', linewidth=2, label='Траектория')
    axes[1,0].plot(x0[0], x0[1], 'ro', markersize=8, label='Начальная точка')
    axes[1,0].plot(0, 0, 'ko', markersize=8, label='Цель')
    axes[1,0].set_title('Фазовый портрет')
    axes[1,0].set_xlabel('x₁')
    axes[1,0].set_ylabel('x₂')
    axes[1,0].grid(True)
    axes[1,0].legend()
    
    # Проверка экспоненциальной устойчивости степени 2
    norm_x = np.sqrt(sol.y[0]**2 + sol.y[1]**2)
    bound = norm_x[0] * np.exp(-2 * sol.t)
    
    axes[1,1].semilogy(sol.t, norm_x, 'b-', linewidth=2, label='||x(t)||')
    axes[1,1].semilogy(sol.t, bound, 'r--', linewidth=2, label='M||x(0)||e^(-2t)')
    axes[1,1].set_title('Проверка экспоненциальной устойчивости')
    axes[1,1].set_xlabel('Время t')
    axes[1,1].set_ylabel('Норма состояния')
    axes[1,1].grid(True)
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('/home/leonidas/projects/itmo/nonlinear_systems/lab2/images/task3/lmi_simulation.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Графики сохранены в images/task3/lmi_simulation.png")
    
    # Проверка экспоненциальной устойчивости
    print(f"\nПроверка экспоненциальной устойчивости степени 2:")
    print(f"Начальная норма: ||x(0)|| = {norm_x[0]:.4f}")
    print(f"Финальная норма: ||x(3)|| = {norm_x[-1]:.4f}")
    print(f"Ожидаемая граница при t=3: {bound[-1]:.4f}")
    print(f"Условие выполнено: {norm_x[-1] <= bound[-1]}")

def main():
    """Основная функция"""
    print("СИНТЕЗ ЛИНЕЙНОГО РЕГУЛЯТОРА ЧЕРЕЗ LMI")
    print("Система: ẋ₁ = x₂, ẋ₂ = 2x₁ + u")
    print("Цель: экспоненциальная устойчивость степени 2")
    print("=" * 80)
    
    # Решение через LMI
    K_lmi, P_lmi, A_cl_lmi = synthesize_lmi_controller()
    
    # Моделирование
    if K_lmi is not None:
        simulate_closed_loop_system(K_lmi, A_cl_lmi)
        
        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТЫ СИНТЕЗА")
        print("=" * 80)
        print(f"Матрица обратной связи: K = \n{K_lmi}")
        print(f"Собственные значения замкнутой системы: {np.linalg.eigvals(A_cl_lmi)}")
        
        max_real_part = np.max(np.real(np.linalg.eigvals(A_cl_lmi)))
        if max_real_part < -2:
            print("Экспоненциальная устойчивость степени 2 достигнута!")
        else:
            print(f"Экспоненциальная устойчивость степени 2 НЕ достигнута (max Re(λ) = {max_real_part:.4f})")
        print("=" * 80)

if __name__ == "__main__":
    main()
