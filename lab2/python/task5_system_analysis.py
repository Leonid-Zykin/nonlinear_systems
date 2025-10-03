#!/usr/bin/env python3
"""
Анализ системы с управлением u = Kx
Задача 5 лабораторной работы №2
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, diff, simplify, solve, Matrix
from scipy.integrate import solve_ivp
from scipy.linalg import eig

# Настройка для корректного отображения русского текста
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def analyze_system_with_control():
    """Анализ системы с управлением u = Kx"""
    print("=" * 80)
    print("АНАЛИЗ СИСТЕМЫ С УПРАВЛЕНИЕМ u = Kx")
    print("=" * 80)
    
    print("Дано:")
    print("Система: ẋ₁ = x₂ - 0.5x₁³, ẋ₂ = u")
    print("Управление: u = Kx")
    
    # Матрицы системы
    A = np.array([[0, 1],
                  [0, 0]])
    B = np.array([[0],
                  [1]])
    
    print(f"\nМатрицы системы:")
    print(f"A = \n{A}")
    print(f"B = \n{B}")
    
    return analyze_linear_part(A, B)

def analyze_linear_part(A, B):
    """Анализ линейной части системы"""
    print("\n" + "=" * 60)
    print("АНАЛИЗ ЛИНЕЙНОЙ ЧАСТИ СИСТЕМЫ")
    print("=" * 60)
    
    # Собственные значения разомкнутой системы
    eigenvals_open = np.linalg.eigvals(A)
    print(f"Собственные значения разомкнутой системы: λ = {eigenvals_open}")
    print(f"Система неустойчива: {np.any(np.real(eigenvals_open) >= 0)}")
    
    # Проверим управляемость
    C = np.hstack([B, A@B])
    rank_C = np.linalg.matrix_rank(C)
    print(f"\nМатрица управляемости C = [B, AB] = \n{C}")
    print(f"Ранг матрицы управляемости: {rank_C}")
    print(f"Система управляема: {rank_C == 2}")
    
    return synthesize_controller(A, B)

def synthesize_controller(A, B):
    """Синтез регулятора для системы"""
    print("\n" + "=" * 60)
    print("СИНТЕЗ РЕГУЛЯТОРА")
    print("=" * 60)
    
    # Метод 1: Размещение полюсов
    print("Метод 1: Размещение полюсов")
    desired_poles = [-2, -3]
    
    K_pp = place_poles(A, B, desired_poles)
    print(f"Регулятор размещения полюсов: K = \n{K_pp}")
    
    # Проверка собственных значений замкнутой системы
    A_cl_pp = A + B @ K_pp
    eigenvals_pp = np.linalg.eigvals(A_cl_pp)
    print(f"Собственные значения: λ = {eigenvals_pp}")
    
    # Метод 2: LQR
    print("\nМетод 2: LQR")
    from scipy.linalg import solve_continuous_are
    
    Q = np.eye(2)
    R = 1
    
    P = solve_continuous_are(A, B, Q, R)
    K_lqr = R**(-1) * B.T @ P
    
    print(f"LQR регулятор: K = \n{K_lqr}")
    
    A_cl_lqr = A + B @ K_lqr
    eigenvals_lqr = np.linalg.eigvals(A_cl_lqr)
    print(f"Собственные значения: λ = {eigenvals_lqr}")
    
    # Выбираем регулятор размещения полюсов
    print("\nВыбран регулятор размещения полюсов")
    return K_pp, A_cl_pp

def place_poles(A, B, poles):
    """Размещение полюсов для системы"""
    # Для системы ẋ = Ax + Bu
    # Желаемый характеристический полином: (s - p₁)(s - p₂) = s² - (p₁+p₂)s + p₁p₂
    
    p1, p2 = poles
    
    # Желаемый характеристический полином: s² + a₁s + a₀
    a1 = -(p1 + p2)
    a0 = p1 * p2
    
    print(f"Желаемый характеристический полином: s² + {a1}s + {a0}")
    
    # Для нашей системы A = [[0,1],[0,0]], B = [[0],[1]]
    # A + BK = [[0, 1], [k₁, k₂]]
    # Характеристический полином: s² - k₂s - k₁ = 0
    
    # Приравниваем коэффициенты:
    # -k₂ = a₁  =>  k₂ = -a₁
    # -k₁ = a₀  =>  k₁ = -a₀
    
    k1 = -a0
    k2 = -a1
    
    K = np.array([[k1, k2]])
    
    print(f"Коэффициенты регулятора: k₁ = {k1}, k₂ = {k2}")
    
    return K

def analyze_nonlinear_system(K):
    """Анализ нелинейной системы с управлением"""
    print("\n" + "=" * 60)
    print("АНАЛИЗ НЕЛИНЕЙНОЙ СИСТЕМЫ")
    print("=" * 60)
    
    # Система: ẋ₁ = x₂ - 0.5x₁³, ẋ₂ = u = Kx
    # С учетом K = [k₁, k₂]: ẋ₂ = k₁x₁ + k₂x₂
    
    print("Нелинейная система с управлением:")
    print("ẋ₁ = x₂ - 0.5x₁³")
    print("ẋ₂ = k₁x₁ + k₂x₂")
    
    # Линеаризация в начале координат
    print("\nЛинеаризация в начале координат:")
    print("Матрица Якоби:")
    print("J = [[0, 1], [k₁, k₂]]")
    
    # Характеристический полином: det(sI - J) = s² - k₂s - k₁
    print(f"Характеристический полином: s² - k₂s - k₁")
    
    return simulate_nonlinear_system(K)

def simulate_nonlinear_system(K):
    """Моделирование нелинейной системы"""
    print("\n" + "=" * 60)
    print("МОДЕЛИРОВАНИЕ НЕЛИНЕЙНОЙ СИСТЕМЫ")
    print("=" * 60)
    
    k1, k2 = K[0]
    
    # Система: ẋ₁ = x₂ - 0.5x₁³, ẋ₂ = k₁x₁ + k₂x₂
    def nonlinear_system(t, x):
        x1, x2 = x
        dx1 = x2 - 0.5 * x1**3
        dx2 = k1 * x1 + k2 * x2
        return [dx1, dx2]
    
    # Начальные условия
    x0 = [1.0, 1.0]
    
    # Время моделирования
    t_span = (0, 5)
    t_eval = np.linspace(0, 5, 1000)
    
    # Решение
    sol = solve_ivp(nonlinear_system, t_span, x0, t_eval=t_eval)
    
    # Построение графиков
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Моделирование нелинейной системы с управлением', fontsize=14)
    
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
    
    # Проверка устойчивости
    norm_x = np.sqrt(sol.y[0]**2 + sol.y[1]**2)
    axes[1,1].semilogy(sol.t, norm_x, 'b-', linewidth=2, label='||x(t)||')
    axes[1,1].set_title('Норма состояния')
    axes[1,1].set_xlabel('Время t')
    axes[1,1].set_ylabel('Норма состояния')
    axes[1,1].grid(True)
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('/home/leonidas/projects/itmo/nonlinear_systems/lab2/images/task5/nonlinear_system.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Графики сохранены в images/task5/nonlinear_system.png")
    
    # Анализ результатов
    print(f"\nАнализ результатов:")
    print(f"Начальная норма: ||x(0)|| = {norm_x[0]:.4f}")
    print(f"Финальная норма: ||x(5)|| = {norm_x[-1]:.4f}")
    print(f"Система устойчива: {norm_x[-1] < norm_x[0]}")
    
    return analyze_stability_properties(K)

def analyze_stability_properties(K):
    """Анализ свойств устойчивости"""
    print("\n" + "=" * 60)
    print("АНАЛИЗ СВОЙСТВ УСТОЙЧИВОСТИ")
    print("=" * 60)
    
    k1, k2 = K[0]
    
    # Линеаризованная система: ẋ = Jx, где J = [[0, 1], [k₁, k₂]]
    J = np.array([[0, 1], [k1, k2]])
    
    eigenvals = np.linalg.eigvals(J)
    print(f"Собственные значения линеаризованной системы: λ = {eigenvals}")
    
    max_real_part = np.max(np.real(eigenvals))
    print(f"Максимальная вещественная часть: {max_real_part:.4f}")
    
    if max_real_part < 0:
        print("Линеаризованная система асимптотически устойчива")
    else:
        print("Линеаризованная система неустойчива")
    
    # Анализ нелинейного члена
    print(f"\nАнализ нелинейного члена:")
    print(f"Нелинейный член: -0.5x₁³")
    print(f"В малой окрестности начала координат этот член мал")
    print(f"Поэтому локальная устойчивость определяется линеаризованной системой")
    
    return K

def main():
    """Основная функция"""
    print("АНАЛИЗ СИСТЕМЫ С УПРАВЛЕНИЕМ u = Kx")
    print("Система: ẋ₁ = x₂ - 0.5x₁³, ẋ₂ = u")
    print("Управление: u = Kx")
    print("=" * 80)
    
    # Анализ системы
    K, A_cl = analyze_system_with_control()
    
    # Анализ нелинейной системы
    K_final = analyze_nonlinear_system(K)
    
    print("\n" + "=" * 80)
    print("ЗАКЛЮЧЕНИЕ ПО ЗАДАЧЕ 5")
    print("=" * 80)
    print(f"Матрица обратной связи: K = \n{K_final}")
    print(f"Собственные значения линеаризованной системы: {np.linalg.eigvals(np.array([[0, 1], [K_final[0,0], K_final[0,1]]]))}")
    print("Система локально асимптотически устойчива")
    print("Нелинейный член -0.5x₁³ не нарушает локальную устойчивость")
    print("=" * 80)

if __name__ == "__main__":
    main()
