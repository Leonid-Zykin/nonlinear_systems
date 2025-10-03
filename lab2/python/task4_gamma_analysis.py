#!/usr/bin/env python3
"""
Анализ ограничивающего условия на параметр γ для асимптотической устойчивости степени 1
Задача 4 лабораторной работы №2
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, diff, simplify, solve, Matrix
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

# Настройка для корректного отображения русского текста
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def analyze_gamma_condition():
    """Анализ ограничивающего условия на параметр γ"""
    print("=" * 80)
    print("АНАЛИЗ ОГРАНИЧИВАЮЩЕГО УСЛОВИЯ НА ПАРАМЕТР γ")
    print("=" * 80)
    
    print("Дано:")
    print("Система: ẋ₁ = x₂ + γ sin x₂, ẋ₂ = 2x₁ + u")
    print("Требуется: асимптотическая устойчивость степени 1")
    print("Закон управления взят из предыдущего задания: u = Kx")
    print("где K = [-14, -7]")
    
    # Матрицы системы
    A = np.array([[0, 1],
                  [2, 0]])
    B = np.array([[0],
                  [1]])
    K = np.array([[-14, -7]])
    
    print(f"\nМатрицы системы:")
    print(f"A = \n{A}")
    print(f"B = \n{B}")
    print(f"K = \n{K}")
    
    return analyze_stability_with_gamma(A, B, K)

def analyze_stability_with_gamma(A, B, K):
    """Анализ устойчивости с учетом параметра γ"""
    print("\n" + "=" * 60)
    print("АНАЛИЗ УСТОЙЧИВОСТИ С ПАРАМЕТРОМ γ")
    print("=" * 60)
    
    # Замкнутая система без γ: ẋ = (A + BK)x
    A_cl = A + B @ K
    print(f"Замкнутая система без γ: A + BK = \n{A_cl}")
    
    eigenvals_cl = np.linalg.eigvals(A_cl)
    print(f"Собственные значения замкнутой системы: λ = {eigenvals_cl}")
    
    # С учетом γ: ẋ₁ = x₂ + γ sin x₂, ẋ₂ = 2x₁ + u
    # где u = Kx = -14x₁ - 7x₂
    # Получаем: ẋ₁ = x₂ + γ sin x₂, ẋ₂ = 2x₁ - 14x₁ - 7x₂ = -12x₁ - 7x₂
    
    print("\nСистема с параметром γ:")
    print("ẋ₁ = x₂ + γ sin x₂")
    print("ẋ₂ = -12x₁ - 7x₂")
    
    return analyze_nonlinear_system()

def analyze_nonlinear_system():
    """Анализ нелинейной системы с γ"""
    print("\n" + "=" * 60)
    print("АНАЛИЗ НЕЛИНЕЙНОЙ СИСТЕМЫ")
    print("=" * 60)
    
    # Линеаризация в начале координат
    # f₁(x) = x₂ + γ sin x₂
    # f₂(x) = -12x₁ - 7x₂
    
    # Матрица Якоби в точке (0,0):
    # ∂f₁/∂x₁ = 0, ∂f₁/∂x₂ = 1 + γ cos(0) = 1 + γ
    # ∂f₂/∂x₁ = -12, ∂f₂/∂x₂ = -7
    
    print("Линеаризация в начале координат:")
    print("Матрица Якоби:")
    print("J = [[0, 1+γ], [-12, -7]]")
    
    # Характеристический полином: det(sI - J) = s² - tr(J)s + det(J)
    # tr(J) = 0 + (-7) = -7
    # det(J) = 0*(-7) - (1+γ)*(-12) = 12(1+γ)
    
    print(f"\nХарактеристический полином: s² + 7s + 12(1+γ)")
    
    # Условие устойчивости: все коэффициенты положительны
    # 7 > 0 (всегда выполнено)
    # 12(1+γ) > 0 => 1+γ > 0 => γ > -1
    
    print(f"\nУсловие устойчивости:")
    print(f"12(1+γ) > 0 => γ > -1")
    
    # Для асимптотической устойчивости степени 1 нужно Re(λ) < -1
    # Корни: s = (-7 ± √(49 - 48(1+γ)))/2 = (-7 ± √(1-48γ))/2
    
    print(f"\nКорни характеристического уравнения:")
    print(f"s = (-7 ± √(49 - 48(1+γ)))/2 = (-7 ± √(1-48γ))/2")
    
    # Для Re(s) < -1 нужно:
    # (-7 + √(1-48γ))/2 < -1
    # -7 + √(1-48γ) < -2
    # √(1-48γ) < 5
    # 1-48γ < 25
    # -48γ < 24
    # γ > -0.5
    
    print(f"\nУсловие асимптотической устойчивости степени 1:")
    print(f"Re(s) < -1 => γ > -0.5")
    
    return find_critical_gamma()

def find_critical_gamma():
    """Поиск критического значения γ"""
    print("\n" + "=" * 60)
    print("ПОИСК КРИТИЧЕСКОГО ЗНАЧЕНИЯ γ")
    print("=" * 60)
    
    # Для точного анализа рассмотрим дискриминант
    # D = 49 - 48(1+γ) = 49 - 48 - 48γ = 1 - 48γ
    
    print("Дискриминант: D = 1 - 48γ")
    
    # Случай 1: D > 0 (вещественные корни)
    # 1 - 48γ > 0 => γ < 1/48 ≈ 0.0208
    print(f"\nСлучай 1: D > 0 (вещественные корни)")
    print(f"γ < 1/48 ≈ 0.0208")
    
    # Случай 2: D = 0 (кратный корень)
    # γ = 1/48 ≈ 0.0208
    print(f"\nСлучай 2: D = 0 (кратный корень)")
    print(f"γ = 1/48 ≈ 0.0208")
    
    # Случай 3: D < 0 (комплексные корни)
    # γ > 1/48 ≈ 0.0208
    print(f"\nСлучай 3: D < 0 (комплексные корни)")
    print(f"γ > 1/48 ≈ 0.0208")
    
    # Для комплексных корней: s = (-7 ± i√(48γ-1))/2
    # Re(s) = -7/2 = -3.5 < -1 (всегда выполнено)
    
    print(f"\nДля комплексных корней:")
    print(f"Re(s) = -3.5 < -1 (всегда выполнено)")
    
    # Итоговое условие: γ > -0.5
    print(f"\nИТОГОВОЕ УСЛОВИЕ:")
    print(f"γ > -0.5")
    
    return simulate_different_gamma_values()

def simulate_different_gamma_values():
    """Моделирование для различных значений γ"""
    print("\n" + "=" * 60)
    print("МОДЕЛИРОВАНИЕ ДЛЯ РАЗЛИЧНЫХ ЗНАЧЕНИЙ γ")
    print("=" * 60)
    
    gamma_values = [-0.6, -0.5, -0.3, 0.0, 0.5]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Моделирование системы для различных значений γ', fontsize=14)
    
    for i, gamma in enumerate(gamma_values):
        row = i // 3
        col = i % 3
        
        # Система: ẋ₁ = x₂ + γ sin x₂, ẋ₂ = -12x₁ - 7x₂
        def system_gamma(t, x):
            x1, x2 = x
            dx1 = x2 + gamma * np.sin(x2)
            dx2 = -12 * x1 - 7 * x2
            return [dx1, dx2]
        
        # Начальные условия
        x0 = [1.0, 1.0]
        
        # Время моделирования
        t_span = (0, 5)
        t_eval = np.linspace(0, 5, 1000)
        
        # Решение
        sol = solve_ivp(system_gamma, t_span, x0, t_eval=t_eval)
        
        # График состояний
        axes[row, col].plot(sol.t, sol.y[0], 'b-', linewidth=2, label='x₁(t)')
        axes[row, col].plot(sol.t, sol.y[1], 'r-', linewidth=2, label='x₂(t)')
        axes[row, col].set_title(f'γ = {gamma}')
        axes[row, col].set_xlabel('Время t')
        axes[row, col].set_ylabel('Состояние')
        axes[row, col].grid(True)
        axes[row, col].legend()
        
        # Проверка устойчивости
        final_norm = np.sqrt(sol.y[0, -1]**2 + sol.y[1, -1]**2)
        print(f"γ = {gamma:4.1f}: финальная норма = {final_norm:.4f}")
    
    # Убираем последний subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/leonidas/projects/itmo/nonlinear_systems/lab2/images/task4/gamma_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Графики сохранены в images/task4/gamma_analysis.png")

def main():
    """Основная функция"""
    print("АНАЛИЗ ОГРАНИЧИВАЮЩЕГО УСЛОВИЯ НА ПАРАМЕТР γ")
    print("Система: ẋ₁ = x₂ + γ sin x₂, ẋ₂ = 2x₁ + u")
    print("Закон управления: u = Kx, где K = [-14, -7]")
    print("=" * 80)
    
    analyze_gamma_condition()
    
    print("\n" + "=" * 80)
    print("ЗАКЛЮЧЕНИЕ ПО ЗАДАЧЕ 4")
    print("=" * 80)
    print("Ограничивающее условие на параметр γ:")
    print("γ > -0.5")
    print("\nОбоснование:")
    print("- При γ > -0.5 все собственные значения линеаризованной системы")
    print("  имеют вещественную часть меньше -1")
    print("- Это обеспечивает асимптотическую устойчивость степени 1")
    print("- При γ ≤ -0.5 система становится неустойчивой")
    print("=" * 80)

if __name__ == "__main__":
    main()
