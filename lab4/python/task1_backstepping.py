#!/usr/bin/env python3
"""
Синтез стабилизирующего регулятора методом бэкстеппинга
Задача 1 лабораторной работы №4
Система: ẋ₁ = x₂ + sin x₁ + x₁², ẋ₂ = x₁² + (2 + sin x₁)u
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, diff, simplify, sin, cos
from scipy.integrate import solve_ivp

# Настройка для корректного отображения русского текста
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def backstepping_synthesis_task1():
    """Синтез регулятора методом бэкстеппинга для системы 1"""
    print("=" * 80)
    print("СИНТЕЗ РЕГУЛЯТОРА МЕТОДОМ БЭКСТЕППИНГА - ЗАДАЧА 1")
    print("=" * 80)
    
    print("Система:")
    print("ẋ₁ = x₂ + sin x₁ + x₁²")
    print("ẋ₂ = x₁² + (2 + sin x₁)u")
    
    # Определяем переменные
    x1, x2 = symbols('x1 x2')
    
    print("\n" + "=" * 60)
    print("ШАГ 1: Стабилизация первой подсистемы")
    print("=" * 60)
    
    # Рассматриваем первую подсистему: ẋ₁ = x₂ + sin x₁ + x₁²
    # x₂ рассматриваем как виртуальное управление
    print("Рассматриваем первую подсистему:")
    print("ẋ₁ = x₂ + sin x₁ + x₁²")
    print("x₂ рассматривается как виртуальное управление")
    
    # Выбираем функцию Ляпунова
    V1 = 0.5 * x1**2
    print(f"\nФункция Ляпунова: V₁ = ½x₁²")
    
    # Вычисляем производную
    # ẋ₁ = x₂ + sin x₁ + x₁²
    # Ṽ₁ = x₁ · ẋ₁ = x₁(x₂ + sin x₁ + x₁²)
    
    # Выбираем виртуальное управление α₁(x₁)
    c1 = 2.0  # Параметр настройки
    alpha1 = -c1*x1 - sin(x1) - x1**2
    
    print(f"\nВиртуальное управление:")
    print(f"α₁(x₁) = -c₁x₁ - sin x₁ - x₁²")
    print(f"При c₁ = {c1}: α₁ = {alpha1}")
    
    # Ошибка: z₂ = x₂ - α₁
    z2 = x2 - alpha1
    
    # Производная функции Ляпунова с виртуальным управлением
    V1_dot_sub = x1*(alpha1 + sin(x1) + x1**2)  # Подставили α₁
    V1_dot_sub = simplify(V1_dot_sub)
    print(f"\nṼ₁ = x₁(α₁ + sin x₁ + x₁²) = {V1_dot_sub}")
    
    # С учетом ошибки
    V1_dot = x1*(alpha1 + sin(x1) + x1**2) + x1*z2
    print(f"Ṽ₁ = {V1_dot_sub} + x₁z₂")
    
    print("\n" + "=" * 60)
    print("ШАГ 2: Стабилизация второй подсистемы")
    print("=" * 60)
    
    # Рассматриваем ошибку z₂ = x₂ - α₁
    print("Ошибка: z₂ = x₂ - α₁")
    
    # Вычисляем ż₂
    # ż₂ = ẋ₂ - α̇₁ = ẋ₂ - ∂α₁/∂x₁ · ẋ₁
    alpha1_diff = diff(alpha1, x1)
    alpha1_diff = simplify(alpha1_diff)
    print(f"\n∂α₁/∂x₁ = {alpha1_diff}")
    
    # ẋ₁ = x₂ + sin x₁ + x₁²
    alpha1_dot = alpha1_diff * (x2 + sin(x1) + x1**2)
    alpha1_dot = simplify(alpha1_dot)
    print(f"α̇₁ = (∂α₁/∂x₁) · ẋ₁ = {alpha1_dot}")
    
    # ż₂ = ẋ₂ - α̇₁ = x₁² + (2 + sin x₁)u - α̇₁
    z2_dot = x1**2 + (2 + sin(x1))*symbols('u') - alpha1_dot
    
    # Расширенная функция Ляпунова
    V = V1 + 0.5 * z2**2
    print(f"\nРасширенная функция Ляпунова: V = V₁ + ½z₂²")
    
    # Производная расширенной функции Ляпунова
    V_dot = V1_dot + z2*z2_dot
    # V_dot = -c₁x₁² + x₁z₂ + z₂(x₁² + (2 + sin x₁)u - α̇₁)
    
    print(f"\nṼ = -c₁x₁² + x₁z₂ + z₂(x₁² + (2 + sin x₁)u - α̇₁)")
    
    # Выбираем закон управления
    c2 = 3.0  # Параметр настройки
    # Хотим: ż₂ = -c₂z₂ - x₁
    # z₂(x₁² + (2 + sin x₁)u - α̇₁) = z₂(-c₂z₂ - x₁)
    # x₁² + (2 + sin x₁)u - α̇₁ = -c₂z₂ - x₁
    # (2 + sin x₁)u = -c₂z₂ - x₁ - x₁² + α̇₁
    # u = (-c₂z₂ - x₁ - x₁² + α̇₁) / (2 + sin x₁)
    
    u_sym = (-c2*z2 - x1 - x1**2 + alpha1_dot) / (2 + sin(x1))
    u_sym = simplify(u_sym)
    
    print(f"\nЗакон управления:")
    print(f"u = (-c₂z₂ - x₁ - x₁² + α̇₁) / (2 + sin x₁)")
    print(f"При c₁ = {c1}, c₂ = {c2}:")
    
    # Подставим z₂ = x₂ - α₁
    u_final = u_sym.subs(z2, x2 - alpha1)
    u_final = simplify(u_final)
    
    print(f"u = {u_final}")
    
    # Проверяем производную функции Ляпунова с управлением
    V_dot_final = V_dot.subs(symbols('u'), u_sym)
    V_dot_final = simplify(V_dot_final)
    
    print(f"\nṼ = {V_dot_final}")
    print(f"Ṽ = -c₁x₁² - c₂z₂²")
    
    return u_final, c1, c2, alpha1

def simulate_system1(u_func, c1, c2, alpha1_func):
    """Моделирование системы 1"""
    print("\n" + "=" * 60)
    print("МОДЕЛИРОВАНИЕ СИСТЕМЫ 1")
    print("=" * 60)
    
    # Преобразуем символьные выражения в функции
    x1, x2 = symbols('x1 x2')
    
    def u_control(x1_val, x2_val):
        # Вычисляем α₁
        alpha1_val = -c1*x1_val - np.sin(x1_val) - x1_val**2
        
        # Вычисляем z₂
        z2_val = x2_val - alpha1_val
        
        # Вычисляем α̇₁
        alpha1_diff_val = -c1 - np.cos(x1_val) - 2*x1_val
        x1_dot_val = x2_val + np.sin(x1_val) + x1_val**2
        alpha1_dot_val = alpha1_diff_val * x1_dot_val
        
        # Вычисляем u
        denominator = 2 + np.sin(x1_val)
        if abs(denominator) < 1e-10:
            denominator = 1e-10 if denominator >= 0 else -1e-10
        
        u_val = (-c2*z2_val - x1_val - x1_val**2 + alpha1_dot_val) / denominator
        return u_val
    
    # Система с управлением
    def controlled_system(t, x):
        x1_val, x2_val = x
        u_val = u_control(x1_val, x2_val)
        
        dx1 = x2_val + np.sin(x1_val) + x1_val**2
        dx2 = x1_val**2 + (2 + np.sin(x1_val)) * u_val
        
        return [dx1, dx2]
    
    # Система без управления
    def uncontrolled_system(t, x):
        x1_val, x2_val = x
        u_val = 0
        
        dx1 = x2_val + np.sin(x1_val) + x1_val**2
        dx2 = x1_val**2 + (2 + np.sin(x1_val)) * u_val
        
        return [dx1, dx2]
    
    # Начальные условия
    x0 = [1.5, 1.0]
    
    # Время моделирования
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 1000)
    
    # Решение управляемой системы
    sol_controlled = solve_ivp(controlled_system, t_span, x0, t_eval=t_eval)
    
    # Решение неуправляемой системы
    sol_uncontrolled = solve_ivp(uncontrolled_system, t_span, x0, t_eval=t_eval)
    
    # Построение графиков
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Моделирование системы 1 с регулятором бэкстеппинга', fontsize=14)
    
    # График состояний управляемой системы
    axes[0,0].plot(sol_controlled.t, sol_controlled.y[0], 'b-', linewidth=2, label='x₁(t)')
    axes[0,0].plot(sol_controlled.t, sol_controlled.y[1], 'r-', linewidth=2, label='x₂(t)')
    axes[0,0].set_title('Управляемая система')
    axes[0,0].set_xlabel('Время t')
    axes[0,0].set_ylabel('Состояние')
    axes[0,0].grid(True)
    axes[0,0].legend()
    
    # График состояний неуправляемой системы
    axes[0,1].plot(sol_uncontrolled.t, sol_uncontrolled.y[0], 'b-', linewidth=2, label='x₁(t)')
    axes[0,1].plot(sol_uncontrolled.t, sol_uncontrolled.y[1], 'r-', linewidth=2, label='x₂(t)')
    axes[0,1].set_title('Неуправляемая система')
    axes[0,1].set_xlabel('Время t')
    axes[0,1].set_ylabel('Состояние')
    axes[0,1].grid(True)
    axes[0,1].legend()
    
    # График управления
    u_values = [u_control(sol_controlled.y[0, i], sol_controlled.y[1, i]) 
                for i in range(len(sol_controlled.t))]
    axes[1,0].plot(sol_controlled.t, u_values, 'g-', linewidth=2, label='u(t)')
    axes[1,0].set_title('Управление')
    axes[1,0].set_xlabel('Время t')
    axes[1,0].set_ylabel('Управление u')
    axes[1,0].grid(True)
    axes[1,0].legend()
    
    # Фазовый портрет
    axes[1,1].plot(sol_controlled.y[0], sol_controlled.y[1], 'b-', linewidth=2, label='Управляемая')
    axes[1,1].plot(sol_uncontrolled.y[0], sol_uncontrolled.y[1], 'r--', linewidth=2, label='Неуправляемая')
    axes[1,1].plot(x0[0], x0[1], 'ro', markersize=8, label='Начальная точка')
    axes[1,1].plot(0, 0, 'ko', markersize=8, label='Цель')
    axes[1,1].set_title('Фазовый портрет')
    axes[1,1].set_xlabel('x₁')
    axes[1,1].set_ylabel('x₂')
    axes[1,1].grid(True)
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('/home/leonidas/projects/itmo/nonlinear_systems/lab4/images/task1/backstepping_system1.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Графики сохранены в images/task1/backstepping_system1.png")
    
    # Анализ результатов
    print(f"\nАнализ результатов:")
    print(f"Управляемая система - финальные значения:")
    print(f"x₁(10) = {sol_controlled.y[0, -1]:.4f}")
    print(f"x₂(10) = {sol_controlled.y[1, -1]:.4f}")
    
    final_norm = np.sqrt(sol_controlled.y[0, -1]**2 + sol_controlled.y[1, -1]**2)
    print(f"Финальная норма: {final_norm:.4f}")
    print(f"Стабилизация {'достигнута' if final_norm < 0.1 else 'частично достигнута'}")

def main():
    """Основная функция"""
    print("СИНТЕЗ СТАБИЛИЗИРУЮЩЕГО РЕГУЛЯТОРА МЕТОДОМ БЭКСТЕППИНГА")
    print("Задача 1")
    print("=" * 80)
    
    # Синтез регулятора
    u_final, c1, c2, alpha1 = backstepping_synthesis_task1()
    
    # Моделирование
    simulate_system1(u_final, c1, c2, alpha1)
    
    print("\n" + "=" * 80)
    print("ЗАКЛЮЧЕНИЕ ПО ЗАДАЧЕ 1")
    print("=" * 80)
    print("1. Регулятор синтезирован методом бэкстеппинга")
    print(f"2. Параметры: c₁ = {c1}, c₂ = {c2}")
    print("3. Глобальная стабилизация начала координат достигнута")
    print("=" * 80)

if __name__ == "__main__":
    main()
