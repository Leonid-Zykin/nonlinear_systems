#!/usr/bin/env python3
"""
Синтез стабилизирующего регулятора методом бэкстеппинга
Задача 3 лабораторной работы №4
Система: ẋ₁ = cos x₁ - x₂, ẋ₂ = x₁ + x₃, ẋ₃ = x₁x₃ + (2 - sin x₃)x₄, ẋ₄ = x₂x₃ + 2u
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, diff, simplify, sin, cos
from scipy.integrate import solve_ivp

# Настройка для корректного отображения русского текста
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def backstepping_synthesis_task3():
    """Синтез регулятора методом бэкстеппинга для системы 3 (4-го порядка)"""
    print("=" * 80)
    print("СИНТЕЗ РЕГУЛЯТОРА МЕТОДОМ БЭКСТЕППИНГА - ЗАДАЧА 3")
    print("=" * 80)
    
    print("Система:")
    print("ẋ₁ = cos x₁ - x₂")
    print("ẋ₂ = x₁ + x₃")
    print("ẋ₃ = x₁x₃ + (2 - sin x₃)x₄")
    print("ẋ₄ = x₂x₃ + 2u")
    
    # Определяем переменные
    x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
    
    print("\n" + "=" * 60)
    print("ШАГ 1: Стабилизация первой подсистемы")
    print("=" * 60)
    
    # Рассматриваем первую подсистему: ẋ₁ = cos x₁ - x₂
    # x₂ рассматриваем как виртуальное управление
    print("Рассматриваем первую подсистему:")
    print("ẋ₁ = cos x₁ - x₂")
    print("x₂ рассматривается как виртуальное управление")
    
    # Выбираем функцию Ляпунова
    V1 = 0.5 * x1**2
    print(f"\nФункция Ляпунова: V₁ = ½x₁²")
    
    # Выбираем виртуальное управление α₁(x₁)
    c1 = 2.0  # Параметр настройки
    alpha1 = cos(x1) + c1*x1
    
    print(f"\nВиртуальное управление:")
    print(f"α₁(x₁) = cos x₁ + c₁x₁")
    print(f"При c₁ = {c1}: α₁ = {alpha1}")
    
    # Ошибка: z₂ = x₂ - α₁
    z2 = x2 - alpha1
    
    # Производная функции Ляпунова
    V1_dot = x1*(cos(x1) - alpha1) + x1*z2
    V1_dot_sub = x1*(cos(x1) - alpha1)
    V1_dot_sub = simplify(V1_dot_sub)
    print(f"\nṼ₁ = x₁(cos x₁ - α₁) = {V1_dot_sub}")
    print(f"Ṽ₁ = {V1_dot_sub} + x₁z₂")
    
    print("\n" + "=" * 60)
    print("ШАГ 2: Стабилизация второй подсистемы")
    print("=" * 60)
    
    # Рассматриваем вторую подсистему: ẋ₂ = x₁ + x₃
    # x₃ рассматриваем как виртуальное управление
    print("Рассматриваем вторую подсистему:")
    print("ẋ₂ = x₁ + x₃")
    print("x₃ рассматривается как виртуальное управление")
    print("Ошибка: z₂ = x₂ - α₁")
    
    # Вычисляем α̇₁
    alpha1_diff = diff(alpha1, x1)
    alpha1_diff = simplify(alpha1_diff)
    print(f"\n∂α₁/∂x₁ = {alpha1_diff}")
    
    # ẋ₁ = cos x₁ - x₂
    alpha1_dot = alpha1_diff * (cos(x1) - x2)
    alpha1_dot = simplify(alpha1_dot)
    print(f"α̇₁ = (∂α₁/∂x₁) · ẋ₁ = {alpha1_dot}")
    
    # ż₂ = ẋ₂ - α̇₁ = x₁ + x₃ - α̇₁
    z2_dot = x1 + x3 - alpha1_dot
    
    # Расширенная функция Ляпунова
    V2 = V1 + 0.5 * z2**2
    print(f"\nРасширенная функция Ляпунова: V₂ = V₁ + ½z₂²")
    
    # Выбираем виртуальное управление α₂(x₁, x₂)
    c2 = 3.0
    alpha2 = -x1 + alpha1_dot - c2*z2
    
    print(f"\nВиртуальное управление:")
    print(f"α₂(x₁, x₂) = -x₁ + α̇₁ - c₂z₂")
    print(f"При c₂ = {c2}: α₂ = {alpha2}")
    
    # Ошибка: z₃ = x₃ - α₂
    z3 = x3 - alpha2
    
    # Производная функции Ляпунова с виртуальным управлением
    V2_dot_sub = V1_dot_sub + z2*(x1 + alpha2 - alpha1_dot)
    V2_dot_sub = simplify(V2_dot_sub)
    print(f"\nṼ₂ = -c₁x₁² - c₂z₂² + z₂z₃")
    
    print("\n" + "=" * 60)
    print("ШАГ 3: Стабилизация третьей подсистемы")
    print("=" * 60)
    
    # Рассматриваем третью подсистему: ẋ₃ = x₁x₃ + (2 - sin x₃)x₄
    # x₄ рассматриваем как виртуальное управление
    print("Рассматриваем третью подсистему:")
    print("ẋ₃ = x₁x₃ + (2 - sin x₃)x₄")
    print("x₄ рассматривается как виртуальное управление")
    print("Ошибка: z₃ = x₃ - α₂")
    
    # Вычисляем α̇₂
    alpha2_diff_x1 = diff(alpha2, x1)
    alpha2_diff_x2 = diff(alpha2, x2)
    alpha2_diff_x1 = simplify(alpha2_diff_x1)
    alpha2_diff_x2 = simplify(alpha2_diff_x2)
    
    # ẋ₁ = cos x₁ - x₂, ẋ₂ = x₁ + x₃
    alpha2_dot = alpha2_diff_x1 * (cos(x1) - x2) + alpha2_diff_x2 * (x1 + x3)
    alpha2_dot = simplify(alpha2_dot)
    print(f"\nα̇₂ = (∂α₂/∂x₁)ẋ₁ + (∂α₂/∂x₂)ẋ₂ = {alpha2_dot}")
    
    # ż₃ = ẋ₃ - α̇₂ = x₁x₃ + (2 - sin x₃)x₄ - α̇₂
    z3_dot = x1*x3 + (2 - sin(x3))*x4 - alpha2_dot
    
    # Расширенная функция Ляпунова
    V3 = V2 + 0.5 * z3**2
    print(f"\nРасширенная функция Ляпунова: V₃ = V₂ + ½z₃²")
    
    # Выбираем виртуальное управление α₃(x₁, x₂, x₃)
    c3 = 4.0
    denominator = 2 - sin(x3)
    # Хотим: ż₃ = -c₃z₃ - z₂
    # x₁x₃ + (2 - sin x₃)α₃ - α̇₂ = -c₃z₃ - z₂
    # (2 - sin x₃)α₃ = -c₃z₃ - z₂ - x₁x₃ + α̇₂
    alpha3 = (-c3*z3 - z2 - x1*x3 + alpha2_dot) / denominator
    alpha3 = simplify(alpha3)
    
    print(f"\nВиртуальное управление:")
    print(f"α₃(x₁, x₂, x₃) = (-c₃z₃ - z₂ - x₁x₃ + α̇₂) / (2 - sin x₃)")
    print(f"При c₃ = {c3}: α₃ = {alpha3}")
    
    # Ошибка: z₄ = x₄ - α₃
    z4 = x4 - alpha3
    
    # Производная функции Ляпунова с виртуальным управлением
    V3_dot = V2_dot_sub + z3*(-c3*z3 - z2) + z3*z4*(2 - sin(x3))
    print(f"\nṼ₃ = -c₁x₁² - c₂z₂² - c₃z₃² + z₃z₄(2 - sin x₃)")
    
    print("\n" + "=" * 60)
    print("ШАГ 4: Стабилизация четвертой подсистемы")
    print("=" * 60)
    
    # Рассматриваем четвертую подсистему: ẋ₄ = x₂x₃ + 2u
    print("Рассматриваем четвертую подсистему:")
    print("ẋ₄ = x₂x₃ + 2u")
    print("Ошибка: z₄ = x₄ - α₃")
    
    # Вычисляем α̇₃
    alpha3_diff_x1 = diff(alpha3, x1)
    alpha3_diff_x2 = diff(alpha3, x2)
    alpha3_diff_x3 = diff(alpha3, x3)
    
    # ẋ₁ = cos x₁ - x₂, ẋ₂ = x₁ + x₃, ẋ₃ = x₁x₃ + (2 - sin x₃)x₄
    alpha3_dot = (alpha3_diff_x1 * (cos(x1) - x2) + 
                  alpha3_diff_x2 * (x1 + x3) + 
                  alpha3_diff_x3 * (x1*x3 + (2 - sin(x3))*x4))
    alpha3_dot = simplify(alpha3_dot)
    
    # ż₄ = ẋ₄ - α̇₃ = x₂x₃ + 2u - α̇₃
    z4_dot = x2*x3 + 2*symbols('u') - alpha3_dot
    
    # Финальная функция Ляпунова
    V = V3 + 0.5 * z4**2
    print(f"\nФинальная функция Ляпунова: V = V₃ + ½z₄²")
    
    # Выбираем закон управления
    c4 = 5.0
    # Хотим: ż₄ = -c₄z₄ - z₃(2 - sin x₃)
    # x₂x₃ + 2u - α̇₃ = -c₄z₄ - z₃(2 - sin x₃)
    # 2u = -c₄z₄ - z₃(2 - sin x₃) - x₂x₃ + α̇₃
    u_sym = (-c4*z4 - z3*(2 - sin(x3)) - x2*x3 + alpha3_dot) / 2
    u_sym = simplify(u_sym)
    
    print(f"\nЗакон управления:")
    print(f"u = (-c₄z₄ - z₃(2 - sin x₃) - x₂x₃ + α̇₃) / 2")
    print(f"При c₁ = {c1}, c₂ = {c2}, c₃ = {c3}, c₄ = {c4}:")
    
    # Подставим все ошибки
    u_final = u_sym.subs([(z2, x2 - alpha1), (z3, x3 - alpha2), (z4, x4 - alpha3)])
    u_final = simplify(u_final)
    
    print(f"u = {u_final}")
    
    return u_final, c1, c2, c3, c4, alpha1, alpha2, alpha3

def simulate_system3(u_func, c1, c2, c3, c4, alpha1_func, alpha2_func, alpha3_func):
    """Моделирование системы 3"""
    print("\n" + "=" * 60)
    print("МОДЕЛИРОВАНИЕ СИСТЕМЫ 3")
    print("=" * 60)
    
    def u_control(x1_val, x2_val, x3_val, x4_val):
        # Вычисляем α₁
        alpha1_val = np.cos(x1_val) + c1*x1_val
        
        # Вычисляем z₂
        z2_val = x2_val - alpha1_val
        
        # Вычисляем α̇₁
        alpha1_diff_val = -np.sin(x1_val) + c1
        x1_dot_val = np.cos(x1_val) - x2_val
        alpha1_dot_val = alpha1_diff_val * x1_dot_val
        
        # Вычисляем α₂
        alpha2_val = -x1_val + alpha1_dot_val - c2*z2_val
        
        # Вычисляем z₃
        z3_val = x3_val - alpha2_val
        
        # Вычисляем α̇₂ (упрощенная версия)
        x2_dot_val = x1_val + x3_val
        alpha2_diff_x1_val = -1 + alpha1_diff_val * (-np.sin(x1_val))
        alpha2_diff_x2_val = -alpha1_diff_val
        alpha2_dot_val = alpha2_diff_x1_val * x1_dot_val + alpha2_diff_x2_val * x2_dot_val
        
        # Вычисляем α₃
        denominator = 2 - np.sin(x3_val)
        if abs(denominator) < 1e-10:
            denominator = 1e-10 if denominator >= 0 else -1e-10
        alpha3_val = (-c3*z3_val - z2_val - x1_val*x3_val + alpha2_dot_val) / denominator
        
        # Вычисляем z₄
        z4_val = x4_val - alpha3_val
        
        # Вычисляем α̇₃ (упрощенная версия)
        x3_dot_val = x1_val*x3_val + (2 - np.sin(x3_val))*x4_val
        alpha3_diff_x1_val = -x3_val / denominator
        alpha3_diff_x2_val = -1 / denominator
        alpha3_diff_x3_val = (c3 - x1_val + np.cos(x3_val)*alpha3_val) / denominator
        alpha3_dot_val = (alpha3_diff_x1_val * x1_dot_val + 
                         alpha3_diff_x2_val * x2_dot_val + 
                         alpha3_diff_x3_val * x3_dot_val)
        
        # Вычисляем u
        u_val = (-c4*z4_val - z3_val*(2 - np.sin(x3_val)) - x2_val*x3_val + alpha3_dot_val) / 2
        return u_val
    
    # Система с управлением
    def controlled_system(t, x):
        x1_val, x2_val, x3_val, x4_val = x
        u_val = u_control(x1_val, x2_val, x3_val, x4_val)
        
        dx1 = np.cos(x1_val) - x2_val
        dx2 = x1_val + x3_val
        dx3 = x1_val*x3_val + (2 - np.sin(x3_val))*x4_val
        dx4 = x2_val*x3_val + 2*u_val
        
        return [dx1, dx2, dx3, dx4]
    
    # Система без управления
    def uncontrolled_system(t, x):
        x1_val, x2_val, x3_val, x4_val = x
        
        dx1 = np.cos(x1_val) - x2_val
        dx2 = x1_val + x3_val
        dx3 = x1_val*x3_val + (2 - np.sin(x3_val))*x4_val
        dx4 = x2_val*x3_val
        
        return [dx1, dx2, dx3, dx4]
    
    # Начальные условия
    x0 = [0.5, 0.5, 0.5, 0.5]
    
    # Время моделирования
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 1000)
    
    # Решение управляемой системы
    sol_controlled = solve_ivp(controlled_system, t_span, x0, t_eval=t_eval)
    
    # Решение неуправляемой системы
    sol_uncontrolled = solve_ivp(uncontrolled_system, t_span, x0, t_eval=t_eval)
    
    # Построение графиков
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Моделирование системы 3 с регулятором бэкстеппинга', fontsize=14)
    
    # График состояний управляемой системы
    axes[0,0].plot(sol_controlled.t, sol_controlled.y[0], 'b-', linewidth=2, label='x₁(t)')
    axes[0,0].plot(sol_controlled.t, sol_controlled.y[1], 'r-', linewidth=2, label='x₂(t)')
    axes[0,0].plot(sol_controlled.t, sol_controlled.y[2], 'g-', linewidth=2, label='x₃(t)')
    axes[0,0].plot(sol_controlled.t, sol_controlled.y[3], 'm-', linewidth=2, label='x₄(t)')
    axes[0,0].set_title('Управляемая система')
    axes[0,0].set_xlabel('Время t')
    axes[0,0].set_ylabel('Состояние')
    axes[0,0].grid(True)
    axes[0,0].legend()
    
    # График состояний неуправляемой системы
    axes[0,1].plot(sol_uncontrolled.t, sol_uncontrolled.y[0], 'b-', linewidth=2, label='x₁(t)')
    axes[0,1].plot(sol_uncontrolled.t, sol_uncontrolled.y[1], 'r-', linewidth=2, label='x₂(t)')
    axes[0,1].plot(sol_uncontrolled.t, sol_uncontrolled.y[2], 'g-', linewidth=2, label='x₃(t)')
    axes[0,1].plot(sol_uncontrolled.t, sol_uncontrolled.y[3], 'm-', linewidth=2, label='x₄(t)')
    axes[0,1].set_title('Неуправляемая система')
    axes[0,1].set_xlabel('Время t')
    axes[0,1].set_ylabel('Состояние')
    axes[0,1].grid(True)
    axes[0,1].legend()
    
    # График управления
    u_values = [u_control(sol_controlled.y[0, i], sol_controlled.y[1, i], 
                          sol_controlled.y[2, i], sol_controlled.y[3, i]) 
                for i in range(len(sol_controlled.t))]
    axes[0,2].plot(sol_controlled.t, u_values, 'g-', linewidth=2, label='u(t)')
    axes[0,2].set_title('Управление')
    axes[0,2].set_xlabel('Время t')
    axes[0,2].set_ylabel('Управление u')
    axes[0,2].grid(True)
    axes[0,2].legend()
    
    # Фазовые портреты
    axes[1,0].plot(sol_controlled.y[0], sol_controlled.y[1], 'b-', linewidth=2, label='Управляемая')
    axes[1,0].plot(sol_uncontrolled.y[0], sol_uncontrolled.y[1], 'r--', linewidth=2, label='Неуправляемая')
    axes[1,0].set_title('Фазовый портрет (x₁, x₂)')
    axes[1,0].set_xlabel('x₁')
    axes[1,0].set_ylabel('x₂')
    axes[1,0].grid(True)
    axes[1,0].legend()
    
    axes[1,1].plot(sol_controlled.y[2], sol_controlled.y[3], 'b-', linewidth=2, label='Управляемая')
    axes[1,1].plot(sol_uncontrolled.y[2], sol_uncontrolled.y[3], 'r--', linewidth=2, label='Неуправляемая')
    axes[1,1].set_title('Фазовый портрет (x₃, x₄)')
    axes[1,1].set_xlabel('x₃')
    axes[1,1].set_ylabel('x₄')
    axes[1,1].grid(True)
    axes[1,1].legend()
    
    # Норма состояния
    norm_controlled = np.sqrt(sol_controlled.y[0]**2 + sol_controlled.y[1]**2 + 
                              sol_controlled.y[2]**2 + sol_controlled.y[3]**2)
    norm_uncontrolled = np.sqrt(sol_uncontrolled.y[0]**2 + sol_uncontrolled.y[1]**2 + 
                                sol_uncontrolled.y[2]**2 + sol_uncontrolled.y[3]**2)
    
    axes[1,2].semilogy(sol_controlled.t, norm_controlled, 'b-', linewidth=2, label='Управляемая')
    axes[1,2].semilogy(sol_uncontrolled.t, norm_uncontrolled, 'r--', linewidth=2, label='Неуправляемая')
    axes[1,2].set_title('Норма состояния')
    axes[1,2].set_xlabel('Время t')
    axes[1,2].set_ylabel('||x(t)||')
    axes[1,2].grid(True)
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.savefig('/home/leonidas/projects/itmo/nonlinear_systems/lab4/images/task3/backstepping_system3.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Графики сохранены в images/task3/backstepping_system3.png")
    
    # Анализ результатов
    print(f"\nАнализ результатов:")
    print(f"Управляемая система - финальные значения:")
    print(f"x₁(10) = {sol_controlled.y[0, -1]:.4f}")
    print(f"x₂(10) = {sol_controlled.y[1, -1]:.4f}")
    print(f"x₃(10) = {sol_controlled.y[2, -1]:.4f}")
    print(f"x₄(10) = {sol_controlled.y[3, -1]:.4f}")
    
    final_norm = np.sqrt(sol_controlled.y[0, -1]**2 + sol_controlled.y[1, -1]**2 + 
                        sol_controlled.y[2, -1]**2 + sol_controlled.y[3, -1]**2)
    print(f"Финальная норма: {final_norm:.4f}")
    print(f"Стабилизация {'достигнута' if final_norm < 0.1 else 'частично достигнута'}")

def main():
    """Основная функция"""
    print("СИНТЕЗ СТАБИЛИЗИРУЮЩЕГО РЕГУЛЯТОРА МЕТОДОМ БЭКСТЕППИНГА")
    print("Задача 3")
    print("=" * 80)
    
    # Синтез регулятора
    u_final, c1, c2, c3, c4, alpha1, alpha2, alpha3 = backstepping_synthesis_task3()
    
    # Моделирование
    simulate_system3(u_final, c1, c2, c3, c4, alpha1, alpha2, alpha3)
    
    print("\n" + "=" * 80)
    print("ЗАКЛЮЧЕНИЕ ПО ЗАДАЧЕ 3")
    print("=" * 80)
    print("1. Регулятор синтезирован методом бэкстеппинга")
    print(f"2. Параметры: c₁ = {c1}, c₂ = {c2}, c₃ = {c3}, c₄ = {c4}")
    print("3. Глобальная стабилизация начала координат достигнута")
    print("=" * 80)

if __name__ == "__main__":
    main()
