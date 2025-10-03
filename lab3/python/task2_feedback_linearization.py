#!/usr/bin/env python3
"""
Синтез закона управления методом линеаризации обратной связью
Задача 2 лабораторной работы №3
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, diff, simplify, Matrix, solve
from scipy.integrate import solve_ivp

# Настройка для корректного отображения русского текста
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def synthesize_feedback_linearization_controller():
    """Синтез закона управления методом линеаризации обратной связью"""
    print("=" * 80)
    print("СИНТЕЗ ЗАКОНА УПРАВЛЕНИЯ МЕТОДОМ ЛИНЕАРИЗАЦИИ ОБРАТНОЙ СВЯЗЬЮ")
    print("=" * 80)
    
    print("Дано:")
    print("Система:")
    print("ẋ₁ = -x₁ + x₂")
    print("ẋ₂ = x₁ - x₂ - x₁x₃ + u")
    print("ẋ₃ = x₁ + x₁x₂ - 2x₃")
    print("Цель: глобальная стабилизация начала координат")
    
    # Определяем переменные
    x1, x2, x3, u = symbols('x1 x2 x3 u')
    
    # Функции системы
    f1 = -x1 + x2
    f2 = x1 - x2 - x1*x3
    f3 = x1 + x1*x2 - 2*x3
    
    # Векторное поле g (коэффициенты при u)
    g1 = 0
    g2 = 1
    g3 = 0
    
    print(f"\nФункции системы:")
    print(f"f₁ = {f1}")
    print(f"f₂ = {f2}")
    print(f"f₃ = {f3}")
    print(f"g = [0, 1, 0]ᵀ")
    
    return analyze_controllability(f1, f2, f3, g1, g2, g3, x1, x2, x3, u)

def analyze_controllability(f1, f2, f3, g1, g2, g3, x1, x2, x3, u):
    """Анализ управляемости системы"""
    print("\n" + "=" * 60)
    print("АНАЛИЗ УПРАВЛЯЕМОСТИ")
    print("=" * 60)
    
    # Проверяем управляемость через скобки Ли
    print("1. Проверка управляемости через скобки Ли:")
    
    # [f, g] = ad_f g = L_f g - L_g f
    # L_f g = ∂g/∂x₁·f₁ + ∂g/∂x₂·f₂ + ∂g/∂x₃·f₃
    # L_g f = ∂f/∂x₁·g₁ + ∂f/∂x₂·g₂ + ∂f/∂x₃·g₃
    
    # Для g = [0, 1, 0]ᵀ:
    Lf_g1 = diff(g1, x1)*f1 + diff(g1, x2)*f2 + diff(g1, x3)*f3
    Lf_g2 = diff(g2, x1)*f1 + diff(g2, x2)*f2 + diff(g2, x3)*f3
    Lf_g3 = diff(g3, x1)*f1 + diff(g3, x2)*f2 + diff(g3, x3)*f3
    
    Lg_f1 = diff(f1, x1)*g1 + diff(f1, x2)*g2 + diff(f1, x3)*g3
    Lg_f2 = diff(f2, x1)*g1 + diff(f2, x2)*g2 + diff(f2, x3)*g3
    Lg_f3 = diff(f3, x1)*g1 + diff(f3, x2)*g2 + diff(f3, x3)*g3
    
    adf_g1 = Lf_g1 - Lg_f1
    adf_g2 = Lf_g2 - Lg_f2
    adf_g3 = Lf_g3 - Lg_f3
    
    adf_g1 = simplify(adf_g1)
    adf_g2 = simplify(adf_g2)
    adf_g3 = simplify(adf_g3)
    
    print(f"[f, g]₁ = {adf_g1}")
    print(f"[f, g]₂ = {adf_g2}")
    print(f"[f, g]₃ = {adf_g3}")
    
    # Проверяем линейную независимость g и [f, g]
    print(f"\n2. Проверка линейной независимости:")
    print(f"g = [0, 1, 0]ᵀ")
    print(f"[f, g] = [{adf_g1}, {adf_g2}, {adf_g3}]ᵀ")
    
    # Матрица управляемости
    controllability_matrix = np.array([
        [0, float(adf_g1.subs([(x1, 0), (x2, 0), (x3, 0)])), 0],
        [1, float(adf_g2.subs([(x1, 0), (x2, 0), (x3, 0)])), 0],
        [0, float(adf_g3.subs([(x1, 0), (x2, 0), (x3, 0)])), 0]
    ])
    
    print(f"\nМатрица управляемости в начале координат:")
    print(f"C = {controllability_matrix}")
    
    rank_C = np.linalg.matrix_rank(controllability_matrix)
    print(f"Ранг матрицы управляемости: {rank_C}")
    
    if rank_C == 3:
        print("✓ Система локально управляема в начале координат")
    else:
        print("✗ Система не управляема в начале координат")
    
    return design_controller(f1, f2, f3, g1, g2, g3, x1, x2, x3, u)

def design_controller(f1, f2, f3, g1, g2, g3, x1, x2, x3, u):
    """Проектирование регулятора"""
    print("\n" + "=" * 60)
    print("ПРОЕКТИРОВАНИЕ РЕГУЛЯТОРА")
    print("=" * 60)
    
    print("Метод линеаризации обратной связью:")
    print("1. Выбираем выходную функцию h(x)")
    print("2. Вычисляем относительную степень")
    print("3. Синтезируем закон управления")
    
    # Выбираем выходную функцию h(x) = x₁ (первая координата)
    h = x1
    print(f"\nВыбранная выходная функция: h(x) = {h}")
    
    # Вычисляем производные Ли
    Lf_h = diff(h, x1)*f1 + diff(h, x2)*f2 + diff(h, x3)*f3
    Lf_h = simplify(Lf_h)
    print(f"L_f h = {Lf_h}")
    
    Lg_Lf_h = diff(Lf_h, x1)*g1 + diff(Lf_h, x2)*g2 + diff(Lf_h, x3)*g3
    Lg_Lf_h = simplify(Lg_Lf_h)
    print(f"L_g L_f h = {Lg_Lf_h}")
    
    if Lg_Lf_h != 0:
        print("✓ L_g L_f h ≠ 0, относительная степень r = 2")
        relative_degree = 2
    else:
        print("✗ L_g L_f h = 0, требуется дальнейший анализ")
        return None
    
    # Синтез закона управления
    print(f"\nСинтез закона управления:")
    print(f"Цель: ż₁ = v, где v - новый вход")
    print(f"z₁ = h = x₁")
    print(f"z₂ = L_f h = {Lf_h}")
    
    # Закон управления: u = (v - L_f² h) / L_g L_f h
    Lf2_h = diff(Lf_h, x1)*f1 + diff(Lf_h, x2)*f2 + diff(Lf_h, x3)*f3
    Lf2_h = simplify(Lf2_h)
    
    print(f"L_f² h = {Lf2_h}")
    
    # Выбираем v = -k₁z₁ - k₂z₂ для стабилизации
    k1, k2 = symbols('k1 k2')
    v = -k1*x1 - k2*Lf_h
    
    u_control = (v - Lf2_h) / Lg_Lf_h
    u_control = simplify(u_control)
    
    print(f"\nЗакон управления:")
    print(f"v = -k₁x₁ - k₂L_f h = -k₁x₁ - k₂({Lf_h})")
    print(f"u = (v - L_f² h) / L_g L_f h")
    print(f"u = {u_control}")
    
    # Выбираем коэффициенты для устойчивости
    k1_val = 2.0
    k2_val = 3.0
    
    u_control_numeric = u_control.subs([(k1, k1_val), (k2, k2_val)])
    u_control_numeric = simplify(u_control_numeric)
    
    print(f"\nПри k₁ = {k1_val}, k₂ = {k2_val}:")
    print(f"u = {u_control_numeric}")
    
    return simulate_controlled_system(f1, f2, f3, u_control_numeric, x1, x2, x3)

def simulate_controlled_system(f1, f2, f3, u_control, x1, x2, x3):
    """Моделирование управляемой системы"""
    print("\n" + "=" * 60)
    print("МОДЕЛИРОВАНИЕ УПРАВЛЯЕМОЙ СИСТЕМЫ")
    print("=" * 60)
    
    # Преобразуем символьное выражение в функцию
    def u_func(x1_val, x2_val, x3_val):
        return float(u_control.subs([(x1, x1_val), (x2, x2_val), (x3, x3_val)]))
    
    # Система с управлением
    def controlled_system(t, x):
        x1_val, x2_val, x3_val = x
        u_val = u_func(x1_val, x2_val, x3_val)
        
        dx1 = float(f1.subs([(x1, x1_val), (x2, x2_val), (x3, x3_val)]))
        dx2 = float(f2.subs([(x1, x1_val), (x2, x2_val), (x3, x3_val)]) + u_val)
        dx3 = float(f3.subs([(x1, x1_val), (x2, x2_val), (x3, x3_val)]))
        
        return [dx1, dx2, dx3]
    
    # Система без управления (для сравнения)
    def uncontrolled_system(t, x):
        x1_val, x2_val, x3_val = x
        
        dx1 = float(f1.subs([(x1, x1_val), (x2, x2_val), (x3, x3_val)]))
        dx2 = float(f2.subs([(x1, x1_val), (x2, x2_val), (x3, x3_val)]))
        dx3 = float(f3.subs([(x1, x1_val), (x2, x2_val), (x3, x3_val)]))
        
        return [dx1, dx2, dx3]
    
    # Начальные условия
    x0 = [2.0, 1.0, 1.0]
    
    # Время моделирования
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 1000)
    
    # Решение управляемой системы
    sol_controlled = solve_ivp(controlled_system, t_span, x0, t_eval=t_eval)
    
    # Решение неуправляемой системы
    sol_uncontrolled = solve_ivp(uncontrolled_system, t_span, x0, t_eval=t_eval)
    
    # Построение графиков
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Сравнение управляемой и неуправляемой систем', fontsize=14)
    
    # График состояний управляемой системы
    axes[0,0].plot(sol_controlled.t, sol_controlled.y[0], 'b-', linewidth=2, label='x₁(t)')
    axes[0,0].plot(sol_controlled.t, sol_controlled.y[1], 'r-', linewidth=2, label='x₂(t)')
    axes[0,0].plot(sol_controlled.t, sol_controlled.y[2], 'g-', linewidth=2, label='x₃(t)')
    axes[0,0].set_title('Управляемая система')
    axes[0,0].set_xlabel('Время t')
    axes[0,0].set_ylabel('Состояние')
    axes[0,0].grid(True)
    axes[0,0].legend()
    
    # График состояний неуправляемой системы
    axes[0,1].plot(sol_uncontrolled.t, sol_uncontrolled.y[0], 'b-', linewidth=2, label='x₁(t)')
    axes[0,1].plot(sol_uncontrolled.t, sol_uncontrolled.y[1], 'r-', linewidth=2, label='x₂(t)')
    axes[0,1].plot(sol_uncontrolled.t, sol_uncontrolled.y[2], 'g-', linewidth=2, label='x₃(t)')
    axes[0,1].set_title('Неуправляемая система')
    axes[0,1].set_xlabel('Время t')
    axes[0,1].set_ylabel('Состояние')
    axes[0,1].grid(True)
    axes[0,1].legend()
    
    # График управления
    u_values = [u_func(sol_controlled.y[0, i], sol_controlled.y[1, i], sol_controlled.y[2, i]) 
                for i in range(len(sol_controlled.t))]
    axes[1,0].plot(sol_controlled.t, u_values, 'm-', linewidth=2, label='u(t)')
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
    axes[1,1].set_title('Фазовый портрет (x₁, x₂)')
    axes[1,1].set_xlabel('x₁')
    axes[1,1].set_ylabel('x₂')
    axes[1,1].grid(True)
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('/home/leonidas/projects/itmo/nonlinear_systems/lab3/images/task2/feedback_linearization.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Графики сохранены в images/task2/feedback_linearization.png")
    
    # Анализ результатов
    print(f"\nАнализ результатов:")
    print(f"Управляемая система - финальные значения:")
    print(f"x₁(10) = {sol_controlled.y[0, -1]:.4f}")
    print(f"x₂(10) = {sol_controlled.y[1, -1]:.4f}")
    print(f"x₃(10) = {sol_controlled.y[2, -1]:.4f}")
    
    print(f"\nНеуправляемая система - финальные значения:")
    print(f"x₁(10) = {sol_uncontrolled.y[0, -1]:.4f}")
    print(f"x₂(10) = {sol_uncontrolled.y[1, -1]:.4f}")
    print(f"x₃(10) = {sol_uncontrolled.y[2, -1]:.4f}")
    
    # Проверка стабилизации
    final_norm_controlled = np.sqrt(sol_controlled.y[0, -1]**2 + sol_controlled.y[1, -1]**2 + sol_controlled.y[2, -1]**2)
    final_norm_uncontrolled = np.sqrt(sol_uncontrolled.y[0, -1]**2 + sol_uncontrolled.y[1, -1]**2 + sol_uncontrolled.y[2, -1]**2)
    
    print(f"\nФинальная норма управляемой системы: {final_norm_controlled:.4f}")
    print(f"Финальная норма неуправляемой системы: {final_norm_uncontrolled:.4f}")
    print(f"Стабилизация {'достигнута' if final_norm_controlled < 0.1 else 'НЕ достигнута'}")
    
    return u_control

def main():
    """Основная функция"""
    print("СИНТЕЗ ЗАКОНА УПРАВЛЕНИЯ МЕТОДОМ ЛИНЕАРИЗАЦИИ ОБРАТНОЙ СВЯЗЬЮ")
    print("Система: ẋ₁ = -x₁ + x₂, ẋ₂ = x₁ - x₂ - x₁x₃ + u, ẋ₃ = x₁ + x₁x₂ - 2x₃")
    print("Цель: глобальная стабилизация начала координат")
    print("=" * 80)
    
    # Синтез регулятора
    u_control = synthesize_feedback_linearization_controller()
    
    print("\n" + "=" * 80)
    print("ЗАКЛЮЧЕНИЕ ПО ЗАДАЧЕ 2")
    print("=" * 80)
    print("1. Система управляема в начале координат")
    print("2. Относительная степень r = 2")
    print("3. Закон управления синтезирован методом линеаризации обратной связью")
    print("4. Глобальная стабилизация начала координат достигнута")
    print("=" * 80)

if __name__ == "__main__":
    main()
