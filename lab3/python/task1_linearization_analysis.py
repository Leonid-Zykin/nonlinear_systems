#!/usr/bin/env python3
"""
Анализ линеаризуемости по входу-выходу и минимально-фазовости
Задача 1 лабораторной работы №3
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, diff, simplify, Matrix, Function
from scipy.integrate import solve_ivp

# Настройка для корректного отображения русского текста
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def analyze_input_output_linearization():
    """Анализ линеаризуемости по входу-выходу системы"""
    print("=" * 80)
    print("АНАЛИЗ ЛИНЕАРИЗУЕМОСТИ ПО ВХОДУ-ВЫХОДУ")
    print("=" * 80)
    
    print("Дано:")
    print("Система:")
    print("ẋ₁ = -x₁ + x₂ - x₃")
    print("ẋ₂ = -x₁x₃ - x₂ + u")
    print("ẋ₃ = -x₁ + u")
    print("y = x₃")
    
    # Определяем переменные
    x1, x2, x3, u = symbols('x1 x2 x3 u')
    
    # Функции системы
    f1 = -x1 + x2 - x3
    f2 = -x1*x3 - x2 + u
    f3 = -x1 + u
    
    # Выход
    h = x3
    
    print(f"\nФункции системы:")
    print(f"f₁ = {f1}")
    print(f"f₂ = {f2}")
    print(f"f₃ = {f3}")
    print(f"h = {h}")
    
    return check_linearizability(f1, f2, f3, h, x1, x2, x3, u)

def check_linearizability(f1, f2, f3, h, x1, x2, x3, u):
    """Проверка линеаризуемости по входу-выходу"""
    print("\n" + "=" * 60)
    print("ПРОВЕРКА ЛИНЕАРИЗУЕМОСТИ ПО ВХОДУ-ВЫХОДУ")
    print("=" * 60)
    
    # Вычисляем производные Ли выходной функции
    print("1. Вычисление производных Ли выходной функции:")
    
    # L_f^0 h = h
    Lf0_h = h
    print(f"L_f⁰ h = {Lf0_h}")
    
    # L_f^1 h = ∂h/∂x₁·f₁ + ∂h/∂x₂·f₂ + ∂h/∂x₃·f₃
    Lf1_h = diff(h, x1)*f1 + diff(h, x2)*f2 + diff(h, x3)*f3
    Lf1_h = simplify(Lf1_h)
    print(f"L_f¹ h = {Lf1_h}")
    
    # L_g L_f^0 h = ∂h/∂x₁·g₁ + ∂h/∂x₂·g₂ + ∂h/∂x₃·g₃
    # где g₁ = 0, g₂ = 1, g₃ = 1 (коэффициенты при u)
    Lg_Lf0_h = diff(h, x1)*0 + diff(h, x2)*1 + diff(h, x3)*1
    Lg_Lf0_h = simplify(Lg_Lf0_h)
    print(f"L_g L_f⁰ h = {Lg_Lf0_h}")
    
    # Проверяем условие линеаризуемости
    print(f"\n2. Проверка условия линеаризуемости:")
    print(f"L_g L_f⁰ h = {Lg_Lf0_h}")
    
    if Lg_Lf0_h != 0:
        print("✓ L_g L_f⁰ h ≠ 0, система линеаризуема по входу-выходу")
        relative_degree = 1
    else:
        # Проверяем следующую производную
        Lg_Lf1_h = diff(Lf1_h, x1)*0 + diff(Lf1_h, x2)*1 + diff(Lf1_h, x3)*1
        Lg_Lf1_h = simplify(Lg_Lf1_h)
        print(f"L_g L_f¹ h = {Lg_Lf1_h}")
        
        if Lg_Lf1_h != 0:
            print("✓ L_g L_f¹ h ≠ 0, система линеаризуема по входу-выходу")
            relative_degree = 2
        else:
            print("✗ Система не линеаризуема по входу-выходу")
            return None
    
    print(f"\nОтносительная степень: r = {relative_degree}")
    
    return transform_to_normal_form(f1, f2, f3, h, x1, x2, x3, u, relative_degree)

def transform_to_normal_form(f1, f2, f3, h, x1, x2, x3, u, relative_degree):
    """Преобразование в нормальную форму"""
    print("\n" + "=" * 60)
    print("ПРЕОБРАЗОВАНИЕ В НОРМАЛЬНУЮ ФОРМУ")
    print("=" * 60)
    
    print(f"Относительная степень r = {relative_degree}")
    print(f"Размерность системы n = 3")
    print(f"Размерность внутренней динамики n - r = 2")
    
    # Выбираем координаты нормальной формы
    print(f"\nКоординаты нормальной формы:")
    
    # Внешние координаты (связанные с выходом)
    z1 = h  # z₁ = h = x₃
    print(f"z₁ = h = {z1}")
    
    if relative_degree == 1:
        # Для относительной степени 1
        z2 = diff(h, x1)*f1 + diff(h, x2)*f2 + diff(h, x3)*f3  # z₂ = L_f h
        z2 = simplify(z2)
        print(f"z₂ = L_f h = {z2}")
        
        # Внутренние координаты (независимые от выхода)
        # Выбираем η₁ = x₁, η₂ = x₂
        eta1 = x1
        eta2 = x2
        print(f"η₁ = x₁")
        print(f"η₂ = x₂")
        
    else:  # relative_degree == 2
        # Для относительной степени 2
        z2 = diff(h, x1)*f1 + diff(h, x2)*f2 + diff(h, x3)*f3  # z₂ = L_f h
        z2 = simplify(z2)
        print(f"z₂ = L_f h = {z2}")
        
        # Внутренние координаты
        # Выбираем η₁ = x₁
        eta1 = x1
        print(f"η₁ = x₁")
        print(f"η₂ = x₂ (или другая независимая координата)")
    
    # Вычисляем производные новых координат
    print(f"\nПроизводные координат нормальной формы:")
    
    # ẋ₁ = -x₁ + x₂ - x₃
    # ẋ₂ = -x₁x₃ - x₂ + u  
    # ẋ₃ = -x₁ + u
    
    # ż₁ = ẋ₃ = -x₁ + u
    dz1_dt = -x1 + u
    print(f"ż₁ = ẋ₃ = {dz1_dt}")
    
    if relative_degree == 1:
        # ż₂ = d/dt(L_f h) = d/dt(-x₁ + x₂ - x₃)
        # = -ẋ₁ + ẋ₂ - ẋ₃
        # = -(-x₁ + x₂ - x₃) + (-x₁x₃ - x₂ + u) - (-x₁ + u)
        # = x₁ - x₂ + x₃ - x₁x₃ - x₂ + u + x₁ - u
        # = 2x₁ - 2x₂ + x₃ - x₁x₃
        dz2_dt = 2*x1 - 2*x2 + x3 - x1*x3
        dz2_dt = simplify(dz2_dt)
        print(f"ż₂ = {dz2_dt}")
        
        # η̇₁ = ẋ₁ = -x₁ + x₂ - x₃
        deta1_dt = -x1 + x2 - x3
        print(f"η̇₁ = ẋ₁ = {deta1_dt}")
        
        # η̇₂ = ẋ₂ = -x₁x₃ - x₂ + u
        deta2_dt = -x1*x3 - x2 + u
        print(f"η̇₂ = ẋ₂ = {deta2_dt}")
        
    else:  # relative_degree == 2
        # Для относительной степени 2
        # ż₂ = L_f² h + L_g L_f h · u
        Lf2_h = diff(z2, x1)*f1 + diff(z2, x2)*f2 + diff(z2, x3)*f3
        Lf2_h = simplify(Lf2_h)
        Lg_Lf1_h = diff(z2, x1)*0 + diff(z2, x2)*1 + diff(z2, x3)*1
        Lg_Lf1_h = simplify(Lg_Lf1_h)
        
        dz2_dt = Lf2_h + Lg_Lf1_h * u
        dz2_dt = simplify(dz2_dt)
        print(f"ż₂ = L_f² h + L_g L_f h · u = {dz2_dt}")
    
    # Область определения преобразования
    print(f"\nОбласть определения преобразования:")
    print(f"Преобразование определено для всех x ∈ ℝ³")
    print(f"Обратное преобразование:")
    print(f"x₁ = η₁")
    print(f"x₂ = η₂") 
    print(f"x₃ = z₁")
    
    return check_minimum_phase(f1, f2, f3, h, x1, x2, x3, u, relative_degree)

def check_minimum_phase(f1, f2, f3, h, x1, x2, x3, u, relative_degree):
    """Проверка минимально-фазовости системы"""
    print("\n" + "=" * 60)
    print("ПРОВЕРКА МИНИМАЛЬНО-ФАЗОВОСТИ")
    print("=" * 60)
    
    print("Для проверки минимально-фазовости анализируем внутреннюю динамику")
    print("при нулевом выходе y = z₁ = 0")
    
    # При y = 0 имеем z₁ = x₃ = 0
    print(f"\nПри y = 0:")
    print(f"x₃ = 0")
    
    # Внутренняя динамика при x₃ = 0:
    print(f"\nВнутренняя динамика при x₃ = 0:")
    print(f"ẋ₁ = -x₁ + x₂ - 0 = -x₁ + x₂")
    print(f"ẋ₂ = -x₁·0 - x₂ + u = -x₂ + u")
    
    # Для минимально-фазовой системы внутренняя динамика должна быть устойчивой
    # при u = 0
    
    print(f"\nПри u = 0:")
    print(f"ẋ₁ = -x₁ + x₂")
    print(f"ẋ₂ = -x₂")
    
    # Матрица линеаризации внутренней динамики
    A_internal = np.array([[-1, 1],
                          [0, -1]])
    
    eigenvals = np.linalg.eigvals(A_internal)
    print(f"\nМатрица линеаризации внутренней динамики:")
    print(f"A = [[-1, 1], [0, -1]]")
    print(f"Собственные значения: λ = {eigenvals}")
    
    if np.all(np.real(eigenvals) < 0):
        print("✓ Все собственные значения имеют отрицательную вещественную часть")
        print("✓ Система минимально-фазовая")
        is_minimum_phase = True
    else:
        print("✗ Не все собственные значения имеют отрицательную вещественную часть")
        print("✗ Система НЕ минимально-фазовая")
        is_minimum_phase = False
    
    return simulate_system(f1, f2, f3, h, is_minimum_phase)

def simulate_system(f1, f2, f3, h, is_minimum_phase):
    """Моделирование системы"""
    print("\n" + "=" * 60)
    print("МОДЕЛИРОВАНИЕ СИСТЕМЫ")
    print("=" * 60)
    
    # Система: ẋ₁ = -x₁ + x₂ - x₃, ẋ₂ = -x₁x₃ - x₂ + u, ẋ₃ = -x₁ + u
    def system_dynamics(t, x):
        x1, x2, x3 = x
        u = 0  # Нулевое управление для анализа свободной динамики
        dx1 = -x1 + x2 - x3
        dx2 = -x1*x3 - x2 + u
        dx3 = -x1 + u
        return [dx1, dx2, dx3]
    
    # Начальные условия
    x0 = [1.0, 1.0, 1.0]
    
    # Время моделирования
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 1000)
    
    # Решение системы
    sol = solve_ivp(system_dynamics, t_span, x0, t_eval=t_eval)
    
    # Построение графиков
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Моделирование системы (u=0)', fontsize=14)
    
    # График состояний
    axes[0,0].plot(sol.t, sol.y[0], 'b-', linewidth=2, label='x₁(t)')
    axes[0,0].plot(sol.t, sol.y[1], 'r-', linewidth=2, label='x₂(t)')
    axes[0,0].plot(sol.t, sol.y[2], 'g-', linewidth=2, label='x₃(t)')
    axes[0,0].set_title('Состояния системы')
    axes[0,0].set_xlabel('Время t')
    axes[0,0].set_ylabel('Состояние')
    axes[0,0].grid(True)
    axes[0,0].legend()
    
    # График выхода
    y = sol.y[2]  # y = x₃
    axes[0,1].plot(sol.t, y, 'm-', linewidth=2, label='y(t) = x₃(t)')
    axes[0,1].set_title('Выход системы')
    axes[0,1].set_xlabel('Время t')
    axes[0,1].set_ylabel('Выход y')
    axes[0,1].grid(True)
    axes[0,1].legend()
    
    # Фазовый портрет (x₁, x₂)
    axes[1,0].plot(sol.y[0], sol.y[1], 'b-', linewidth=2, label='Траектория')
    axes[1,0].plot(x0[0], x0[1], 'ro', markersize=8, label='Начальная точка')
    axes[1,0].plot(0, 0, 'ko', markersize=8, label='Начало координат')
    axes[1,0].set_title('Фазовый портрет (x₁, x₂)')
    axes[1,0].set_xlabel('x₁')
    axes[1,0].set_ylabel('x₂')
    axes[1,0].grid(True)
    axes[1,0].legend()
    
    # Фазовый портрет (x₁, x₃)
    axes[1,1].plot(sol.y[0], sol.y[2], 'g-', linewidth=2, label='Траектория')
    axes[1,1].plot(x0[0], x0[2], 'ro', markersize=8, label='Начальная точка')
    axes[1,1].plot(0, 0, 'ko', markersize=8, label='Начало координат')
    axes[1,1].set_title('Фазовый портрет (x₁, x₃)')
    axes[1,1].set_xlabel('x₁')
    axes[1,1].set_ylabel('x₃')
    axes[1,1].grid(True)
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('/home/leonidas/projects/itmo/nonlinear_systems/lab3/images/task1/system_simulation.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Графики сохранены в images/task1/system_simulation.png")
    
    # Анализ результатов
    print(f"\nАнализ результатов:")
    print(f"Финальные значения: x₁(10) = {sol.y[0, -1]:.4f}, x₂(10) = {sol.y[1, -1]:.4f}, x₃(10) = {sol.y[2, -1]:.4f}")
    print(f"Система {'устойчива' if np.all(np.abs(sol.y[:, -1]) < 0.1) else 'неустойчива'}")
    
    return is_minimum_phase

def main():
    """Основная функция"""
    print("АНАЛИЗ ЛИНЕАРИЗУЕМОСТИ ПО ВХОДУ-ВЫХОДУ И МИНИМАЛЬНО-ФАЗОВОСТИ")
    print("Система: ẋ₁ = -x₁ + x₂ - x₃, ẋ₂ = -x₁x₃ - x₂ + u, ẋ₃ = -x₁ + u, y = x₃")
    print("=" * 80)
    
    # Анализ системы
    is_minimum_phase = analyze_input_output_linearization()
    
    print("\n" + "=" * 80)
    print("ЗАКЛЮЧЕНИЕ ПО ЗАДАЧЕ 1")
    print("=" * 80)
    print("1. Система линеаризуема по входу-выходу")
    print("2. Относительная степень r = 1")
    print("3. Нормальная форма получена")
    print(f"4. Система {'минимально-фазовая' if is_minimum_phase else 'НЕ минимально-фазовая'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
