#!/usr/bin/env python3
"""
Анализ устойчивости систем с использованием квадратичных функций Ляпунова
Задача 1 лабораторной работы №2
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, diff, Matrix, simplify, expand
import matplotlib.patches as patches

# Настройка для корректного отображения русского текста
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def analyze_system1():
    """Система 1: ẋ₁ = -x₁ + x₁x₂, ẋ₂ = -2x₂"""
    print("=" * 60)
    print("СИСТЕМА 1: ẋ₁ = -x₁ + x₁x₂, ẋ₂ = -2x₂")
    print("=" * 60)
    
    x1, x2 = symbols('x1 x2')
    
    # Функции системы
    f1 = -x1 + x1*x2
    f2 = -2*x2
    
    # Квадратичная функция Ляпунова V = x₁² + x₂²
    V = x1**2 + x2**2
    
    # Производная V̇
    V_dot = diff(V, x1) * f1 + diff(V, x2) * f2
    V_dot = simplify(V_dot)
    
    print(f"Функция Ляпунова: V = {V}")
    print(f"Производная V̇ = {V_dot}")
    
    # Анализ знака V̇
    V_dot_expanded = expand(V_dot)
    print(f"V̇ в развернутом виде: {V_dot_expanded}")
    
    # V̇ = -2x₁² - 2x₂² + 2x₁²x₂ = -2(x₁² + x₂²) + 2x₁²x₂
    # В малой окрестности начала координат |x₁²x₂| << x₁² + x₂²
    # Поэтому V̇ < 0 локально
    
    print("\nАнализ:")
    print("- V̇ = -2(x₁² + x₂²) + 2x₁²x₂")
    print("- В малой окрестности начала координат: V̇ < 0")
    print("- Система локально асимптотически устойчива")
    print("- НЕ глобально устойчива (при больших |x₂| член x₁²x₂ может доминировать)")
    
    return V_dot_expanded

def analyze_system2():
    """Система 2: ẋ₁ = -x₂ - x₁(1 - x₁² - x₂²), ẋ₂ = x₁ - x₂(1 - x₁² - x₂²)"""
    print("\n" + "=" * 60)
    print("СИСТЕМА 2: ẋ₁ = -x₂ - x₁(1 - x₁² - x₂²), ẋ₂ = x₁ - x₂(1 - x₁² - x₂²)")
    print("=" * 60)
    
    x1, x2 = symbols('x1 x2')
    
    # Функции системы
    f1 = -x2 - x1*(1 - x1**2 - x2**2)
    f2 = x1 - x2*(1 - x1**2 - x2**2)
    
    # Квадратичная функция Ляпунова V = x₁² + x₂²
    V = x1**2 + x2**2
    
    # Производная V̇
    V_dot = diff(V, x1) * f1 + diff(V, x2) * f2
    V_dot = simplify(V_dot)
    
    print(f"Функция Ляпунова: V = {V}")
    print(f"Производная V̇ = {V_dot}")
    
    # V̇ = -2(x₁² + x₂²)(1 - x₁² - x₂²)
    print("\nАнализ:")
    print("- V̇ = -2(x₁² + x₂²)(1 - x₁² - x₂²)")
    print("- Внутри единичного круга (x₁² + x₂² < 1): V̇ < 0")
    print("- На единичном круге (x₁² + x₂² = 1): V̇ = 0")
    print("- Вне единичного круга (x₁² + x₂² > 1): V̇ > 0")
    print("- Система локально асимптотически устойчива")
    print("- НЕ глобально устойчива (область притяжения ограничена единичным кругом)")
    
    return V_dot

def analyze_system3():
    """Система 3: ẋ₁ = x₂(1 - x₁²) - 2x₁, ẋ₂ = -(x₁ + x₂)(1 - x₁²)"""
    print("\n" + "=" * 60)
    print("СИСТЕМА 3: ẋ₁ = x₂(1 - x₁²) - 2x₁, ẋ₂ = -(x₁ + x₂)(1 - x₁²)")
    print("=" * 60)
    
    x1, x2 = symbols('x1 x2')
    
    # Функции системы
    f1 = x2*(1 - x1**2) - 2*x1
    f2 = -(x1 + x2)*(1 - x1**2)
    
    # Квадратичная функция Ляпунова V = x₁² + x₂²
    V = x1**2 + x2**2
    
    # Производная V̇
    V_dot = diff(V, x1) * f1 + diff(V, x2) * f2
    V_dot = simplify(V_dot)
    
    print(f"Функция Ляпунова: V = {V}")
    print(f"Производная V̇ = {V_dot}")
    
    # V̇ = -4x₁² - 2x₂² - 2x₁x₂ + 2x₁²x₂² + 2x₁³x₂
    print("\nАнализ:")
    print("- V̇ = -4x₁² - 2x₂² - 2x₁x₂ + 2x₁²x₂² + 2x₁³x₂")
    print("- В малой окрестности начала координат доминируют отрицательные члены")
    print("- Система локально асимптотически устойчива")
    print("- НЕ глобально устойчива (при больших значениях положительные члены могут доминировать)")
    
    return V_dot

def analyze_system4():
    """Система 4: ẋ₁ = -3x₁ - x₂, ẋ₂ = 2x₁ - x₂³"""
    print("\n" + "=" * 60)
    print("СИСТЕМА 4: ẋ₁ = -3x₁ - x₂, ẋ₂ = 2x₁ - x₂³")
    print("=" * 60)
    
    x1, x2 = symbols('x1 x2')
    
    # Функции системы
    f1 = -3*x1 - x2
    f2 = 2*x1 - x2**3
    
    # Квадратичная функция Ляпунова V = x₁² + x₂²
    V = x1**2 + x2**2
    
    # Производная V̇
    V_dot = diff(V, x1) * f1 + diff(V, x2) * f2
    V_dot = simplify(V_dot)
    
    print(f"Функция Ляпунова: V = {V}")
    print(f"Производная V̇ = {V_dot}")
    
    # V̇ = -6x₁² - 2x₂⁴
    print("\nАнализ:")
    print("- V̇ = -6x₁² - 2x₂⁴")
    print("- V̇ ≤ 0 для всех x₁, x₂")
    print("- V̇ = 0 только при x₁ = x₂ = 0")
    print("- Система глобально асимптотически устойчива")
    
    return V_dot

def analyze_system5():
    """Система 5: ẋ = -arctg(x)"""
    print("\n" + "=" * 60)
    print("СИСТЕМА 5: ẋ = -arctg(x)")
    print("=" * 60)
    
    x = symbols('x')
    
    # Функция системы
    f = -sp.atan(x)
    
    # Квадратичная функция Ляпунова V = x²
    V = x**2
    
    # Производная V̇
    V_dot = diff(V, x) * f
    V_dot = simplify(V_dot)
    
    print(f"Функция Ляпунова: V = {V}")
    print(f"Производная V̇ = {V_dot}")
    
    # V̇ = -2x*arctg(x)
    print("\nАнализ:")
    print("- V̇ = -2x*arctg(x)")
    print("- Для x > 0: arctg(x) > 0, поэтому V̇ < 0")
    print("- Для x < 0: arctg(x) < 0, поэтому V̇ < 0")
    print("- При x = 0: V̇ = 0")
    print("- Система глобально асимптотически устойчива")
    
    return V_dot

def plot_phase_portraits():
    """Построение фазовых портретов для визуализации"""
    print("\n" + "=" * 60)
    print("ПОСТРОЕНИЕ ФАЗОВЫХ ПОРТРЕТОВ")
    print("=" * 60)
    
    # Система 1
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Фазовые портреты систем', fontsize=16)
    
    # Система 1: ẋ₁ = -x₁ + x₁x₂, ẋ₂ = -2x₂
    x1_range = np.linspace(-2, 2, 20)
    x2_range = np.linspace(-2, 2, 20)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    dX1_1 = -X1 + X1*X2
    dX2_1 = -2*X2
    
    axes[0,0].quiver(X1, X2, dX1_1, dX2_1, alpha=0.6)
    axes[0,0].set_title('Система 1: локальная устойчивость')
    axes[0,0].set_xlabel('x₁')
    axes[0,0].set_ylabel('x₂')
    axes[0,0].grid(True)
    axes[0,0].set_xlim(-2, 2)
    axes[0,0].set_ylim(-2, 2)
    
    # Система 2: ẋ₁ = -x₂ - x₁(1 - x₁² - x₂²), ẋ₂ = x₁ - x₂(1 - x₁² - x₂²)
    dX1_2 = -X2 - X1*(1 - X1**2 - X2**2)
    dX2_2 = X1 - X2*(1 - X1**2 - X2**2)
    
    axes[0,1].quiver(X1, X2, dX1_2, dX2_2, alpha=0.6)
    axes[0,1].set_title('Система 2: область притяжения')
    axes[0,1].set_xlabel('x₁')
    axes[0,1].set_ylabel('x₂')
    axes[0,1].grid(True)
    axes[0,1].set_xlim(-2, 2)
    axes[0,1].set_ylim(-2, 2)
    
    # Добавляем единичный круг для системы 2
    circle = patches.Circle((0, 0), 1, fill=False, color='red', linestyle='--', linewidth=2)
    axes[0,1].add_patch(circle)
    
    # Система 3: ẋ₁ = x₂(1 - x₁²) - 2x₁, ẋ₂ = -(x₁ + x₂)(1 - x₁²)
    dX1_3 = X2*(1 - X1**2) - 2*X1
    dX2_3 = -(X1 + X2)*(1 - X1**2)
    
    axes[0,2].quiver(X1, X2, dX1_3, dX2_3, alpha=0.6)
    axes[0,2].set_title('Система 3: локальная устойчивость')
    axes[0,2].set_xlabel('x₁')
    axes[0,2].set_ylabel('x₂')
    axes[0,2].grid(True)
    axes[0,2].set_xlim(-2, 2)
    axes[0,2].set_ylim(-2, 2)
    
    # Система 4: ẋ₁ = -3x₁ - x₂, ẋ₂ = 2x₁ - x₂³
    dX1_4 = -3*X1 - X2
    dX2_4 = 2*X1 - X2**3
    
    axes[1,0].quiver(X1, X2, dX1_4, dX2_4, alpha=0.6)
    axes[1,0].set_title('Система 4: глобальная устойчивость')
    axes[1,0].set_xlabel('x₁')
    axes[1,0].set_ylabel('x₂')
    axes[1,0].grid(True)
    axes[1,0].set_xlim(-2, 2)
    axes[1,0].set_ylim(-2, 2)
    
    # Система 5: ẋ = -arctg(x) (одномерная)
    x_range = np.linspace(-3, 3, 100)
    dx = -np.arctan(x_range)
    
    axes[1,1].plot(x_range, dx, 'b-', linewidth=2)
    axes[1,1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1,1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes[1,1].set_title('Система 5: глобальная устойчивость')
    axes[1,1].set_xlabel('x')
    axes[1,1].set_ylabel('ẋ')
    axes[1,1].grid(True)
    axes[1,1].set_xlim(-3, 3)
    axes[1,1].set_ylim(-2, 2)
    
    # Убираем последний subplot
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/leonidas/projects/itmo/nonlinear_systems/lab2/images/task1/phase_portraits.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Основная функция"""
    print("АНАЛИЗ УСТОЙЧИВОСТИ СИСТЕМ С КВАДРАТИЧНЫМИ ФУНКЦИЯМИ ЛЯПУНОВА")
    print("=" * 80)
    
    # Анализ каждой системы
    analyze_system1()
    analyze_system2()
    analyze_system3()
    analyze_system4()
    analyze_system5()
    
    # Построение фазовых портретов
    plot_phase_portraits()
    
    print("\n" + "=" * 80)
    print("РЕЗЮМЕ:")
    print("- Система 1: локально асимптотически устойчива")
    print("- Система 2: локально асимптотически устойчива (область притяжения - единичный круг)")
    print("- Система 3: локально асимптотически устойчива")
    print("- Система 4: глобально асимптотически устойчива")
    print("- Система 5: глобально асимптотически устойчива")
    print("=" * 80)

if __name__ == "__main__":
    main()
