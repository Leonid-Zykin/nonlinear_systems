#!/usr/bin/env python3
"""
Анализ условий асимптотической устойчивости скалярной системы
Задача 2 лабораторной работы №2
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, diff, limit, simplify, expand

# Настройка для корректного отображения русского текста
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def analyze_scalar_system():
    """Анализ скалярной системы ẋ = axᵖ + h(x)"""
    print("=" * 80)
    print("АНАЛИЗ СКАЛЯРНОЙ СИСТЕМЫ: ẋ = axᵖ + h(x)")
    print("=" * 80)
    
    x, a, p, k = symbols('x a p k', real=True)
    
    # Условие на h(x): |h(x)| ≤ k|x|^(p+1)
    print("Дано:")
    print("- Система: ẋ = axᵖ + h(x)")
    print("- p - натуральное число")
    print("- |h(x)| ≤ k|x|^(p+1) в некоторой окрестности начала координат")
    print("- k > 0")
    
    print("\n" + "=" * 60)
    print("АНАЛИЗ УСТОЙЧИВОСТИ")
    print("=" * 60)
    
    # Рассмотрим различные случаи
    print("\n1. СЛУЧАЙ: p - четное число")
    print("-" * 40)
    print("При p четном: xᵖ ≥ 0 для всех x")
    print("Если a < 0:")
    print("  - axᵖ ≤ 0 для всех x")
    print("  - В малой окрестности: |h(x)| ≤ k|x|^(p+1) << |axᵖ|")
    print("  - Поэтому ẋ ≈ axᵖ < 0 при x > 0 и ẋ > 0 при x < 0")
    print("  - Система асимптотически устойчива")
    print("Если a > 0:")
    print("  - axᵖ ≥ 0 для всех x")
    print("  - Система неустойчива")
    
    print("\n2. СЛУЧАЙ: p - нечетное число")
    print("-" * 40)
    print("При p нечетном: xᵖ имеет тот же знак, что и x")
    print("Если a < 0:")
    print("  - axᵖ < 0 при x > 0 и axᵖ > 0 при x < 0")
    print("  - В малой окрестности: |h(x)| ≤ k|x|^(p+1) << |axᵖ|")
    print("  - Поэтому ẋ < 0 при x > 0 и ẋ > 0 при x < 0")
    print("  - Система асимптотически устойчива")
    print("Если a > 0:")
    print("  - axᵖ > 0 при x > 0 и axᵖ < 0 при x < 0")
    print("  - Система неустойчива")
    
    print("\n3. СЛУЧАЙ: a = 0")
    print("-" * 40)
    print("При a = 0: ẋ = h(x)")
    print("Условие |h(x)| ≤ k|x|^(p+1) означает:")
    print("- h(x) = O(x^(p+1)) при x → 0")
    print("- Система может быть устойчива или неустойчива в зависимости от h(x)")
    print("- Требуется дополнительный анализ")
    
    print("\n" + "=" * 60)
    print("УСЛОВИЯ АСИМПТОТИЧЕСКОЙ УСТОЙЧИВОСТИ")
    print("=" * 60)
    
    print("\nОТВЕТ:")
    print("Система асимптотически устойчива при выполнении одного из условий:")
    print("1. a < 0 (для любого натурального p)")
    print("2. a = 0 и дополнительный анализ функции h(x)")
    
    print("\nОбоснование:")
    print("- При a < 0 главный член axᵖ обеспечивает возврат к началу координат")
    print("- При a = 0 поведение определяется функцией h(x)")
    print("- При a > 0 система неустойчива")
    
    return analyze_specific_cases()

def analyze_specific_cases():
    """Анализ конкретных случаев"""
    print("\n" + "=" * 60)
    print("АНАЛИЗ КОНКРЕТНЫХ СЛУЧАЕВ")
    print("=" * 60)
    
    x = symbols('x')
    
    # Случай 1: p = 1, a = -1
    print("\nСлучай 1: p = 1, a = -1")
    print("Система: ẋ = -x + h(x), где |h(x)| ≤ kx²")
    print("Линейная часть: ẋ = -x (экспоненциально устойчива)")
    print("Нелинейное возмущение h(x) = O(x²) не нарушает устойчивость")
    
    # Случай 2: p = 2, a = -1
    print("\nСлучай 2: p = 2, a = -1")
    print("Система: ẋ = -x² + h(x), где |h(x)| ≤ kx³")
    print("Главный член: -x² < 0 для всех x ≠ 0")
    print("Нелинейное возмущение h(x) = O(x³) не нарушает устойчивость")
    
    # Случай 3: p = 1, a = 0
    print("\nСлучай 3: p = 1, a = 0")
    print("Система: ẋ = h(x), где |h(x)| ≤ kx²")
    print("Требуется анализ конкретного вида h(x)")
    print("Если h(x) = -x³, то система устойчива")
    print("Если h(x) = x³, то система неустойчива")
    
    return plot_examples()

def plot_examples():
    """Построение примеров для различных случаев"""
    print("\n" + "=" * 60)
    print("ПОСТРОЕНИЕ ПРИМЕРОВ")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Примеры скалярных систем ẋ = axᵖ + h(x)', fontsize=14)
    
    x = np.linspace(-2, 2, 1000)
    
    # Случай 1: p = 1, a = -1, h(x) = -0.1x²
    dx1 = -x - 0.1*x**2
    axes[0,0].plot(x, dx1, 'b-', linewidth=2, label='ẋ = -x - 0.1x²')
    axes[0,0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0,0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes[0,0].set_title('p=1, a=-1: Устойчива')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('ẋ')
    axes[0,0].grid(True)
    axes[0,0].legend()
    
    # Случай 2: p = 2, a = -1, h(x) = -0.1x³
    dx2 = -x**2 - 0.1*x**3
    axes[0,1].plot(x, dx2, 'r-', linewidth=2, label='ẋ = -x² - 0.1x³')
    axes[0,1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0,1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes[0,1].set_title('p=2, a=-1: Устойчива')
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('ẋ')
    axes[0,1].grid(True)
    axes[0,1].legend()
    
    # Случай 3: p = 1, a = 1, h(x) = -0.1x²
    dx3 = x - 0.1*x**2
    axes[1,0].plot(x, dx3, 'g-', linewidth=2, label='ẋ = x - 0.1x²')
    axes[1,0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1,0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes[1,0].set_title('p=1, a=1: Неустойчива')
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('ẋ')
    axes[1,0].grid(True)
    axes[1,0].legend()
    
    # Случай 4: p = 1, a = 0, h(x) = -x³
    dx4 = -x**3
    axes[1,1].plot(x, dx4, 'm-', linewidth=2, label='ẋ = -x³')
    axes[1,1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1,1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes[1,1].set_title('p=1, a=0: Устойчива')
    axes[1,1].set_xlabel('x')
    axes[1,1].set_ylabel('ẋ')
    axes[1,1].grid(True)
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('/home/leonidas/projects/itmo/nonlinear_systems/lab2/images/task2/scalar_systems.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Графики сохранены в images/task2/scalar_systems.png")

def main():
    """Основная функция"""
    analyze_scalar_system()
    
    print("\n" + "=" * 80)
    print("ЗАКЛЮЧЕНИЕ ПО ЗАДАЧЕ 2")
    print("=" * 80)
    print("Условие асимптотической устойчивости скалярной системы ẋ = axᵖ + h(x):")
    print("a < 0")
    print("\nОбоснование:")
    print("- При a < 0 главный член axᵖ обеспечивает возврат к началу координат")
    print("- Нелинейное возмущение h(x) = O(x^(p+1)) не нарушает устойчивость")
    print("- При a ≥ 0 система неустойчива или требует дополнительного анализа")
    print("=" * 80)

if __name__ == "__main__":
    main()
