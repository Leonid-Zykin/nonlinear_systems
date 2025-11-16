#!/usr/bin/env python3
"""
Задание 2: Синтез стабилизирующего регулятора на основе скользящих режимов
для системы с неизвестными параметрами
Система: ẋ₁ = x₂ + a₁x₁ sin x₁, ẋ₂ = a₂x₁x₂ + 3u
Параметры: |a₁ - 1| ≤ 1, |a₂ - 1| ≤ 1 (т.е. a₁ ∈ [0, 2], a₂ ∈ [0, 2])
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp
from sympy import symbols, diff, simplify

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Символические переменные
x1, x2, u, a1, a2 = symbols('x1 x2 u a1 a2', real=True)

# Система
f1 = x2 + a1 * x1 * sp.sin(x1)
f2 = a2 * x1 * x2 + 3 * u

print("=" * 70)
print("ЗАДАНИЕ 2: СИНТЕЗ РЕГУЛЯТОРА ДЛЯ СИСТЕМЫ С НЕИЗВЕСТНЫМИ ПАРАМЕТРАМИ")
print("=" * 70)
print(f"\nСистема:")
print(f"ẋ₁ = {f1}")
print(f"ẋ₂ = {f2}")
print(f"\nПараметры: |a₁ - 1| ≤ 1, |a₂ - 1| ≤ 1")
print(f"  → a₁ ∈ [0, 2], a₂ ∈ [0, 2]")

# Выбираем поверхность скольжения: s = c₁x₁ + x₂ = 0
c1 = symbols('c1', real=True, positive=True)
s = c1 * x1 + x2

print(f"\nПоверхность скольжения: s = {s} = 0")

# Вычисляем производную поверхности скольжения
s_dot = diff(s, x1) * f1 + diff(s, x2) * f2
s_dot = simplify(s_dot)
print(f"\nṡ = {s_dot}")

# Для обеспечения ṡ = -k·s (k > 0) нужно:
# c₁(x₂ + a₁x₁ sin x₁) + a₂x₁x₂ + 3u = -k·s
# Отсюда: 3u = -k·s - c₁(x₂ + a₁x₁ sin x₁) - a₂x₁x₂
# u = [-k·s - c₁(x₂ + a₁x₁ sin x₁) - a₂x₁x₂] / 3

# Эквивалентное управление (номинальное, при a₁ = 1, a₂ = 1):
u_eq = - (c1 * (x2 + x1 * sp.sin(x1)) + x1 * x2) / 3
print(f"\nЭквивалентное управление (номинальное): u_eq = {simplify(u_eq)}")

# Разрывная часть для компенсации неопределенностей параметров
M = symbols('M', real=True, positive=True)

# Для непрерывного регулятора используем функцию насыщения
phi = symbols('phi', real=True, positive=True)  # ширина граничного слоя

print("\n" + "=" * 70)
print("НЕПРЕРЫВНЫЙ РЕГУЛЯТОР: u = u_eq - M·tanh(s/φ)")
print("=" * 70)

# Численное моделирование
def system_continuous(t, x, c1_val, M_val, phi_val, a1_val, a2_val):
    """Система с непрерывным регулятором"""
    x1_val, x2_val = x
    s_val = c1_val * x1_val + x2_val
    # Номинальное эквивалентное управление
    u_eq_val = - (c1_val * (x2_val + x1_val * np.sin(x1_val)) + x1_val * x2_val) / 3.0
    # Разрывная часть с насыщением
    u_sw_val = -M_val * np.tanh(s_val / phi_val)
    u_val = u_eq_val + u_sw_val
    
    dx1 = x2_val + a1_val * x1_val * np.sin(x1_val)
    dx2 = a2_val * x1_val * x2_val + 3.0 * u_val
    return [dx1, dx2]

def system_uncontrolled(t, x, a1_val, a2_val):
    """Неуправляемая система"""
    x1_val, x2_val = x
    dx1 = x2_val + a1_val * x1_val * np.sin(x1_val)
    dx2 = a2_val * x1_val * x2_val
    return [dx1, dx2]

# Параметры регулятора
c1_val = 2.5
M_val = 8.0
phi_val = 0.15  # ширина граничного слоя

# Параметры системы (различные комбинации для демонстрации робастности)
test_cases = [
    {"name": "Номинальные (a₁=1, a₂=1)", "a1": 1.0, "a2": 1.0, "color": "b"},
    {"name": "Максимальные (a₁=2, a₂=2)", "a1": 2.0, "a2": 2.0, "color": "g"},
    {"name": "Минимальные (a₁=0, a₂=0)", "a1": 0.0, "a2": 0.0, "color": "r"},
    {"name": "Смешанные (a₁=2, a₂=0)", "a1": 2.0, "a2": 0.0, "color": "m"},
]

# Начальные условия
x0 = [1.2, -0.9]
t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)

print("\n" + "=" * 70)
print("МОДЕЛИРОВАНИЕ")
print("=" * 70)
print(f"Параметры регулятора: c₁ = {c1_val}, M = {M_val}, φ = {phi_val}")
print(f"Начальные условия: x₀ = {x0}")

# Построение графиков
fig = plt.figure(figsize=(16, 12))

# Фазовые портреты для разных случаев
ax1 = plt.subplot(3, 3, 1)
for case in test_cases:
    sol = solve_ivp(
        lambda t, x: system_continuous(t, x, c1_val, M_val, phi_val, case["a1"], case["a2"]),
        t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
    )
    ax1.plot(sol.y[0], sol.y[1], color=case["color"], linewidth=2, label=case["name"])

ax1.plot(0, 0, 'ko', markersize=8, label='Цель (0,0)')
ax1.set_xlabel('$x_1$', fontsize=12)
ax1.set_ylabel('$x_2$', fontsize=12)
ax1.set_title('Фазовые портреты (разные параметры)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)
ax1.axis('equal')

# Временные зависимости x₁
ax2 = plt.subplot(3, 3, 2)
for case in test_cases:
    sol = solve_ivp(
        lambda t, x: system_continuous(t, x, c1_val, M_val, phi_val, case["a1"], case["a2"]),
        t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
    )
    ax2.plot(sol.t, sol.y[0], color=case["color"], linewidth=2, label=case["name"])

ax2.set_xlabel('$t$', fontsize=12)
ax2.set_ylabel('$x_1(t)$', fontsize=12)
ax2.set_title('Временная зависимость $x_1$', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

# Временные зависимости x₂
ax3 = plt.subplot(3, 3, 3)
for case in test_cases:
    sol = solve_ivp(
        lambda t, x: system_continuous(t, x, c1_val, M_val, phi_val, case["a1"], case["a2"]),
        t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
    )
    ax3.plot(sol.t, sol.y[1], color=case["color"], linewidth=2, label=case["name"])

ax3.set_xlabel('$t$', fontsize=12)
ax3.set_ylabel('$x_2(t)$', fontsize=12)
ax3.set_title('Временная зависимость $x_2$', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9)

# Поверхности скольжения
ax4 = plt.subplot(3, 3, 4)
for case in test_cases:
    sol = solve_ivp(
        lambda t, x: system_continuous(t, x, c1_val, M_val, phi_val, case["a1"], case["a2"]),
        t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
    )
    s_vals = c1_val * sol.y[0] + sol.y[1]
    ax4.plot(sol.t, s_vals, color=case["color"], linewidth=2, label=case["name"])

ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax4.set_xlabel('$t$', fontsize=12)
ax4.set_ylabel('$s(t)$', fontsize=12)
ax4.set_title('Поверхность скольжения', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=9)

# Норма состояния
ax5 = plt.subplot(3, 3, 5)
for case in test_cases:
    sol = solve_ivp(
        lambda t, x: system_continuous(t, x, c1_val, M_val, phi_val, case["a1"], case["a2"]),
        t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
    )
    norm_vals = np.sqrt(sol.y[0]**2 + sol.y[1]**2)
    ax5.plot(sol.t, norm_vals, color=case["color"], linewidth=2, label=case["name"])

ax5.set_xlabel('$t$', fontsize=12)
ax5.set_ylabel('$||x(t)||$', fontsize=12)
ax5.set_title('Норма вектора состояния', fontsize=12, fontweight='bold')
ax5.set_yscale('log')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=9)

# Управление для номинального случая
ax6 = plt.subplot(3, 3, 6)
case_nominal = test_cases[0]
sol_nominal = solve_ivp(
    lambda t, x: system_continuous(t, x, c1_val, M_val, phi_val, case_nominal["a1"], case_nominal["a2"]),
    t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
)
u_vals = []
for i in range(len(sol_nominal.t)):
    x1_n, x2_n = sol_nominal.y[0, i], sol_nominal.y[1, i]
    s_n = c1_val * x1_n + x2_n
    u_eq_n = - (c1_val * (x2_n + x1_n * np.sin(x1_n)) + x1_n * x2_n) / 3.0
    u_sw_n = -M_val * np.tanh(s_n / phi_val)
    u_vals.append(u_eq_n + u_sw_n)

ax6.plot(sol_nominal.t, u_vals, 'b-', linewidth=1.5)
ax6.set_xlabel('$t$', fontsize=12)
ax6.set_ylabel('$u(t)$', fontsize=12)
ax6.set_title('Управление (номинальный случай)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Сравнение с неуправляемой системой
ax7 = plt.subplot(3, 3, 7)
sol_uncontrolled = solve_ivp(
    lambda t, x: system_uncontrolled(t, x, case_nominal["a1"], case_nominal["a2"]),
    t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
)
ax7.plot(sol_uncontrolled.y[0], sol_uncontrolled.y[1], 'r--', linewidth=2, label='Без управления')
ax7.plot(sol_nominal.y[0], sol_nominal.y[1], 'b-', linewidth=2, label='С управлением')
ax7.plot(0, 0, 'ko', markersize=8, label='Цель (0,0)')
ax7.set_xlabel('$x_1$', fontsize=12)
ax7.set_ylabel('$x_2$', fontsize=12)
ax7.set_title('Сравнение с неуправляемой системой', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.legend()
ax7.axis('equal')

# Детальный вид поверхности скольжения
ax8 = plt.subplot(3, 3, 8)
for case in test_cases:
    sol = solve_ivp(
        lambda t, x: system_continuous(t, x, c1_val, M_val, phi_val, case["a1"], case["a2"]),
        t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
    )
    s_vals = c1_val * sol.y[0] + sol.y[1]
    ax8.plot(sol.t, s_vals, color=case["color"], linewidth=2, label=case["name"])

ax8.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax8.axhline(y=phi_val, color='gray', linestyle=':', alpha=0.5, label=f'Граничный слой ±φ')
ax8.axhline(y=-phi_val, color='gray', linestyle=':', alpha=0.5)
ax8.set_xlabel('$t$', fontsize=12)
ax8.set_ylabel('$s(t)$', fontsize=12)
ax8.set_title('Поверхность скольжения (детальный вид)', fontsize=12, fontweight='bold')
ax8.set_ylim([-0.3, 0.3])
ax8.grid(True, alpha=0.3)
ax8.legend(fontsize=8)

# Сравнение норм для всех случаев
ax9 = plt.subplot(3, 3, 9)
for case in test_cases:
    sol = solve_ivp(
        lambda t, x: system_continuous(t, x, c1_val, M_val, phi_val, case["a1"], case["a2"]),
        t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
    )
    norm_vals = np.sqrt(sol.y[0]**2 + sol.y[1]**2)
    ax9.plot(sol.t, norm_vals, color=case["color"], linewidth=2, label=case["name"])

ax9.set_xlabel('$t$', fontsize=12)
ax9.set_ylabel('$||x(t)||$', fontsize=12)
ax9.set_title('Сравнение норм состояния', fontsize=12, fontweight='bold')
ax9.set_yscale('log')
ax9.grid(True, alpha=0.3)
ax9.legend(fontsize=8)

plt.tight_layout()
plt.savefig('../images/task2_sliding_mode.png', dpi=300, bbox_inches='tight')
print("\nГрафик сохранен: images/task2_sliding_mode.png")
plt.close()

print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
print("=" * 70)
for case in test_cases:
    sol = solve_ivp(
        lambda t, x: system_continuous(t, x, c1_val, M_val, phi_val, case["a1"], case["a2"]),
        t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
    )
    norm_final = np.sqrt(sol.y[0, -1]**2 + sol.y[1, -1]**2)
    s_final = c1_val * sol.y[0, -1] + sol.y[1, -1]
    print(f"\n{case['name']}:")
    print(f"  Финальная норма: ||x(T)|| = {norm_final:.6f}")
    print(f"  Финальное значение s: s(T) = {s_final:.6f}")

