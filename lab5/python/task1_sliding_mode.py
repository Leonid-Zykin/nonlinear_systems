#!/usr/bin/env python3
"""
Задание 1: Синтез разрывного и непрерывного регуляторов на основе скользящих режимов
Система: ẋ₁ = x₂ + sin x₁, ẋ₂ = θ₁x₁² + (2 + θ₂)u
Параметры: |θ₁| ≤ 1, |θ₂| ≤ 1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp
from sympy import symbols, diff, simplify

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Символические переменные
x1, x2, u, theta1, theta2 = symbols('x1 x2 u theta1 theta2', real=True)

# Система
f1 = x2 + sp.sin(x1)
f2 = theta1 * x1**2 + (2 + theta2) * u

print("=" * 70)
print("ЗАДАНИЕ 1: СИНТЕЗ РЕГУЛЯТОРОВ НА ОСНОВЕ СКОЛЬЗЯЩИХ РЕЖИМОВ")
print("=" * 70)
print(f"\nСистема:")
print(f"ẋ₁ = {f1}")
print(f"ẋ₂ = {f2}")
print(f"\nПараметры: |θ₁| ≤ 1, |θ₂| ≤ 1")

# Выбираем поверхность скольжения: s = c₁x₁ + x₂ = 0
# Для стабилизации в начало координат выбираем c₁ > 0
c1 = symbols('c1', real=True, positive=True)
s = c1 * x1 + x2

print(f"\nПоверхность скольжения: s = {s} = 0")

# Вычисляем производную поверхности скольжения
s_dot = diff(s, x1) * f1 + diff(s, x2) * f2
s_dot = simplify(s_dot)
print(f"\nṡ = {s_dot}")

# Для обеспечения ṡ = -k·s (k > 0) нужно:
# c₁(x₂ + sin x₁) + θ₁x₁² + (2 + θ₂)u = -k·s
# Отсюда: (2 + θ₂)u = -k·s - c₁(x₂ + sin x₁) - θ₁x₁²
# u = [-k·s - c₁(x₂ + sin x₁) - θ₁x₁²] / (2 + θ₂)

# Учитывая неопределенность параметров, используем робастный закон:
# u = u_eq + u_sw, где u_eq - эквивалентное управление, u_sw - разрывная часть

# Эквивалентное управление (при θ₁ = 0, θ₂ = 0):
u_eq = - (c1 * (x2 + sp.sin(x1))) / 2
print(f"\nЭквивалентное управление (номинальное): u_eq = {simplify(u_eq)}")

# Разрывная часть для компенсации неопределенностей
# u_sw = -M·sign(s), где M выбирается из условия робастности
M = symbols('M', real=True, positive=True)

# Для непрерывного регулятора используем функцию насыщения или гиперболический тангенс
phi = symbols('phi', real=True, positive=True)  # ширина граничного слоя

print("\n" + "=" * 70)
print("РАЗРЫВНЫЙ РЕГУЛЯТОР: u = u_eq - M·sign(s)")
print("=" * 70)

print("\n" + "=" * 70)
print("НЕПРЕРЫВНЫЙ РЕГУЛЯТОР: u = u_eq - M·tanh(s/φ)")
print("=" * 70)

# Численное моделирование
def system_discontinuous(t, x, c1_val, M_val, k_val, theta1_val, theta2_val):
    """Система с разрывным регулятором"""
    x1_val, x2_val = x
    s_val = c1_val * x1_val + x2_val
    # Эквивалентное управление: (2 + θ₂)u_eq = -k·s - c₁(x₂ + sin x₁) - θ₁x₁²
    # При номинальных значениях (θ₁=0, θ₂=0): u_eq = -[k·s + c₁(x₂ + sin x₁)] / 2
    u_eq_val = - (k_val * s_val + c1_val * (x2_val + np.sin(x1_val))) / 2.0
    u_sw_val = -M_val * np.sign(s_val) if abs(s_val) > 1e-10 else 0.0
    u_val = u_eq_val + u_sw_val
    
    dx1 = x2_val + np.sin(x1_val)
    dx2 = theta1_val * x1_val**2 + (2 + theta2_val) * u_val
    return [dx1, dx2]

def system_continuous(t, x, c1_val, M_val, k_val, phi_val, theta1_val, theta2_val):
    """Система с непрерывным регулятором"""
    x1_val, x2_val = x
    s_val = c1_val * x1_val + x2_val
    # Эквивалентное управление: (2 + θ₂)u_eq = -k·s - c₁(x₂ + sin x₁) - θ₁x₁²
    # При номинальных значениях (θ₁=0, θ₂=0): u_eq = -[k·s + c₁(x₂ + sin x₁)] / 2
    u_eq_val = - (k_val * s_val + c1_val * (x2_val + np.sin(x1_val))) / 2.0
    u_sw_val = -M_val * np.tanh(s_val / phi_val)
    u_val = u_eq_val + u_sw_val
    
    dx1 = x2_val + np.sin(x1_val)
    dx2 = theta1_val * x1_val**2 + (2 + theta2_val) * u_val
    return [dx1, dx2]

def system_uncontrolled(t, x, theta1_val):
    """Неуправляемая система"""
    x1_val, x2_val = x
    dx1 = x2_val + np.sin(x1_val)
    dx2 = theta1_val * x1_val**2
    return [dx1, dx2]

# Параметры регулятора
c1_val = 2.0
M_val = 5.0
phi_val = 0.1  # ширина граничного слоя для непрерывного регулятора
k_val = 3.0  # коэффициент в ṡ = -k·s

# Параметры системы (наихудший случай для демонстрации робастности)
theta1_val = 1.0  # максимальное значение
theta2_val = 1.0  # максимальное значение

# Начальные условия
x0 = [1.5, -0.8]
# Уменьшаем время моделирования для быстрого выполнения
t_span = (0, 3)
t_eval = np.linspace(0, 3, 300)  # Еще меньше точек для быстрого выполнения

# Для неуправляемой системы используем меньшее время, чтобы избежать слишком больших значений
t_span_uncontrolled = (0, 1.0)
t_eval_uncontrolled = np.linspace(0, 1.0, 200)

print("\n" + "=" * 70)
print("МОДЕЛИРОВАНИЕ")
print("=" * 70)
print(f"Параметры регулятора: c₁ = {c1_val}, M = {M_val}, φ = {phi_val}")
print(f"Параметры системы: θ₁ = {theta1_val}, θ₂ = {theta2_val}")
print(f"Начальные условия: x₀ = {x0}")

# Решение для разрывного регулятора (упрощенные параметры для быстрого выполнения)
print("Интегрирование разрывного регулятора...")
sol_discontinuous = solve_ivp(
    lambda t, x: system_discontinuous(t, x, c1_val, M_val, k_val, theta1_val, theta2_val),
    t_span, x0, t_eval=t_eval, rtol=1e-4, atol=1e-7, max_step=0.1, dense_output=False, method='RK45'
)

# Решение для непрерывного регулятора
print("Интегрирование непрерывного регулятора...")
sol_continuous = solve_ivp(
    lambda t, x: system_continuous(t, x, c1_val, M_val, k_val, phi_val, theta1_val, theta2_val),
    t_span, x0, t_eval=t_eval, rtol=1e-4, atol=1e-7, max_step=0.1, dense_output=False, method='RK45'
)

# Решение для неуправляемой системы (ограниченное время)
print("Интегрирование неуправляемой системы...")
sol_uncontrolled = solve_ivp(
    lambda t, x: system_uncontrolled(t, x, theta1_val),
    t_span_uncontrolled, x0, t_eval=t_eval_uncontrolled, rtol=1e-4, atol=1e-7, max_step=0.1, dense_output=False, method='RK45'
)

# Вычисление поверхности скольжения и управления (векторизованное для скорости)
print("Вычисление управления...")
s_discontinuous = c1_val * sol_discontinuous.y[0] + sol_discontinuous.y[1]
s_continuous = c1_val * sol_continuous.y[0] + sol_continuous.y[1]

# Векторизованное вычисление управления
x1_d = sol_discontinuous.y[0]
x2_d = sol_discontinuous.y[1]
u_eq_d = - (k_val * s_discontinuous + c1_val * (x2_d + np.sin(x1_d))) / 2.0
u_sw_d = -M_val * np.sign(s_discontinuous)
u_sw_d[np.abs(s_discontinuous) < 1e-10] = 0.0
u_discontinuous = u_eq_d + u_sw_d

x1_c = sol_continuous.y[0]
x2_c = sol_continuous.y[1]
u_eq_c = - (k_val * s_continuous + c1_val * (x2_c + np.sin(x1_c))) / 2.0
u_sw_c = -M_val * np.tanh(s_continuous / phi_val)
u_continuous = u_eq_c + u_sw_c

# Построение графиков
fig = plt.figure(figsize=(16, 12))

# Фазовые портреты
ax1 = plt.subplot(3, 3, 1)
# Ограничиваем показ неуправляемой системы, чтобы не искажать масштаб
mask_uncontrolled = np.sqrt(sol_uncontrolled.y[0]**2 + sol_uncontrolled.y[1]**2) < 5
ax1.plot(sol_uncontrolled.y[0][mask_uncontrolled], sol_uncontrolled.y[1][mask_uncontrolled], 
         'r--', linewidth=2, label='Без управления', alpha=0.7)
ax1.plot(sol_discontinuous.y[0], sol_discontinuous.y[1], 'b-', linewidth=2, label='Разрывный регулятор')
ax1.plot(0, 0, 'ko', markersize=8, label='Цель (0,0)')
ax1.set_xlabel('$x_1$', fontsize=12)
ax1.set_ylabel('$x_2$', fontsize=12)
ax1.set_title('Фазовый портрет (разрывный регулятор)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
# Автоматический масштаб на основе управляемых систем
x1_range = max(np.max(np.abs(sol_discontinuous.y[0])), 1.5)
x2_range = max(np.max(np.abs(sol_discontinuous.y[1])), 0.8)
ax1.set_xlim([-max(2, 1.2*x1_range), max(2, 1.2*x1_range)])
ax1.set_ylim([-max(3, 1.2*x2_range), max(3, 1.2*x2_range)])

ax2 = plt.subplot(3, 3, 2)
ax2.plot(sol_uncontrolled.y[0][mask_uncontrolled], sol_uncontrolled.y[1][mask_uncontrolled], 
         'r--', linewidth=2, label='Без управления', alpha=0.7)
ax2.plot(sol_continuous.y[0], sol_continuous.y[1], 'g-', linewidth=2, label='Непрерывный регулятор')
ax2.plot(0, 0, 'ko', markersize=8, label='Цель (0,0)')
ax2.set_xlabel('$x_1$', fontsize=12)
ax2.set_ylabel('$x_2$', fontsize=12)
ax2.set_title('Фазовый портрет (непрерывный регулятор)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
# Автоматический масштаб на основе управляемых систем
x1_range = max(np.max(np.abs(sol_continuous.y[0])), 1.5)
x2_range = max(np.max(np.abs(sol_continuous.y[1])), 0.8)
ax2.set_xlim([-max(2, 1.2*x1_range), max(2, 1.2*x1_range)])
ax2.set_ylim([-max(3, 1.2*x2_range), max(3, 1.2*x2_range)])

# Поверхность скольжения
ax3 = plt.subplot(3, 3, 3)
ax3.plot(sol_discontinuous.t, s_discontinuous, 'b-', linewidth=2, label='Разрывный')
ax3.plot(sol_continuous.t, s_continuous, 'g-', linewidth=2, label='Непрерывный')
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax3.set_xlabel('$t$', fontsize=12)
ax3.set_ylabel('$s(t)$', fontsize=12)
ax3.set_title('Поверхность скольжения', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Временные зависимости x₁
ax4 = plt.subplot(3, 3, 4)
# Показываем неуправляемую систему только до момента, когда она не слишком расходится
mask_time = sol_uncontrolled.t <= 1.2
ax4.plot(sol_uncontrolled.t[mask_time], sol_uncontrolled.y[0][mask_time], 
         'r--', linewidth=2, label='Без управления', alpha=0.7)
ax4.plot(sol_discontinuous.t, sol_discontinuous.y[0], 'b-', linewidth=2, label='Разрывный')
ax4.plot(sol_continuous.t, sol_continuous.y[0], 'g-', linewidth=2, label='Непрерывный')
ax4.set_xlabel('$t$', fontsize=12)
ax4.set_ylabel('$x_1(t)$', fontsize=12)
ax4.set_title('Временная зависимость $x_1$', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()
# Автоматический масштаб для управляемых систем
x1_max = max(np.max(np.abs(sol_discontinuous.y[0])), np.max(np.abs(sol_continuous.y[0])))
ax4.set_ylim([-max(2, 1.2*x1_max), max(2, 1.2*x1_max)])

# Временные зависимости x₂
ax5 = plt.subplot(3, 3, 5)
ax5.plot(sol_uncontrolled.t[mask_time], sol_uncontrolled.y[1][mask_time], 
         'r--', linewidth=2, label='Без управления', alpha=0.7)
ax5.plot(sol_discontinuous.t, sol_discontinuous.y[1], 'b-', linewidth=2, label='Разрывный')
ax5.plot(sol_continuous.t, sol_continuous.y[1], 'g-', linewidth=2, label='Непрерывный')
ax5.set_xlabel('$t$', fontsize=12)
ax5.set_ylabel('$x_2(t)$', fontsize=12)
ax5.set_title('Временная зависимость $x_2$', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend()
# Автоматический масштаб для управляемых систем
x2_max = max(np.max(np.abs(sol_discontinuous.y[1])), np.max(np.abs(sol_continuous.y[1])))
ax5.set_ylim([-max(3, 1.2*x2_max), max(3, 1.2*x2_max)])

# Управление (разрывное)
ax6 = plt.subplot(3, 3, 6)
ax6.plot(sol_discontinuous.t, u_discontinuous, 'b-', linewidth=1.5)
ax6.set_xlabel('$t$', fontsize=12)
ax6.set_ylabel('$u(t)$', fontsize=12)
ax6.set_title('Управление (разрывный регулятор)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Управление (непрерывное)
ax7 = plt.subplot(3, 3, 7)
ax7.plot(sol_continuous.t, u_continuous, 'g-', linewidth=1.5)
ax7.set_xlabel('$t$', fontsize=12)
ax7.set_ylabel('$u(t)$', fontsize=12)
ax7.set_title('Управление (непрерывный регулятор)', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)

# Норма состояния
ax8 = plt.subplot(3, 3, 8)
norm_uncontrolled = np.sqrt(sol_uncontrolled.y[0]**2 + sol_uncontrolled.y[1]**2)
norm_discontinuous = np.sqrt(sol_discontinuous.y[0]**2 + sol_discontinuous.y[1]**2)
norm_continuous = np.sqrt(sol_continuous.y[0]**2 + sol_continuous.y[1]**2)
# Ограничиваем показ неуправляемой системы
ax8.plot(sol_uncontrolled.t[mask_time], norm_uncontrolled[mask_time], 
         'r--', linewidth=2, label='Без управления', alpha=0.7)
ax8.plot(sol_discontinuous.t, norm_discontinuous, 'b-', linewidth=2, label='Разрывный')
ax8.plot(sol_continuous.t, norm_continuous, 'g-', linewidth=2, label='Непрерывный')
ax8.set_xlabel('$t$', fontsize=12)
ax8.set_ylabel('$||x(t)||$', fontsize=12)
ax8.set_title('Норма вектора состояния', fontsize=12, fontweight='bold')
ax8.set_yscale('log')
ax8.set_ylim([1e-4, 10])
ax8.grid(True, alpha=0.3)
ax8.legend()

# Сравнение поверхностей скольжения (увеличенный масштаб)
ax9 = plt.subplot(3, 3, 9)
ax9.plot(sol_discontinuous.t, s_discontinuous, 'b-', linewidth=2, label='Разрывный')
ax9.plot(sol_continuous.t, s_continuous, 'g-', linewidth=2, label='Непрерывный')
ax9.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax9.set_xlabel('$t$', fontsize=12)
ax9.set_ylabel('$s(t)$', fontsize=12)
ax9.set_title('Поверхность скольжения (детальный вид)', fontsize=12, fontweight='bold')
ax9.set_ylim([-0.5, 0.5])
ax9.grid(True, alpha=0.3)
ax9.legend()

plt.tight_layout()
plt.savefig('../images/task1_sliding_mode.png', dpi=300, bbox_inches='tight')
print("\nГрафик сохранен: images/task1_sliding_mode.png")
plt.close()

print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
print("=" * 70)
print(f"Разрывный регулятор:")
print(f"  Финальная норма: ||x(T)|| = {norm_discontinuous[-1]:.6f}")
print(f"  Финальное значение s: s(T) = {s_discontinuous[-1]:.6f}")
print(f"\nНепрерывный регулятор:")
print(f"  Финальная норма: ||x(T)|| = {norm_continuous[-1]:.6f}")
print(f"  Финальное значение s: s(T) = {s_continuous[-1]:.6f}")
print(f"\nБез управления:")
print(f"  Финальная норма: ||x(T)|| = {norm_uncontrolled[-1]:.6f}")

