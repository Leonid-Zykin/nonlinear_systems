#!/usr/bin/env python3
"""
Упрощенное моделирование системы 3 - оптимизированная версия
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def u_control(x1, x2, x3, x4):
    """Оптимизированная функция управления"""
    c1, c2, c3, c4 = 2.0, 3.0, 4.0, 5.0
    
    # Шаг 1: α₁
    alpha1 = np.cos(x1) + c1 * x1
    z2 = x2 - alpha1
    
    # α̇₁
    alpha1_diff = -np.sin(x1) + c1
    x1_dot = np.cos(x1) - x2
    alpha1_dot = alpha1_diff * x1_dot
    
    # Шаг 2: α₂
    alpha2 = -x1 + alpha1_dot - c2 * z2
    z3 = x3 - alpha2
    
    # α̇₂ (упрощенная версия)
    x2_dot = x1 + x3
    alpha2_diff_x1 = -1 + alpha1_diff * (-np.sin(x1))
    alpha2_diff_x2 = -alpha1_diff
    alpha2_dot = alpha2_diff_x1 * x1_dot + alpha2_diff_x2 * x2_dot
    
    # Шаг 3: α₃
    denominator = 2.0 - np.sin(x3)
    if abs(denominator) < 1e-10:
        denominator = 1e-10 if denominator >= 0 else -1e-10
    
    alpha3 = (-c3 * z3 - z2 - x1 * x3 + alpha2_dot) / denominator
    z4 = x4 - alpha3
    
    # α̇₃ (упрощенная версия)
    x3_dot = x1 * x3 + (2.0 - np.sin(x3)) * x4
    # Упрощаем вычисление производной α₃
    alpha3_diff_x3 = (c3 - x1 + np.cos(x3) * alpha3) / denominator
    alpha3_dot = alpha3_diff_x3 * x3_dot
    
    # Шаг 4: u
    u = (-c4 * z4 - z3 * (2.0 - np.sin(x3)) - x2 * x3 + alpha3_dot) / 2.0
    
    # Ограничиваем u для стабильности
    u = np.clip(u, -100, 100)
    
    return u

def controlled_system(t, x):
    """Система с управлением"""
    x1, x2, x3, x4 = x
    
    try:
        u = u_control(x1, x2, x3, x4)
    except:
        u = 0.0
    
    dx1 = np.cos(x1) - x2
    dx2 = x1 + x3
    dx3 = x1 * x3 + (2.0 - np.sin(x3)) * x4
    dx4 = x2 * x3 + 2.0 * u
    
    return [dx1, dx2, dx3, dx4]

# Начальные условия
x0 = [0.5, 0.5, 0.5, 0.5]

# Уменьшаем время моделирования для ускорения
t_span = (0, 5)
t_eval = np.linspace(0, 5, 500)  # Меньше точек

# Решение системы
print("Моделирование системы 3...")
sol = solve_ivp(controlled_system, t_span, x0, t_eval=t_eval, 
                method='RK45', rtol=1e-5, atol=1e-7)
print("Моделирование завершено")

# Построение графиков
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Моделирование системы 3 с регулятором бэкстеппинга', fontsize=14)

# График состояний
axes[0,0].plot(sol.t, sol.y[0], 'b-', linewidth=2, label='x₁(t)')
axes[0,0].plot(sol.t, sol.y[1], 'r-', linewidth=2, label='x₂(t)')
axes[0,0].plot(sol.t, sol.y[2], 'g-', linewidth=2, label='x₃(t)')
axes[0,0].plot(sol.t, sol.y[3], 'm-', linewidth=2, label='x₄(t)')
axes[0,0].set_title('Состояния системы')
axes[0,0].set_xlabel('Время t')
axes[0,0].set_ylabel('Состояние')
axes[0,0].grid(True)
axes[0,0].legend()

# График управления (вычисляем только для части точек для ускорения)
u_indices = np.arange(0, len(sol.t), max(1, len(sol.t)//100))
u_values = []
u_times = []
for i in u_indices:
    try:
        u_val = u_control(sol.y[0, i], sol.y[1, i], sol.y[2, i], sol.y[3, i])
        u_values.append(u_val)
        u_times.append(sol.t[i])
    except:
        continue

axes[0,1].plot(u_times, u_values, 'g-', linewidth=2, label='u(t)')
axes[0,1].set_title('Управление')
axes[0,1].set_xlabel('Время t')
axes[0,1].set_ylabel('Управление u')
axes[0,1].grid(True)
axes[0,1].legend()

# Фазовые портреты
axes[0,2].plot(sol.y[0], sol.y[1], 'b-', linewidth=2, label='Траектория')
axes[0,2].plot(x0[0], x0[1], 'ro', markersize=8, label='Начало')
axes[0,2].plot(0, 0, 'ko', markersize=8, label='Цель')
axes[0,2].set_title('Фазовый портрет (x₁, x₂)')
axes[0,2].set_xlabel('x₁')
axes[0,2].set_ylabel('x₂')
axes[0,2].grid(True)
axes[0,2].legend()

axes[1,0].plot(sol.y[2], sol.y[3], 'b-', linewidth=2, label='Траектория')
axes[1,0].plot(x0[2], x0[3], 'ro', markersize=8, label='Начало')
axes[1,0].plot(0, 0, 'ko', markersize=8, label='Цель')
axes[1,0].set_title('Фазовый портрет (x₃, x₄)')
axes[1,0].set_xlabel('x₃')
axes[1,0].set_ylabel('x₄')
axes[1,0].grid(True)
axes[1,0].legend()

# Норма состояния
norm_x = np.sqrt(sol.y[0]**2 + sol.y[1]**2 + sol.y[2]**2 + sol.y[3]**2)
axes[1,1].semilogy(sol.t, norm_x, 'b-', linewidth=2, label='||x(t)||')
axes[1,1].set_title('Норма состояния')
axes[1,1].set_xlabel('Время t')
axes[1,1].set_ylabel('||x(t)||')
axes[1,1].grid(True)
axes[1,1].legend()

axes[1,2].axis('off')

plt.tight_layout()
plt.savefig('/home/leonidas/projects/itmo/nonlinear_systems/lab4/images/task3/backstepping_system3.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print(f"Графики сохранены в images/task3/backstepping_system3.png")
print(f"Финальные значения: x₁={sol.y[0,-1]:.4f}, x₂={sol.y[1,-1]:.4f}, x₃={sol.y[2,-1]:.4f}, x₄={sol.y[3,-1]:.4f}")
print(f"Финальная норма: {norm_x[-1]:.4f}")

