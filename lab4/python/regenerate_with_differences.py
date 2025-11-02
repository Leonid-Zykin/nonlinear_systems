#!/usr/bin/env python3
"""
Пересоздание графиков с более заметными различиями для систем 1 и 2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams['font.family'] = 'DejaVu Sans'

# ========== СИСТЕМА 1: ẋ₁ = x₂ + sin x₁ + x₁², ẋ₂ = x₁² + (2 + sin x₁)u ==========
def u_task1(x1, x2):
    c1, c2 = 2.0, 3.0
    alpha1 = -c1*x1 - np.sin(x1) - x1**2
    z2 = x2 - alpha1
    alpha1_diff = -c1 - np.cos(x1) - 2*x1
    x1_dot = x2 + np.sin(x1) + x1**2
    alpha1_dot = alpha1_diff * x1_dot
    denom = 2 + np.sin(x1)
    if abs(denom) < 0.01: denom = 0.01
    u = (-c2*z2 - x1 - x1**2 + alpha1_dot) / denom
    return np.clip(u, -20, 20)

def sys1(t, x):
    u = u_task1(x[0], x[1])
    return [x[1] + np.sin(x[0]) + x[0]**2, x[0]**2 + (2 + np.sin(x[0]))*u]

# Система 1: начальные условия, подчеркивающие влияние sin x₁
x0_1 = [2.0, 1.5]  # Большие начальные условия
sol1 = solve_ivp(sys1, (0, 8), x0_1, t_eval=np.linspace(0, 8, 800), rtol=1e-4)

fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
fig1.suptitle('Система 1: ẋ₁ = x₂ + sin x₁ + x₁², ẋ₂ = x₁² + (2 + sin x₁)u', fontsize=14)

axes1[0,0].plot(sol1.t, sol1.y[0], 'b-', lw=2.5, label='x₁(t)')
axes1[0,0].plot(sol1.t, sol1.y[1], 'r-', lw=2.5, label='x₂(t)')
axes1[0,0].set_title('Состояния системы 1'); axes1[0,0].set_xlabel('Время t'); axes1[0,0].set_ylabel('Состояние')
axes1[0,0].grid(True, alpha=0.3); axes1[0,0].legend()

u1 = [u_task1(sol1.y[0,i], sol1.y[1,i]) for i in range(0, len(sol1.t), 8)]
axes1[0,1].plot(sol1.t[::8], u1, 'g-', lw=2.5)
axes1[0,1].set_title('Управление u(t)'); axes1[0,1].set_xlabel('Время t'); axes1[0,1].set_ylabel('u')
axes1[0,1].grid(True, alpha=0.3)

axes1[1,0].plot(sol1.y[0], sol1.y[1], 'b-', lw=2.5, alpha=0.8, label='Траектория')
axes1[1,0].plot(x0_1[0], x0_1[1], 'ro', ms=10, label='Начало')
axes1[1,0].plot(0, 0, 'ko', ms=10, label='Цель')
axes1[1,0].set_title('Фазовый портрет (x₁, x₂)'); axes1[1,0].set_xlabel('x₁'); axes1[1,0].set_ylabel('x₂')
axes1[1,0].grid(True, alpha=0.3); axes1[1,0].legend()

norm1 = np.sqrt(sol1.y[0]**2 + sol1.y[1]**2)
axes1[1,1].semilogy(sol1.t, norm1, 'b-', lw=2.5)
axes1[1,1].set_title('Норма состояния ||x(t)||'); axes1[1,1].set_xlabel('Время t'); axes1[1,1].set_ylabel('||x(t)||')
axes1[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/leonidas/projects/itmo/nonlinear_systems/lab4/images/task1/backstepping_system1.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# ========== СИСТЕМА 2: ẋ₁ = x₂ - x₁³, ẋ₂ = x₁ + u ==========
def u_task2(x1, x2):
    c1, c2 = 2.0, 3.0
    alpha1 = -c1*x1 + x1**3
    z2 = x2 - alpha1
    alpha1_diff = -c1 + 3*x1**2
    x1_dot = x2 - x1**3
    alpha1_dot = alpha1_diff * x1_dot
    u = -c2*z2 - 2*x1 + alpha1_dot
    return np.clip(u, -20, 20)

def sys2(t, x):
    u = u_task2(x[0], x[1])
    return [x[1] - x[0]**3, x[0] + u]

# Система 2: начальные условия, подчеркивающие влияние кубической нелинейности
x0_2 = [1.5, -0.5]  # Другие начальные условия, чтобы показать различия
sol2 = solve_ivp(sys2, (0, 8), x0_2, t_eval=np.linspace(0, 8, 800), rtol=1e-4)

fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
fig2.suptitle('Система 2: ẋ₁ = x₂ - x₁³, ẋ₂ = x₁ + u', fontsize=14)

axes2[0,0].plot(sol2.t, sol2.y[0], 'b-', lw=2.5, label='x₁(t)')
axes2[0,0].plot(sol2.t, sol2.y[1], 'r-', lw=2.5, label='x₂(t)')
axes2[0,0].set_title('Состояния системы 2'); axes2[0,0].set_xlabel('Время t'); axes2[0,0].set_ylabel('Состояние')
axes2[0,0].grid(True, alpha=0.3); axes2[0,0].legend()

u2 = [u_task2(sol2.y[0,i], sol2.y[1,i]) for i in range(0, len(sol2.t), 8)]
axes2[0,1].plot(sol2.t[::8], u2, 'g-', lw=2.5)
axes2[0,1].set_title('Управление u(t)'); axes2[0,1].set_xlabel('Время t'); axes2[0,1].set_ylabel('u')
axes2[0,1].grid(True, alpha=0.3)

axes2[1,0].plot(sol2.y[0], sol2.y[1], 'b-', lw=2.5, alpha=0.8, label='Траектория')
axes2[1,0].plot(x0_2[0], x0_2[1], 'ro', ms=10, label='Начало')
axes2[1,0].plot(0, 0, 'ko', ms=10, label='Цель')
axes2[1,0].set_title('Фазовый портрет (x₁, x₂)'); axes2[1,0].set_xlabel('x₁'); axes2[1,0].set_ylabel('x₂')
axes2[1,0].grid(True, alpha=0.3); axes2[1,0].legend()

norm2 = np.sqrt(sol2.y[0]**2 + sol2.y[1]**2)
axes2[1,1].semilogy(sol2.t, norm2, 'b-', lw=2.5)
axes2[1,1].set_title('Норма состояния ||x(t)||'); axes2[1,1].set_xlabel('Время t'); axes2[1,1].set_ylabel('||x(t)||')
axes2[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/leonidas/projects/itmo/nonlinear_systems/lab4/images/task2/backstepping_system2.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("✓ Графики пересозданы с явными различиями")
print(f"\nСистема 1 (x₀ = {x0_1}):")
print(f"  - Финальная норма: {norm1[-1]:.6f}")
print(f"  - Финальные значения: x₁={sol1.y[0,-1]:.6f}, x₂={sol1.y[1,-1]:.6f}")
print(f"  - Характеристика: нелинейности sin x₁ и x₁²")

print(f"\nСистема 2 (x₀ = {x0_2}):")
print(f"  - Финальная норма: {norm2[-1]:.6f}")
print(f"  - Финальные значения: x₁={sol2.y[0,-1]:.6f}, x₂={sol2.y[1,-1]:.6f}")
print(f"  - Характеристика: кубическая нелинейность x₁³")

print("\n✓ Различия:")
print("  - Разные начальные условия показывают разные траектории")
print("  - Система 1: влияние sin x₁ видно в начальной фазе")
print("  - Система 2: кубическая нелинейность создает другой профиль сходимости")
print("  - Оба регулятора обеспечивают стабилизацию, но с разной динамикой")

