#!/usr/bin/env python3
"""
Финальная версия с оптимальными начальными условиями для обеих систем
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

def sys1_controlled(t, x):
    u = u_task1(x[0], x[1])
    return [x[1] + np.sin(x[0]) + x[0]**2, x[0]**2 + (2 + np.sin(x[0]))*u]

def sys1_uncontrolled(t, x):
    return [x[1] + np.sin(x[0]) + x[0]**2, x[0]**2]

# Начальные условия для системы 1 - подчеркивают влияние sin x₁
x0_1 = [1.5, 0.8]
sol1_ctrl = solve_ivp(sys1_controlled, (0, 6), x0_1, t_eval=np.linspace(0, 6, 600), rtol=1e-4)
sol1_unctrl = solve_ivp(sys1_uncontrolled, (0, 6), x0_1, t_eval=np.linspace(0, 6, 600), rtol=1e-4, max_step=0.05)

fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
fig1.suptitle('Система 1: ẋ₁ = x₂ + sin x₁ + x₁², ẋ₂ = x₁² + (2 + sin x₁)u', fontsize=14, fontweight='bold')

axes1[0,0].plot(sol1_ctrl.t, sol1_ctrl.y[0], 'b-', lw=2.5, label='x₁(t), управляемая')
axes1[0,0].plot(sol1_ctrl.t, sol1_ctrl.y[1], 'r-', lw=2.5, label='x₂(t), управляемая')
axes1[0,0].plot(sol1_unctrl.t[:min(len(sol1_unctrl.t), 400)], 
                sol1_unctrl.y[0][:min(len(sol1_unctrl.t), 400)], 
                'b--', lw=1.5, alpha=0.6, label='x₁(t), неуправляемая')
axes1[0,0].plot(sol1_unctrl.t[:min(len(sol1_unctrl.t), 400)], 
                sol1_unctrl.y[1][:min(len(sol1_unctrl.t), 400)], 
                'r--', lw=1.5, alpha=0.6, label='x₂(t), неуправляемая')
axes1[0,0].set_title('Состояния (сплошные - с управлением, пунктир - без)')
axes1[0,0].set_xlabel('Время t'); axes1[0,0].set_ylabel('Состояние')
axes1[0,0].grid(True, alpha=0.3); axes1[0,0].legend(fontsize=9)

u1 = [u_task1(sol1_ctrl.y[0,i], sol1_ctrl.y[1,i]) for i in range(0, len(sol1_ctrl.t), 6)]
axes1[0,1].plot(sol1_ctrl.t[::6], u1, 'g-', lw=2.5)
axes1[0,1].axhline(y=0, color='k', linestyle='--', lw=1, alpha=0.5)
axes1[0,1].set_title('Управление u(t)'); axes1[0,1].set_xlabel('Время t'); axes1[0,1].set_ylabel('u')
axes1[0,1].grid(True, alpha=0.3)

axes1[1,0].plot(sol1_ctrl.y[0], sol1_ctrl.y[1], 'b-', lw=2.5, alpha=0.8, label='Управляемая')
axes1[1,0].plot(sol1_unctrl.y[0][:min(len(sol1_unctrl.y[0]), 400)], 
                sol1_unctrl.y[1][:min(len(sol1_unctrl.y[1]), 400)], 
                'r--', lw=2, alpha=0.6, label='Неуправляемая')
axes1[1,0].plot(x0_1[0], x0_1[1], 'ro', ms=12, label='Начало', zorder=5)
axes1[1,0].plot(0, 0, 'ko', ms=12, label='Цель', zorder=5)
axes1[1,0].set_title('Фазовый портрет (x₁, x₂)')
axes1[1,0].set_xlabel('x₁'); axes1[1,0].set_ylabel('x₂')
axes1[1,0].grid(True, alpha=0.3); axes1[1,0].legend()

norm1 = np.sqrt(sol1_ctrl.y[0]**2 + sol1_ctrl.y[1]**2)
axes1[1,1].semilogy(sol1_ctrl.t, norm1, 'b-', lw=2.5, label='Управляемая')
norm1_unctrl = np.sqrt(sol1_unctrl.y[0]**2 + sol1_unctrl.y[1]**2)
axes1[1,1].semilogy(sol1_unctrl.t[:min(len(sol1_unctrl.t), len(norm1_unctrl))], 
                    norm1_unctrl[:min(len(sol1_unctrl.t), len(norm1_unctrl))], 
                    'r--', lw=2, alpha=0.6, label='Неуправляемая')
axes1[1,1].set_title('Норма состояния ||x(t)||'); axes1[1,1].set_xlabel('Время t'); axes1[1,1].set_ylabel('||x(t)||')
axes1[1,1].grid(True, alpha=0.3); axes1[1,1].legend()

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

def sys2_controlled(t, x):
    u = u_task2(x[0], x[1])
    return [x[1] - x[0]**3, x[0] + u]

def sys2_uncontrolled(t, x):
    return [x[1] - x[0]**3, x[0]]

# Начальные условия для системы 2 - подчеркивают влияние x₁³
x0_2 = [1.2, -0.8]
sol2_ctrl = solve_ivp(sys2_controlled, (0, 6), x0_2, t_eval=np.linspace(0, 6, 600), rtol=1e-4)
sol2_unctrl = solve_ivp(sys2_uncontrolled, (0, 6), x0_2, t_eval=np.linspace(0, 6, 600), rtol=1e-4, max_step=0.05)

fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
fig2.suptitle('Система 2: ẋ₁ = x₂ - x₁³, ẋ₂ = x₁ + u', fontsize=14, fontweight='bold')

axes2[0,0].plot(sol2_ctrl.t, sol2_ctrl.y[0], 'b-', lw=2.5, label='x₁(t), управляемая')
axes2[0,0].plot(sol2_ctrl.t, sol2_ctrl.y[1], 'r-', lw=2.5, label='x₂(t), управляемая')
axes2[0,0].plot(sol2_unctrl.t[:min(len(sol2_unctrl.t), 400)], 
                sol2_unctrl.y[0][:min(len(sol2_unctrl.t), 400)], 
                'b--', lw=1.5, alpha=0.6, label='x₁(t), неуправляемая')
axes2[0,0].plot(sol2_unctrl.t[:min(len(sol2_unctrl.t), 400)], 
                sol2_unctrl.y[1][:min(len(sol2_unctrl.t), 400)], 
                'r--', lw=1.5, alpha=0.6, label='x₂(t), неуправляемая')
axes2[0,0].set_title('Состояния (сплошные - с управлением, пунктир - без)')
axes2[0,0].set_xlabel('Время t'); axes2[0,0].set_ylabel('Состояние')
axes2[0,0].grid(True, alpha=0.3); axes2[0,0].legend(fontsize=9)

u2 = [u_task2(sol2_ctrl.y[0,i], sol2_ctrl.y[1,i]) for i in range(0, len(sol2_ctrl.t), 6)]
axes2[0,1].plot(sol2_ctrl.t[::6], u2, 'g-', lw=2.5)
axes2[0,1].axhline(y=0, color='k', linestyle='--', lw=1, alpha=0.5)
axes2[0,1].set_title('Управление u(t)'); axes2[0,1].set_xlabel('Время t'); axes2[0,1].set_ylabel('u')
axes2[0,1].grid(True, alpha=0.3)

axes2[1,0].plot(sol2_ctrl.y[0], sol2_ctrl.y[1], 'b-', lw=2.5, alpha=0.8, label='Управляемая')
axes2[1,0].plot(sol2_unctrl.y[0][:min(len(sol2_unctrl.y[0]), 400)], 
                sol2_unctrl.y[1][:min(len(sol2_unctrl.y[1]), 400)], 
                'r--', lw=2, alpha=0.6, label='Неуправляемая')
axes2[1,0].plot(x0_2[0], x0_2[1], 'ro', ms=12, label='Начало', zorder=5)
axes2[1,0].plot(0, 0, 'ko', ms=12, label='Цель', zorder=5)
axes2[1,0].set_title('Фазовый портрет (x₁, x₂)')
axes2[1,0].set_xlabel('x₁'); axes2[1,0].set_ylabel('x₂')
axes2[1,0].grid(True, alpha=0.3); axes2[1,0].legend()

norm2 = np.sqrt(sol2_ctrl.y[0]**2 + sol2_ctrl.y[1]**2)
axes2[1,1].semilogy(sol2_ctrl.t, norm2, 'b-', lw=2.5, label='Управляемая')
norm2_unctrl = np.sqrt(sol2_unctrl.y[0]**2 + sol2_unctrl.y[1]**2)
axes2[1,1].semilogy(sol2_unctrl.t[:min(len(sol2_unctrl.t), len(norm2_unctrl))], 
                    norm2_unctrl[:min(len(sol2_unctrl.t), len(norm2_unctrl))], 
                    'r--', lw=2, alpha=0.6, label='Неуправляемая')
axes2[1,1].set_title('Норма состояния ||x(t)||'); axes2[1,1].set_xlabel('Время t'); axes2[1,1].set_ylabel('||x(t)||')
axes2[1,1].grid(True, alpha=0.3); axes2[1,1].legend()

plt.tight_layout()
plt.savefig('/home/leonidas/projects/itmo/nonlinear_systems/lab4/images/task2/backstepping_system2.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("✓ Графики пересозданы с оптимальными начальными условиями")
print(f"\nСистема 1 (x₀ = {x0_1}):")
print(f"  - Управляемая: финальная норма = {norm1[-1]:.6f}")
print(f"  - Характеристика: влияние sin x₁ и x₁²")

print(f"\nСистема 2 (x₀ = {x0_2}):")
print(f"  - Управляемая: финальная норма = {norm2[-1]:.6f}")
print(f"  - Характеристика: кубическая нелинейность x₁³")

print("\n✓ Различия четко видны:")
print("  - Разные начальные условия: [1.5, 0.8] vs [1.2, -0.8]")
print("  - Разные уравнения систем (sin x₁ vs x₁³)")
print("  - Разные траектории сходимости")
print("  - Сравнение с неуправляемыми системами")

