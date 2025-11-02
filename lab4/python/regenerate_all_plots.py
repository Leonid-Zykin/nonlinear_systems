#!/usr/bin/env python3
"""
Быстрая проверка всех графиков
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams['font.family'] = 'DejaVu Sans'

# Задача 1
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

sol1 = solve_ivp(sys1, (0,5), [1.5, 1.0], t_eval=np.linspace(0,5,500), rtol=1e-4)

# Задача 2
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

sol2 = solve_ivp(sys2, (0,5), [1.0, 1.0], t_eval=np.linspace(0,5,500), rtol=1e-4)

# Задача 3 - упрощенное управление для быстрой генерации
def u_task3(x1, x2, x3, x4):
    c1, c2, c3, c4 = 2.0, 3.0, 4.0, 5.0
    alpha1 = np.cos(x1) + c1*x1
    z2 = x2 - alpha1
    alpha1_dot = (-np.sin(x1) + c1) * (np.cos(x1) - x2)
    alpha2 = -x1 + alpha1_dot - c2*z2
    z3 = x3 - alpha2
    # Упрощенный расчет
    alpha2_dot_simple = (-1 - np.cos(x1) - c1)*(np.cos(x1) - x2) + (np.sin(x1) - c1 + c2)*(x1 + x3)
    denom = 2 - np.sin(x3)
    if abs(denom) < 0.5: denom = 0.5
    alpha3 = (-c3*z3 - z2 - x1*x3 + alpha2_dot_simple) / denom
    z4 = x4 - alpha3
    # Упрощенный расчет α̇₃
    x3_dot = x1*x3 + denom*x4
    alpha3_dot_simple = (c3 - x1) / denom * x3_dot
    u = (-c4*z4 - z3*denom - x2*x3 + alpha3_dot_simple) / 2.0
    return np.clip(u, -30, 30)

def sys3(t, x):
    u = u_task3(x[0], x[1], x[2], x[3])
    return [np.cos(x[0]) - x[1], x[0] + x[2], x[0]*x[2] + (2 - np.sin(x[2]))*x[3], x[1]*x[2] + 2*u]

sol3 = solve_ivp(sys3, (0,2.5), [0.2, 0.2, 0.2, 0.2], t_eval=np.linspace(0,2.5,250), rtol=1e-3, max_step=0.05)

# Создание графиков
fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
fig1.suptitle('Система 1 с регулятором бэкстеппинга', fontsize=14)
axes1[0,0].plot(sol1.t, sol1.y[0], 'b-', lw=2, label='x₁'); axes1[0,0].plot(sol1.t, sol1.y[1], 'r-', lw=2, label='x₂')
axes1[0,0].set_title('Состояния'); axes1[0,0].set_xlabel('t'); axes1[0,0].set_ylabel('Состояние'); axes1[0,0].grid(True); axes1[0,0].legend()
u1 = [u_task1(sol1.y[0,i], sol1.y[1,i]) for i in range(0, len(sol1.t), 10)]
axes1[0,1].plot(sol1.t[::10], u1, 'g-', lw=2); axes1[0,1].set_title('Управление'); axes1[0,1].set_xlabel('t'); axes1[0,1].grid(True)
axes1[1,0].plot(sol1.y[0], sol1.y[1], 'b-', lw=2); axes1[1,0].plot(0,0,'ko',ms=8); axes1[1,0].set_title('Фазовый портрет'); axes1[1,0].set_xlabel('x₁'); axes1[1,0].set_ylabel('x₂'); axes1[1,0].grid(True)
norm1 = np.sqrt(sol1.y[0]**2 + sol1.y[1]**2)
axes1[1,1].semilogy(sol1.t, norm1, 'b-', lw=2); axes1[1,1].set_title('Норма'); axes1[1,1].set_xlabel('t'); axes1[1,1].grid(True)
plt.tight_layout()
plt.savefig('/home/leonidas/projects/itmo/nonlinear_systems/lab4/images/task1/backstepping_system1.png', dpi=300, bbox_inches='tight')
plt.close()

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
fig2.suptitle('Система 2 с регулятором бэкстеппинга', fontsize=14)
axes2[0,0].plot(sol2.t, sol2.y[0], 'b-', lw=2, label='x₁'); axes2[0,0].plot(sol2.t, sol2.y[1], 'r-', lw=2, label='x₂')
axes2[0,0].set_title('Состояния'); axes2[0,0].set_xlabel('t'); axes2[0,0].set_ylabel('Состояние'); axes2[0,0].grid(True); axes2[0,0].legend()
u2 = [u_task2(sol2.y[0,i], sol2.y[1,i]) for i in range(0, len(sol2.t), 10)]
axes2[0,1].plot(sol2.t[::10], u2, 'g-', lw=2); axes2[0,1].set_title('Управление'); axes2[0,1].set_xlabel('t'); axes2[0,1].grid(True)
axes2[1,0].plot(sol2.y[0], sol2.y[1], 'b-', lw=2); axes2[1,0].plot(0,0,'ko',ms=8); axes2[1,0].set_title('Фазовый портрет'); axes2[1,0].set_xlabel('x₁'); axes2[1,0].set_ylabel('x₂'); axes2[1,0].grid(True)
norm2 = np.sqrt(sol2.y[0]**2 + sol2.y[1]**2)
axes2[1,1].semilogy(sol2.t, norm2, 'b-', lw=2); axes2[1,1].set_title('Норма'); axes2[1,1].set_xlabel('t'); axes2[1,1].grid(True)
plt.tight_layout()
plt.savefig('/home/leonidas/projects/itmo/nonlinear_systems/lab4/images/task2/backstepping_system2.png', dpi=300, bbox_inches='tight')
plt.close()

fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
fig3.suptitle('Система 3 с регулятором бэкстеппинга', fontsize=14)
axes3[0,0].plot(sol3.t, sol3.y[0], 'b-', lw=2, label='x₁'); axes3[0,0].plot(sol3.t, sol3.y[1], 'r-', lw=2, label='x₂')
axes3[0,0].plot(sol3.t, sol3.y[2], 'g-', lw=2, label='x₃'); axes3[0,0].plot(sol3.t, sol3.y[3], 'm-', lw=2, label='x₄')
axes3[0,0].set_title('Состояния'); axes3[0,0].set_xlabel('t'); axes3[0,0].set_ylabel('Состояние'); axes3[0,0].grid(True); axes3[0,0].legend()
u3 = [u_task3(sol3.y[0,i], sol3.y[1,i], sol3.y[2,i], sol3.y[3,i]) for i in range(0, len(sol3.t), 5)]
axes3[0,1].plot(sol3.t[::5], u3, 'g-', lw=2); axes3[0,1].set_title('Управление'); axes3[0,1].set_xlabel('t'); axes3[0,1].grid(True)
axes3[0,2].plot(sol3.y[0], sol3.y[1], 'b-', lw=2); axes3[0,2].plot(0,0,'ko',ms=8); axes3[0,2].set_title('(x₁,x₂)'); axes3[0,2].grid(True)
axes3[1,0].plot(sol3.y[2], sol3.y[3], 'b-', lw=2); axes3[1,0].plot(0,0,'ko',ms=8); axes3[1,0].set_title('(x₃,x₄)'); axes3[1,0].grid(True)
norm3 = np.sqrt(sol3.y[0]**2 + sol3.y[1]**2 + sol3.y[2]**2 + sol3.y[3]**2)
axes3[1,1].semilogy(sol3.t, norm3, 'b-', lw=2); axes3[1,1].set_title('Норма'); axes3[1,1].set_xlabel('t'); axes3[1,1].grid(True)
axes3[1,2].axis('off')
plt.tight_layout()
plt.savefig('/home/leonidas/projects/itmo/nonlinear_systems/lab4/images/task3/backstepping_system3.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Графики пересозданы")
print(f"Задача 1: финальная норма = {np.sqrt(sol1.y[0,-1]**2 + sol1.y[1,-1]**2):.6f}")
print(f"Задача 2: финальная норма = {np.sqrt(sol2.y[0,-1]**2 + sol2.y[1,-1]**2):.6f}")
print(f"Задача 3: финальная норма = {norm3[-1]:.6f}")

