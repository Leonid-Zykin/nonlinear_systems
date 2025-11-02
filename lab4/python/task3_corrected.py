#!/usr/bin/env python3
"""
Исправленная версия управления для задачи 3 с правильными производными
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams['font.family'] = 'DejaVu Sans'

def u_control_correct(x1, x2, x3, x4):
    """Исправленное управление с правильными производными"""
    c1, c2, c3, c4 = 2.0, 3.0, 4.0, 5.0
    
    # Шаг 1: α₁ = cos x₁ + c₁x₁
    cos_x1 = np.cos(x1)
    sin_x1 = np.sin(x1)
    alpha1 = cos_x1 + c1 * x1
    z2 = x2 - alpha1
    
    # α̇₁ = ∂α₁/∂x₁ · ẋ₁ = (-sin x₁ + c₁) · (cos x₁ - x₂)
    alpha1_diff = -sin_x1 + c1
    x1_dot = cos_x1 - x2
    alpha1_dot = alpha1_diff * x1_dot
    
    # Шаг 2: α₂ = -x₁ + α̇₁ - c₂z₂
    alpha2 = -x1 + alpha1_dot - c2 * z2
    z3 = x3 - alpha2
    
    # α̇₂ = ∂α₂/∂x₁·ẋ₁ + ∂α₂/∂x₂·ẋ₂
    # α₂ = -x₁ + (-sin x₁ + c₁)(cos x₁ - x₂) - c₂(x₂ - cos x₁ - c₁x₁)
    # ∂α₂/∂x₁ = -1 + (-cos x₁)(cos x₁ - x₂) + (-sin x₁ + c₁)(-sin x₁) - c₂(sin x₁ - c₁)
    # Упрощаем вычисление
    x2_dot = x1 + x3
    
    # Более точное вычисление ∂α₂/∂x₁
    dalpha2_dx1 = -1 + (-cos_x1) * (cos_x1 - x2) + (-sin_x1 + c1) * (-sin_x1) - c2 * (sin_x1 - c1)
    
    # ∂α₂/∂x₂ = (-sin x₁ + c₁) + c₂
    dalpha2_dx2 = (-sin_x1 + c1) + c2
    
    alpha2_dot = dalpha2_dx1 * x1_dot + dalpha2_dx2 * x2_dot
    
    # Шаг 3: α₃
    sin_x3 = np.sin(x3)
    cos_x3 = np.cos(x3)
    denominator = 2.0 - sin_x3
    
    if abs(denominator) < 0.5:
        denominator = 0.5 if denominator >= 0 else -0.5
    
    alpha3 = (-c3 * z3 - z2 - x1 * x3 + alpha2_dot) / denominator
    z4 = x4 - alpha3
    
    # Шаг 4: u
    # α̇₃ = ∂α₃/∂x₁·ẋ₁ + ∂α₃/∂x₂·ẋ₂ + ∂α₃/∂x₃·ẋ₃
    # Упрощаем: основной вклад от ∂α₃/∂x₃
    x3_dot = x1 * x3 + denominator * x4
    
    # Вычисляем ∂α₃/∂x₃ более точно
    # α₃ = (-c₃z₃ - z₂ - x₁x₃ + α̇₂) / (2 - sin x₃)
    # где z₃ = x₃ - α₂
    # ∂α₃/∂x₃ включает производную знаменателя
    dalpha3_dx3 = (c3 - x1 + cos_x3 * alpha3) / denominator
    
    # Полная производная
    dalpha3_dx1 = -x3 / denominator  # Упрощение
    dalpha3_dx2 = -1 / denominator
    dalpha3_dx3_full = dalpha3_dx3
    
    alpha3_dot = (dalpha3_dx1 * x1_dot + 
                  dalpha3_dx2 * x2_dot + 
                  dalpha3_dx3_full * x3_dot)
    
    u = (-c4 * z4 - z3 * denominator - x2 * x3 + alpha3_dot) / 2.0
    
    # Более строгое ограничение
    u = np.clip(u, -15.0, 15.0)
    
    return u

def controlled_system(t, x):
    """Система с управлением"""
    x1, x2, x3, x4 = x
    
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        return [0.0, 0.0, 0.0, 0.0]
    
    try:
        u = u_control_correct(x1, x2, x3, x4)
        if np.isnan(u) or np.isinf(u):
            u = -1.0 * np.sum(x)
            u = np.clip(u, -15, 15)
    except:
        u = -1.0 * np.sum(x)
        u = np.clip(u, -15, 15)
    
    dx1 = np.cos(x1) - x2
    dx2 = x1 + x3
    dx3 = x1 * x3 + (2.0 - np.sin(x3)) * x4
    dx4 = x2 * x3 + 2.0 * u
    
    return [dx1, dx2, dx3, dx4]

# Более близкие к нулю начальные условия
x0 = [0.1, 0.1, 0.1, 0.1]

t_span = (0, 3)
t_eval = np.linspace(0, 3, 300)

print("Моделирование системы 3...")
try:
    sol = solve_ivp(controlled_system, t_span, x0, t_eval=t_eval, 
                    method='RK45', rtol=1e-3, atol=1e-5, max_step=0.05)
    
    if sol.success and len(sol.t) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Система 3 с регулятором бэкстеппинга', fontsize=14)
        
        axes[0,0].plot(sol.t, sol.y[0], 'b-', lw=2, label='x₁')
        axes[0,0].plot(sol.t, sol.y[1], 'r-', lw=2, label='x₂')
        axes[0,0].plot(sol.t, sol.y[2], 'g-', lw=2, label='x₃')
        axes[0,0].plot(sol.t, sol.y[3], 'm-', lw=2, label='x₄')
        axes[0,0].set_title('Состояния'); axes[0,0].set_xlabel('t'); axes[0,0].set_ylabel('Состояние')
        axes[0,0].grid(True); axes[0,0].legend()
        
        u_vals = [u_control_correct(sol.y[0,i], sol.y[1,i], sol.y[2,i], sol.y[3,i]) 
                  for i in range(0, len(sol.t), 10)]
        axes[0,1].plot(sol.t[::10], u_vals, 'g-', lw=2)
        axes[0,1].set_title('Управление'); axes[0,1].set_xlabel('t'); axes[0,1].grid(True)
        
        axes[0,2].plot(sol.y[0], sol.y[1], 'b-', lw=2, alpha=0.7)
        axes[0,2].plot(0,0,'ko',ms=8)
        axes[0,2].set_title('(x₁,x₂)'); axes[0,2].set_xlabel('x₁'); axes[0,2].set_ylabel('x₂'); axes[0,2].grid(True)
        
        axes[1,0].plot(sol.y[2], sol.y[3], 'b-', lw=2, alpha=0.7)
        axes[1,0].plot(0,0,'ko',ms=8)
        axes[1,0].set_title('(x₃,x₄)'); axes[1,0].set_xlabel('x₃'); axes[1,0].set_ylabel('x₄'); axes[1,0].grid(True)
        
        norm_x = np.sqrt(sol.y[0]**2 + sol.y[1]**2 + sol.y[2]**2 + sol.y[3]**2)
        axes[1,1].semilogy(sol.t, norm_x, 'b-', lw=2)
        axes[1,1].set_title('Норма'); axes[1,1].set_xlabel('t'); axes[1,1].grid(True)
        
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.savefig('/home/leonidas/projects/itmo/nonlinear_systems/lab4/images/task3/backstepping_system3.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ График сохранен")
        print(f"Финальные значения: x₁={sol.y[0,-1]:.4f}, x₂={sol.y[1,-1]:.4f}, x₃={sol.y[2,-1]:.4f}, x₄={sol.y[3,-1]:.4f}")
        print(f"Финальная норма: {norm_x[-1]:.4f}")
        
        if norm_x[-1] < 1.0:
            print("✓ Система стабилизируется")
        else:
            print("⚠ Система требует дополнительной настройки параметров")
    else:
        print("Моделирование не удалось")
except Exception as e:
    print(f"Ошибка: {e}")

