#!/usr/bin/env python3
"""
Задание 3: Стабилизация маятника при θ=0
Уравнение движения: mlθ̈ + mg sin θ + klθ̇ = (T/l) + mh(t)cosθ
Параметры: 0.8 ≤ l ≤ 1, 0.5 ≤ m ≤ 1, 0.1 ≤ k ≤ 0.2, |h(t)| ≤ 0.5, g = 9.81
Требуется: непрерывный регулятор на основе скользящего режима
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp
from sympy import symbols, diff, simplify

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Символические переменные
theta, theta_dot, T, h, l, m, k, g = symbols('theta theta_dot T h l m k g', real=True)

# Уравнение движения: mlθ̈ + mg sin θ + klθ̇ = (T/l) + mh(t)cosθ
# Приводим к виду: θ̈ = -g/l sin θ - kθ̇ + T/(ml²) + h(t)cosθ/l

# Вводя переменные состояния: x₁ = θ, x₂ = θ̇
x1, x2 = symbols('x1 x2', real=True)

# Система в нормальной форме:
# ẋ₁ = x₂
# ẋ₂ = -g/l sin x₁ - kx₂ + T/(ml²) + h(t)cos x₁/l

# Управление: u = T (управляющий момент)
f1 = x2
f2 = -g/l * sp.sin(x1) - k * x2 + h * sp.cos(x1) / l
g_control = 1 / (m * l**2)  # коэффициент при управлении

print("=" * 70)
print("ЗАДАНИЕ 3: СТАБИЛИЗАЦИЯ МАЯТНИКА")
print("=" * 70)
print(f"\nУравнение движения: mlθ̈ + mg sin θ + klθ̇ = (T/l) + mh(t)cosθ")
print(f"\nСистема в нормальной форме:")
print(f"ẋ₁ = x₂")
print(f"ẋ₂ = -g/l sin x₁ - kx₂ + T/(ml²) + h(t)cos x₁/l")
print(f"\nПараметры:")
print(f"  0.8 ≤ l ≤ 1")
print(f"  0.5 ≤ m ≤ 1")
print(f"  0.1 ≤ k ≤ 0.2")
print(f"  |h(t)| ≤ 0.5")
print(f"  g = 9.81")

# Выбираем поверхность скольжения: s = c₁x₁ + x₂ = 0
# Для стабилизации в θ = 0 (x₁ = 0, x₂ = 0)
c1 = symbols('c1', real=True, positive=True)
s = c1 * x1 + x2

print(f"\nПоверхность скольжения: s = {s} = 0")

# Вычисляем производную поверхности скольжения
s_dot = diff(s, x1) * f1 + diff(s, x2) * f2 + diff(s, x2) * g_control * T
s_dot = simplify(s_dot)
print(f"\nṡ = {s_dot}")

# Для обеспечения ṡ = -k_sliding·s (k_sliding > 0) нужно:
# c₁x₂ + (-g/l sin x₁ - kx₂ + h cos x₁/l) + T/(ml²) = -k_sliding·s
# Отсюда: T/(ml²) = -k_sliding·s - c₁x₂ + g/l sin x₁ + kx₂ - h cos x₁/l
# T = ml²[-k_sliding·s - c₁x₂ + g/l sin x₁ + kx₂ - h cos x₁/l]

# Эквивалентное управление (номинальное, при средних значениях параметров)
# Учитывая неопределенности, используем робастный закон
k_sliding = symbols('k_sliding', real=True, positive=True)

# Номинальное эквивалентное управление (при h = 0, номинальные l, m, k)
l_nom, m_nom, k_nom = symbols('l_nom m_nom k_nom', real=True, positive=True)
u_eq_nom = m_nom * l_nom**2 * (c1 * x2 - 9.81/l_nom * sp.sin(x1) + k_nom * x2)
print(f"\nЭквивалентное управление (номинальное): T_eq ≈ {simplify(u_eq_nom)}")

# Разрывная часть для компенсации неопределенностей
M = symbols('M', real=True, positive=True)
phi = symbols('phi', real=True, positive=True)  # ширина граничного слоя

print("\n" + "=" * 70)
print("НЕПРЕРЫВНЫЙ РЕГУЛЯТОР: T = T_eq - M·tanh(s/φ)")
print("=" * 70)

# Численное моделирование
def system_pendulum(t, x, c1_val, M_val, phi_val, k_sliding_val, l_val, m_val, k_val, h_func):
    """Система маятника с непрерывным регулятором"""
    theta_val, theta_dot_val = x
    
    # Поверхность скольжения
    s_val = c1_val * theta_val + theta_dot_val
    
    # Эквивалентное управление (номинальное)
    u_eq_val = m_val * l_val**2 * (c1_val * theta_dot_val - 9.81/l_val * np.sin(theta_val) + k_val * theta_dot_val)
    
    # Разрывная часть с насыщением
    u_sw_val = -M_val * np.tanh(s_val / phi_val)
    
    # Полное управление
    T_val = u_eq_val + u_sw_val
    
    # Возмущение h(t)
    h_t = h_func(t)
    
    # Динамика системы
    dtheta = theta_dot_val
    dtheta_dot = -9.81/l_val * np.sin(theta_val) - k_val * theta_dot_val + T_val/(m_val * l_val**2) + h_t * np.cos(theta_val) / l_val
    
    return [dtheta, dtheta_dot]

def system_pendulum_uncontrolled(t, x, l_val, m_val, k_val, h_func):
    """Неуправляемый маятник"""
    theta_val, theta_dot_val = x
    h_t = h_func(t)
    
    dtheta = theta_dot_val
    dtheta_dot = -9.81/l_val * np.sin(theta_val) - k_val * theta_dot_val + h_t * np.cos(theta_val) / l_val
    
    return [dtheta, dtheta_dot]

# Параметры регулятора
c1_val = 3.0
M_val = 15.0
phi_val = 0.2
k_sliding_val = 5.0

# Тестовые случаи с разными параметрами
test_cases = [
    {
        "name": "Номинальные (l=0.9, m=0.75, k=0.15)",
        "l": 0.9, "m": 0.75, "k": 0.15,
        "h_func": lambda t: 0.3 * np.sin(0.5 * t),  # гармоническое возмущение
        "color": "b"
    },
    {
        "name": "Максимальные (l=1.0, m=1.0, k=0.2)",
        "l": 1.0, "m": 1.0, "k": 0.2,
        "h_func": lambda t: 0.5 * np.sin(0.5 * t),
        "color": "g"
    },
    {
        "name": "Минимальные (l=0.8, m=0.5, k=0.1)",
        "l": 0.8, "m": 0.5, "k": 0.1,
        "h_func": lambda t: -0.4 * np.cos(0.3 * t),
        "color": "r"
    },
    {
        "name": "Смешанные (l=0.85, m=1.0, k=0.1)",
        "l": 0.85, "m": 1.0, "k": 0.1,
        "h_func": lambda t: 0.2 * np.sin(0.7 * t) + 0.2 * np.cos(0.4 * t),
        "color": "m"
    },
]

# Начальные условия (маятник отклонен от вертикали)
x0 = [0.8, -0.5]  # θ₀ = 0.8 рад ≈ 45°, θ̇₀ = -0.5 рад/с
t_span = (0, 12)
t_eval = np.linspace(0, 12, 2000)

print("\n" + "=" * 70)
print("МОДЕЛИРОВАНИЕ")
print("=" * 70)
print(f"Параметры регулятора: c₁ = {c1_val}, M = {M_val}, φ = {phi_val}, k = {k_sliding_val}")
print(f"Начальные условия: θ₀ = {x0[0]:.2f} рад, θ̇₀ = {x0[1]:.2f} рад/с")

# Построение графиков
fig = plt.figure(figsize=(16, 14))

# Фазовые портреты
ax1 = plt.subplot(4, 3, 1)
for case in test_cases:
    sol = solve_ivp(
        lambda t, x: system_pendulum(t, x, c1_val, M_val, phi_val, k_sliding_val,
                                     case["l"], case["m"], case["k"], case["h_func"]),
        t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
    )
    ax1.plot(sol.y[0], sol.y[1], color=case["color"], linewidth=2, label=case["name"])

ax1.plot(0, 0, 'ko', markersize=8, label='Цель (0,0)')
ax1.set_xlabel('$\\theta$ [рад]', fontsize=12)
ax1.set_ylabel('$\\dot{\\theta}$ [рад/с]', fontsize=12)
ax1.set_title('Фазовые портреты (разные параметры)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)
ax1.axis('equal')

# Временные зависимости θ
ax2 = plt.subplot(4, 3, 2)
for case in test_cases:
    sol = solve_ivp(
        lambda t, x: system_pendulum(t, x, c1_val, M_val, phi_val, k_sliding_val,
                                     case["l"], case["m"], case["k"], case["h_func"]),
        t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
    )
    ax2.plot(sol.t, sol.y[0], color=case["color"], linewidth=2, label=case["name"])

ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax2.set_xlabel('$t$ [с]', fontsize=12)
ax2.set_ylabel('$\\theta(t)$ [рад]', fontsize=12)
ax2.set_title('Угол отклонения маятника', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8)

# Временные зависимости θ̇
ax3 = plt.subplot(4, 3, 3)
for case in test_cases:
    sol = solve_ivp(
        lambda t, x: system_pendulum(t, x, c1_val, M_val, phi_val, k_sliding_val,
                                     case["l"], case["m"], case["k"], case["h_func"]),
        t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
    )
    ax3.plot(sol.t, sol.y[1], color=case["color"], linewidth=2, label=case["name"])

ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax3.set_xlabel('$t$ [с]', fontsize=12)
ax3.set_ylabel('$\\dot{\\theta}(t)$ [рад/с]', fontsize=12)
ax3.set_title('Угловая скорость маятника', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=8)

# Поверхности скольжения
ax4 = plt.subplot(4, 3, 4)
for case in test_cases:
    sol = solve_ivp(
        lambda t, x: system_pendulum(t, x, c1_val, M_val, phi_val, k_sliding_val,
                                     case["l"], case["m"], case["k"], case["h_func"]),
        t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
    )
    s_vals = c1_val * sol.y[0] + sol.y[1]
    ax4.plot(sol.t, s_vals, color=case["color"], linewidth=2, label=case["name"])

ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax4.set_xlabel('$t$ [с]', fontsize=12)
ax4.set_ylabel('$s(t)$', fontsize=12)
ax4.set_title('Поверхность скольжения', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=8)

# Управление для номинального случая
ax5 = plt.subplot(4, 3, 5)
case_nominal = test_cases[0]
sol_nominal = solve_ivp(
    lambda t, x: system_pendulum(t, x, c1_val, M_val, phi_val, k_sliding_val,
                                 case_nominal["l"], case_nominal["m"], case_nominal["k"], case_nominal["h_func"]),
    t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
)
T_vals = []
for i in range(len(sol_nominal.t)):
    theta_n, theta_dot_n = sol_nominal.y[0, i], sol_nominal.y[1, i]
    s_n = c1_val * theta_n + theta_dot_n
    u_eq_n = case_nominal["m"] * case_nominal["l"]**2 * (
        c1_val * theta_dot_n - 9.81/case_nominal["l"] * np.sin(theta_n) + case_nominal["k"] * theta_dot_n
    )
    u_sw_n = -M_val * np.tanh(s_n / phi_val)
    T_vals.append(u_eq_n + u_sw_n)

ax5.plot(sol_nominal.t, T_vals, 'b-', linewidth=1.5)
ax5.set_xlabel('$t$ [с]', fontsize=12)
ax5.set_ylabel('$T(t)$ [Н·м]', fontsize=12)
ax5.set_title('Управляющий момент (номинальный случай)', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Норма состояния
ax6 = plt.subplot(4, 3, 6)
for case in test_cases:
    sol = solve_ivp(
        lambda t, x: system_pendulum(t, x, c1_val, M_val, phi_val, k_sliding_val,
                                     case["l"], case["m"], case["k"], case["h_func"]),
        t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
    )
    norm_vals = np.sqrt(sol.y[0]**2 + sol.y[1]**2)
    ax6.plot(sol.t, norm_vals, color=case["color"], linewidth=2, label=case["name"])

ax6.set_xlabel('$t$ [с]', fontsize=12)
ax6.set_ylabel('$||x(t)||$', fontsize=12)
ax6.set_title('Норма вектора состояния', fontsize=12, fontweight='bold')
ax6.set_yscale('log')
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=8)

# Сравнение с неуправляемой системой
ax7 = plt.subplot(4, 3, 7)
sol_uncontrolled = solve_ivp(
    lambda t, x: system_pendulum_uncontrolled(t, x, case_nominal["l"], case_nominal["m"], 
                                              case_nominal["k"], case_nominal["h_func"]),
    t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
)
ax7.plot(sol_uncontrolled.y[0], sol_uncontrolled.y[1], 'r--', linewidth=2, label='Без управления')
ax7.plot(sol_nominal.y[0], sol_nominal.y[1], 'b-', linewidth=2, label='С управлением')
ax7.plot(0, 0, 'ko', markersize=8, label='Цель (0,0)')
ax7.set_xlabel('$\\theta$ [рад]', fontsize=12)
ax7.set_ylabel('$\\dot{\\theta}$ [рад/с]', fontsize=12)
ax7.set_title('Сравнение с неуправляемой системой', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.legend()
ax7.axis('equal')

# Детальный вид поверхности скольжения
ax8 = plt.subplot(4, 3, 8)
for case in test_cases:
    sol = solve_ivp(
        lambda t, x: system_pendulum(t, x, c1_val, M_val, phi_val, k_sliding_val,
                                     case["l"], case["m"], case["k"], case["h_func"]),
        t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
    )
    s_vals = c1_val * sol.y[0] + sol.y[1]
    ax8.plot(sol.t, s_vals, color=case["color"], linewidth=2, label=case["name"])

ax8.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax8.axhline(y=phi_val, color='gray', linestyle=':', alpha=0.5, label=f'Граничный слой ±φ')
ax8.axhline(y=-phi_val, color='gray', linestyle=':', alpha=0.5)
ax8.set_xlabel('$t$ [с]', fontsize=12)
ax8.set_ylabel('$s(t)$', fontsize=12)
ax8.set_title('Поверхность скольжения (детальный вид)', fontsize=12, fontweight='bold')
ax8.set_ylim([-0.5, 0.5])
ax8.grid(True, alpha=0.3)
ax8.legend(fontsize=8)

# Возмущение h(t)
ax9 = plt.subplot(4, 3, 9)
h_vals = [case_nominal["h_func"](t) for t in sol_nominal.t]
ax9.plot(sol_nominal.t, h_vals, 'm-', linewidth=1.5)
ax9.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Границы |h| ≤ 0.5')
ax9.axhline(y=-0.5, color='r', linestyle='--', alpha=0.5)
ax9.set_xlabel('$t$ [с]', fontsize=12)
ax9.set_ylabel('$h(t)$ [м/с²]', fontsize=12)
ax9.set_title('Горизонтальное возмущение', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3)
ax9.legend()

# Угол в градусах
ax10 = plt.subplot(4, 3, 10)
for case in test_cases:
    sol = solve_ivp(
        lambda t, x: system_pendulum(t, x, c1_val, M_val, phi_val, k_sliding_val,
                                     case["l"], case["m"], case["k"], case["h_func"]),
        t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
    )
    theta_deg = np.degrees(sol.y[0])
    ax10.plot(sol.t, theta_deg, color=case["color"], linewidth=2, label=case["name"])

ax10.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax10.set_xlabel('$t$ [с]', fontsize=12)
ax10.set_ylabel('$\\theta(t)$ [град]', fontsize=12)
ax10.set_title('Угол отклонения (в градусах)', fontsize=12, fontweight='bold')
ax10.grid(True, alpha=0.3)
ax10.legend(fontsize=8)

# Энергия системы (кинетическая + потенциальная)
ax11 = plt.subplot(4, 3, 11)
for case in test_cases:
    sol = solve_ivp(
        lambda t, x: system_pendulum(t, x, c1_val, M_val, phi_val, k_sliding_val,
                                     case["l"], case["m"], case["k"], case["h_func"]),
        t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
    )
    # Энергия: E = 0.5*m*l²*θ̇² + m*g*l*(1 - cos θ)
    E_vals = 0.5 * case["m"] * case["l"]**2 * sol.y[1]**2 + case["m"] * 9.81 * case["l"] * (1 - np.cos(sol.y[0]))
    ax11.plot(sol.t, E_vals, color=case["color"], linewidth=2, label=case["name"])

ax11.set_xlabel('$t$ [с]', fontsize=12)
ax11.set_ylabel('$E(t)$ [Дж]', fontsize=12)
ax11.set_title('Полная энергия системы', fontsize=12, fontweight='bold')
ax11.grid(True, alpha=0.3)
ax11.legend(fontsize=8)

# Сравнение углов для всех случаев
ax12 = plt.subplot(4, 3, 12)
for case in test_cases:
    sol = solve_ivp(
        lambda t, x: system_pendulum(t, x, c1_val, M_val, phi_val, k_sliding_val,
                                     case["l"], case["m"], case["k"], case["h_func"]),
        t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
    )
    ax12.plot(sol.t, np.abs(sol.y[0]), color=case["color"], linewidth=2, label=case["name"])

ax12.set_xlabel('$t$ [с]', fontsize=12)
ax12.set_ylabel('$|\\theta(t)|$ [рад]', fontsize=12)
ax12.set_title('Модуль угла отклонения', fontsize=12, fontweight='bold')
ax12.set_yscale('log')
ax12.grid(True, alpha=0.3)
ax12.legend(fontsize=8)

plt.tight_layout()
plt.savefig('../images/task3_pendulum.png', dpi=300, bbox_inches='tight')
print("\nГрафик сохранен: images/task3_pendulum.png")
plt.close()

print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
print("=" * 70)
for case in test_cases:
    sol = solve_ivp(
        lambda t, x: system_pendulum(t, x, c1_val, M_val, phi_val, k_sliding_val,
                                     case["l"], case["m"], case["k"], case["h_func"]),
        t_span, x0, t_eval=t_eval, rtol=1e-6, max_step=0.01
    )
    norm_final = np.sqrt(sol.y[0, -1]**2 + sol.y[1, -1]**2)
    s_final = c1_val * sol.y[0, -1] + sol.y[1, -1]
    theta_final_deg = np.degrees(sol.y[0, -1])
    print(f"\n{case['name']}:")
    print(f"  Финальная норма: ||x(T)|| = {norm_final:.6f}")
    print(f"  Финальный угол: θ(T) = {theta_final_deg:.4f}°")
    print(f"  Финальное значение s: s(T) = {s_final:.6f}")

