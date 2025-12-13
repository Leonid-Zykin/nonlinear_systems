import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Система: G(s) = 1 / (s(s+2))
num = [1.0]
den = [1.0, 2.0, 0.0]
sys = signal.lti(num, den)

# Частоты
omega = np.logspace(-2, 2, 1000)
s = 1j * omega
w, H = signal.freqresp(sys, omega)

# Найквист-диаграмма
G_real = np.real(H)
G_imag = np.imag(H)

# Параметры сектора
k = 1.0
nu = 1.2  # параметр Попова

# Запрещенная область для кругового критерия
# Круг с центром (-1/k, 0) и радиусом 1/k
theta = np.linspace(0, 2*np.pi, 100)
circle_center = -1/k
circle_radius = 1/k
circle_x = circle_center + circle_radius * np.cos(theta)
circle_y = circle_radius * np.sin(theta)

# Модифицированная Найквист-диаграмма для критерия Попова
G_popov = (1 + 1j * omega * nu) * H
G_popov_real = np.real(G_popov)
G_popov_imag = np.imag(G_popov)

# Вертикальная линия для критерия Попова
x_line = -1/k
y_line = np.linspace(-2, 2, 100)

# Создаем фигуру с двумя подграфиками
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# График 1: Круговой критерий
ax1 = axes[0]
ax1.plot(G_real, G_imag, 'b-', linewidth=2, label='Найквист $G(j\\omega)$')
ax1.plot(circle_x, circle_y, 'r--', linewidth=2, label='Запрещённая область')
ax1.fill(circle_x, circle_y, color='red', alpha=0.2)
ax1.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
ax1.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
ax1.axvline(x=x_line, color='g', linestyle=':', linewidth=1.5, label='$x = -1/k$')
ax1.plot(circle_center, 0, 'ro', markersize=8, label='Центр $(-1/k, 0)$')
ax1.set_xlabel('$\\Re\\{G(j\\omega)\\}$', fontsize=12)
ax1.set_ylabel('$\\Im\\{G(j\\omega)\\}$', fontsize=12)
ax1.set_title('Круговой критерий: Найквист-диаграмма и запрещённая область', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', fontsize=9)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim(-2, 1)
ax1.set_ylim(-1.5, 1.5)

# График 2: Критерий Попова
ax2 = axes[1]
ax2.plot(G_popov_real, G_popov_imag, 'b-', linewidth=2, label='$(1+j\\omega\\nu)G(j\\omega)$')
ax2.axvline(x=x_line, color='r', linestyle='--', linewidth=2, label='$x = -1/k$')
ax2.fill_betweenx(y_line, -3, x_line, color='red', alpha=0.2, label='Запрещённая область')
ax2.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
ax2.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
ax2.set_xlabel('$\\Re\\{(1+j\\omega\\nu)G(j\\omega)\\}$', fontsize=12)
ax2.set_ylabel('$\\Im\\{(1+j\\omega\\nu)G(j\\omega)\\}$', fontsize=12)
ax2.set_title(f'Критерий Попова: модифицированная Найквист-диаграмма ($\\nu={nu}$)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best', fontsize=9)
ax2.set_aspect('equal', adjustable='box')
ax2.set_xlim(-2, 1)
ax2.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.savefig('../images/criteria_comparison.png', dpi=300, bbox_inches='tight')
print("График сохранен: ../images/criteria_comparison.png")

# Отдельный график функции Ляпунова
fig2, ax = plt.subplots(figsize=(10, 6))

# Пример: 2D система, функция Ляпунова V(x) = x^T P x + q * интеграл
# Для простоты покажем контурные линии V(x) = const
x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1, x2)

# Пример: P = [[1, 0], [0, 1]], q = 0.5
# Для нелинейности в секторе [0,1] интеграл от 0 до sigma от phi(tau) dtau
# Упрощенно: V(x) = x1^2 + x2^2 + 0.5 * интеграл
# Для демонстрации используем упрощенную форму
P11, P12, P21, P22 = 1.0, 0.0, 0.0, 1.0
V = P11 * X1**2 + (P12 + P21) * X1 * X2 + P22 * X2**2

# Добавляем интегральный член (упрощенно для насыщения)
# Для sat(sigma) интеграл от 0 до sigma: если |sigma| <= 1, то sigma^2/2, иначе |sigma| - 0.5
sigma = X1  # для нашей системы
integral_term = np.where(np.abs(sigma) <= 1, 0.5 * sigma**2, 
                         np.abs(sigma) - 0.5)
V = V + 0.5 * integral_term

contour = ax.contour(X1, X2, V, levels=15, colors='blue', alpha=0.6)
ax.clabel(contour, inline=True, fontsize=8)
ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_title('Функция Ляпунова $V(x) = x^\\top P x + q \\int_0^{c^\\top x} \\varphi(\\tau) d\\tau$', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('../images/lyapunov_function.png', dpi=300, bbox_inches='tight')
print("График сохранен: ../images/lyapunov_function.png")

