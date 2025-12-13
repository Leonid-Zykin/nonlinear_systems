import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Параметры сектора
k1 = 0.0
k2 = 1.0

# Диапазон сигма
sigma = np.linspace(-2, 2, 1000)

# Границы сектора
y_lower = k1 * sigma
y_upper = k2 * sigma

# Примеры нелинейностей в секторе [0, 1]
# Насыщение
sat = np.clip(sigma, -1, 1)

# Релей (с ограничением)
relay = np.sign(sigma) * np.minimum(np.abs(sigma), 1)

# Кусочно-линейная функция
piecewise = np.where(np.abs(sigma) < 0.5, 0.5 * sigma, 
                     np.where(sigma > 0, 0.5 + 0.3 * (sigma - 0.5), 
                             -0.5 + 0.3 * (sigma + 0.5)))

fig, ax = plt.subplots(figsize=(8, 6))

# Закрашиваем сектор
ax.fill_between(sigma, y_lower, y_upper, alpha=0.2, color='lightblue', label='Сектор $[k_1, k_2]$')

# Границы сектора
ax.plot(sigma, y_lower, 'k--', linewidth=2, label=f'$y = k_1\\sigma = {k1}\\sigma$')
ax.plot(sigma, y_upper, 'k--', linewidth=2, label=f'$y = k_2\\sigma = {k2}\\sigma$')

# Примеры нелинейностей
ax.plot(sigma, sat, 'r-', linewidth=2, label='Насыщение: $\\mathrm{sat}(\\sigma)$')
ax.plot(sigma, relay, 'g-', linewidth=2, label='Реле (ограниченное)')
ax.plot(sigma, piecewise, 'm-', linewidth=2, label='Кусочно-линейная')

ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

ax.set_xlabel('$\\sigma$', fontsize=14)
ax.set_ylabel('$\\varphi(\\sigma)$', fontsize=14)
ax.set_title('Секторные ограничения: $k_1\\sigma^2 \\leq \\varphi(\\sigma)\\sigma \\leq k_2\\sigma^2$', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

plt.tight_layout()
plt.savefig('../images/sector_constraints.png', dpi=300, bbox_inches='tight')
print("График сохранен: ../images/sector_constraints.png")

