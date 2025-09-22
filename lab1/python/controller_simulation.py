import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import sympy as sp
from sympy import symbols, diff, Matrix, lambdify
import warnings
warnings.filterwarnings('ignore')

# Настройка для отображения русских символов
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def simulate_controlled_system(y_star: float = -1.5):
	"""Численное моделирование управляемых систем.
	Система 1: стабилизация к точке (1, y_star), где y_star ∈ [-2, 0], y_star ≠ -1.
	Система 2: без изменений (стабилизация к (0,0)).
	"""
	print("Численное моделирование управляемых систем...")
	
	x1, x2, u1, u2 = symbols('x1 x2 u1 u2', real=True)
	
	# Управляемая система 1
	f1_1 = -x1 + 2*x1**3 + x2 + sp.sin(u1)
	f2_1 = -x1 - x2 + 3*sp.sin(u2)
	
	# Управляемая система 2
	f1_2 = x2 + x1*x2 + u1
	f2_2 = -x2 + x2**2 - x1**3 + sp.sin(u1)
	
	# Создаем функции
	f1_1_func = lambdify([x1, x2, u1, u2], f1_1, 'numpy')
	f2_1_func = lambdify([x1, x2, u1, u2], f2_1, 'numpy')
	f1_2_func = lambdify([x1, x2, u1], f1_2, 'numpy')
	f2_2_func = lambdify([x1, x2, u1], f2_2, 'numpy')
	
	# Синтез LQR регуляторов
	def synthesize_lqr(A, B, Q, R):
		"""Синтез LQR регулятора"""
		try:
			P = solve_continuous_are(A, B, Q, R)
			K = np.linalg.inv(R) @ B.T @ P
			return K, P
		except:
			return None, None
	
	# -------------------------
	# Система 1: цель (1, y_star)
	# -------------------------
	# Проверим достижимость по каналу u1: sin u1_ss = -1 - y_star ∈ [-1,1] → y_star ∈ [-2,0]
	if not (-2.0 <= y_star <= 0.0) or abs(y_star + 1.0) < 1e-9:
		raise ValueError("y_star должен быть в [-2,0] и не равен -1")
	x1_star, x2_star = 1.0, float(y_star)
	u1_ss = float(np.arcsin(-(1.0 + x2_star)))
	u2_ss = float(np.arcsin((1.0 + x2_star)/3.0))
	# Матрицы линеаризации
	Jx1 = Matrix([f1_1, f2_1]).jacobian([x1, x2])
	Ju1 = Matrix([f1_1, f2_1]).jacobian([u1, u2])
	A1 = np.array(Jx1.subs({x1: x1_star, x2: x2_star, u1: u1_ss, u2: u2_ss})).astype(float)
	B1 = np.array(Ju1.subs({x1: x1_star, x2: x2_star, u1: u1_ss, u2: u2_ss})).astype(float)
	Q1 = 10 * np.eye(2)
	R1 = np.eye(2)
	K1, P1 = synthesize_lqr(A1, B1, Q1, R1)
	
	if K1 is not None:
		print(f"Система 1: цель (1, {x2_star:.3f}), u_ss=({u1_ss:.3f}, {u2_ss:.3f})")
		print(f"A=\n{A1}\nB=\n{B1}\nK=\n{K1}")
		
		# Моделирование системы 1
		def system1_controlled(t, y):
			x1_val, x2_val = y
			dx = np.array([x1_val - x1_star, x2_val - x2_star])
			v = -K1 @ dx
			u = np.array([u1_ss, u2_ss]) + v
			return [f1_1_func(x1_val, x2_val, u[0], u[1]),
			       f2_1_func(x1_val, x2_val, u[0], u[1])]
		
		# Время и сетка
		t_span = (0, 10)
		t_eval = np.linspace(0, 10, 1000)
		
		fig, axes = plt.subplots(2, 2, figsize=(15, 12))
		
		# Система 1 без управления (как раньше)
		def system1_uncontrolled(t, y):
			x1_val, x2_val = y
			return [f1_1_func(x1_val, x2_val, 0, 0),
			       f2_1_func(x1_val, x2_val, 0, 0)]
		
		initial_conditions = [[0.5, 0.5], [-0.5, 0.5], [1.0, 1.0], [-1.0, -1.0]]
		for i, ic in enumerate(initial_conditions):
			sol_uncontrolled = solve_ivp(system1_uncontrolled, t_span, ic, t_eval=t_eval, rtol=1e-8)
			if sol_uncontrolled.success:
				axes[0, 0].plot(sol_uncontrolled.y[0], sol_uncontrolled.y[1], 
				               label=f'Начальная точка {i+1}', alpha=0.7)
		
		# Система 1 с управлением (начальные точки близко к цели)
		initial_conditions_ctrl = [
			[list(np.array([x1_star, x2_star]) + np.array([ 0.1,  0.1]))],
			[list(np.array([x1_star, x2_star]) + np.array([-0.1,  0.1]))]
		]
		initial_conditions_ctrl = [ic[0] for ic in initial_conditions_ctrl]
		for j, icc in enumerate(initial_conditions_ctrl):
			sol_controlled = solve_ivp(system1_controlled, t_span, icc, t_eval=t_eval, rtol=1e-8)
			if sol_controlled.success:
				axes[0, 1].plot(sol_controlled.y[0], sol_controlled.y[1], 
				               label=f'Начальная точка {j+1}', alpha=0.7)
		
		axes[0, 0].set_xlabel('x₁')
		axes[0, 0].set_ylabel('x₂')
		axes[0, 0].set_title('Система 1 без управления')
		axes[0, 0].grid(True, alpha=0.3)
		axes[0, 0].legend()
		
		axes[0, 1].set_xlabel('x₁')
		axes[0, 1].set_ylabel('x₂')
		axes[0, 1].set_title('Система 1 с LQR управлением')
		axes[0, 1].grid(True, alpha=0.3)
		axes[0, 1].legend()
	
	# -------------------------
	# Система 2: без изменений
	# -------------------------
	A2 = np.array([[0, 1], [0, -1]])
	B2 = np.array([[1], [1]])
	Q2 = 10 * np.eye(2)
	R2 = np.array([[1]])
	K2, P2 = synthesize_lqr(A2, B2, Q2, R2)
	if K2 is not None:
		print(f"LQR регулятор для системы 2: K = {K2}")
		def system2_controlled(t, y):
			x1_val, x2_val = y
			u = -K2 @ np.array([x1_val, x2_val])
			u1_val = u[0]
			return [f1_2_func(x1_val, x2_val, u1_val),
			       f2_2_func(x1_val, x2_val, u1_val)]
		def system2_uncontrolled(t, y):
			x1_val, x2_val = y
			return [f1_2_func(x1_val, x2_val, 0),
			       f2_2_func(x1_val, x2_val, 0)]
		for i, ic in enumerate(initial_conditions):
			sol_uncontrolled = solve_ivp(system2_uncontrolled, t_span, ic, t_eval=t_eval, rtol=1e-8)
			if sol_uncontrolled.success:
				axes[1, 0].plot(sol_uncontrolled.y[0], sol_uncontrolled.y[1], 
				               label=f'Начальная точка {i+1}', alpha=0.7)
			sol_controlled = solve_ivp(system2_controlled, t_span, ic, t_eval=t_eval, rtol=1e-8)
			if sol_controlled.success:
				axes[1, 1].plot(sol_controlled.y[0], sol_controlled.y[1], 
				               label=f'Начальная точка {i+1}', alpha=0.7)
		axes[1, 0].set_xlabel('x₁')
		axes[1, 0].set_ylabel('x₂')
		axes[1, 0].set_title('Система 2 без управления')
		axes[1, 0].grid(True, alpha=0.3)
		axes[1, 0].legend()
		axes[1, 1].set_xlabel('x₁')
		axes[1, 1].set_ylabel('x₂')
		axes[1, 1].set_title('Система 2 с LQR управлением')
		axes[1, 1].grid(True, alpha=0.3)
		axes[1, 1].legend()
	
	plt.suptitle('Численное моделирование управляемых систем', fontsize=16)
	plt.tight_layout()
	filename = '/home/leonidas/projects/studying_at_ITMO/nonlinear_systems/lab1/python/controller_simulation.png'
	plt.savefig(filename, dpi=300, bbox_inches='tight')
	print(f"Результаты моделирования сохранены: {filename}")
	plt.show()


def main():
	"""Основная функция"""
	print("=" * 80)
	print("ЧИСЛЕННОЕ МОДЕЛИРОВАНИЕ УПРАВЛЯЕМЫХ СИСТЕМ")
	print("=" * 80)
	
	simulate_controlled_system(y_star=-1.5)
	
	print("\n" + "=" * 80)
	print("МОДЕЛИРОВАНИЕ ЗАВЕРШЕНО!")
	print("=" * 80)

if __name__ == "__main__":
	main()
