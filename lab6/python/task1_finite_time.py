import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def disturbance(t: float) -> float:
    """Внешнее возмущение."""
    return 0.5 * np.sin(2.0 * t)


def system_rhs(t, state, params):
    """
    x1' = x2
    x2' = -x1^3 + sin(x1) + d(t) + u
    z'  = -k3 * sign(s)
    phi_hat' = k4 * sign(s)
    """
    x1, x2, z, phi_hat = state
    lam, k1, k2, k3, k4, u_clip = params

    s = x2 + lam * x1

    # Финитный однородный регулятор (super-twisting с компенсацией phi_hat)
    u = -phi_hat - k1 * np.sqrt(abs(s)) * np.sign(s) - k2 * z
    u = float(np.clip(u, -u_clip, u_clip))

    dx1 = x2
    dx2 = -x1**3 + np.sin(x1) + disturbance(t) + u
    dz = -k3 * np.sign(s)
    dphi = k4 * np.sign(s)

    return [dx1, dx2, dz, dphi]


def simulate():
    lam = 2.0
    k1, k2, k3, k4 = 3.5, 4.0, 5.0, 2.0
    u_clip = 15.0
    params = (lam, k1, k2, k3, k4, u_clip)

    x0 = [1.5, -1.0, 0.0, 0.0]
    t_span = (0.0, 6.0)
    t_eval = np.linspace(t_span[0], t_span[1], 800)

    sol = solve_ivp(
        lambda t, x: system_rhs(t, x, params),
        t_span,
        x0,
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
        max_step=0.02,
    )

    t = sol.t
    x1, x2, z, phi_hat = sol.y
    s = x2 + lam * x1
    # Восстановим u по траектории для графика
    u = -phi_hat - k1 * np.sqrt(np.abs(s)) * np.sign(s) - k2 * z
    u = np.clip(u, -u_clip, u_clip)
    phi_true = -x1**3 + np.sin(x1) + disturbance(t)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    axes[0, 0].plot(t, x1, label=r"$x_1$")
    axes[0, 0].plot(t, x2, label=r"$x_2$")
    axes[0, 0].set_title("Состояния")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(t, s, label="s")
    axes[0, 1].axhline(0, color="k", linestyle="--", alpha=0.6)
    axes[0, 1].set_title("Поверхность скольжения s")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(t, u, label="u")
    axes[1, 0].set_title("Управление")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(t, z, label="z (интеграл знака s)")
    axes[1, 1].set_title("Внутреннее состояние регулятора z")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    axes[2, 0].plot(t, phi_true, label=r"$\varphi(t)$ (истинная)")
    axes[2, 0].plot(t, -phi_hat, label=r"$-\hat{\varphi}$ (компенсация)", linestyle="--")
    axes[2, 0].set_title("Обобщенная неизвестная динамика")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()

    axes[2, 1].plot(t, np.sqrt(x1**2 + x2**2), label=r"$\|x\|$")
    axes[2, 1].set_title("Норма состояния")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()

    plt.tight_layout()
    out_path = "../images/task1_finite_time.png"
    plt.savefig(out_path, dpi=200)
    print(f"График сохранен: {out_path}")
    return sol


if __name__ == "__main__":
    simulate()

