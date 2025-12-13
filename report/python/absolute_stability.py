import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import solve_ivp


def phi_sat(s, k=1.0):
    """Сектор [0, k]: насыщение."""
    return np.clip(s, -k, k)


def lurie_rhs(t, x, A, b, c, k):
    sigma = float(np.dot(c, x))
    return A @ x + b * phi_sat(sigma, k=k)


def simulate_time_response():
    # Линейная часть: G(s) = 1 / ((s+1)(s+2)) = 1/(s^2 + 3s + 2)
    # Это дает гурвицеву матрицу A
    A = np.array([[-1.0, 1.0], [0.0, -2.0]])
    b = np.array([0.0, 1.0])
    c = np.array([1.0, 0.0])
    k_sector = 1.0

    x0 = np.array([0.8, -0.6])
    t_span = (0.0, 10.0)
    t_eval = np.linspace(*t_span, 800)

    sol = solve_ivp(
        lurie_rhs,
        t_span,
        x0,
        t_eval=t_eval,
        args=(A, b, c, k_sector),
        rtol=1e-7,
        atol=1e-9,
        max_step=0.05,
    )

    t = sol.t
    x1, x2 = sol.y
    sigma = c[0] * x1 + c[1] * x2
    phi = phi_sat(sigma, k=k_sector)
    return t, x1, x2, sigma, phi


def plot_nyquist_popov():
    # G(s) = 1 / ((s+1)(s+2)) = 1/(s^2 + 3s + 2)
    num = [1.0]
    den = [1.0, 3.0, 2.0]
    sys = signal.lti(num, den)

    w = np.logspace(-2, 2, 500)
    w, mag, phase = signal.bode(sys, w=w)
    _, H = signal.freqresp(sys, w=w)

    # Попов-модификация
    nu = 1.2
    popov_curve = (1 + 1j * w * nu) * H

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(H.real, H.imag, label="Найквист G(jω)")
    ax[0].axvline(x=-1.0, color="r", linestyle="--", label="x = -1/k")
    ax[0].set_title("Круговой критерий (сектор [0,1])")
    ax[0].set_xlabel("Re")
    ax[0].set_ylabel("Im")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(popov_curve.real, popov_curve.imag, label="Попов: (1+jων)G(jω)")
    ax[1].axvline(x=-1.0, color="r", linestyle="--", label="x = -1/k")
    ax[1].set_title(f"Критерий Попова, ν = {nu}")
    ax[1].set_xlabel("Re")
    ax[1].set_ylabel("Im")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    plt.tight_layout()
    out_path = "../images/nyquist_popov.png"
    plt.savefig(out_path, dpi=200)
    print(f"Сохранено: {out_path}")


def plot_time_response():
    t, x1, x2, sigma, phi = simulate_time_response()
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    ax[0, 0].plot(t, x1, label="x1")
    ax[0, 0].plot(t, x2, label="x2")
    ax[0, 0].set_title("Состояния x1, x2")
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].legend()

    ax[0, 1].plot(t, sigma, label="σ = cᵀx")
    ax[0, 1].plot(t, phi, label="φ(σ)")
    ax[0, 1].set_title("Сигнал σ и нелинейность φ(σ)")
    ax[0, 1].grid(True, alpha=0.3)
    ax[0, 1].legend()

    ax[1, 0].plot(x1, x2)
    ax[1, 0].set_title("Фазовый портрет")
    ax[1, 0].set_xlabel("x1")
    ax[1, 0].set_ylabel("x2")
    ax[1, 0].grid(True, alpha=0.3)

    ax[1, 1].plot(t, np.sqrt(x1 ** 2 + x2 ** 2))
    ax[1, 1].set_title("Норма состояния ||x||")
    ax[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "../images/time_response.png"
    plt.savefig(out_path, dpi=200)
    print(f"Сохранено: {out_path}")


def main():
    plot_nyquist_popov()
    plot_time_response()


if __name__ == "__main__":
    main()

