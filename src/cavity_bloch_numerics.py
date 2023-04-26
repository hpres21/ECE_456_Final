import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from cavity_bloch_util import cavity_bloch_equations_resonant_time_drive_short_pulse, \
    cavity_bloch_equations_resonant_time_drive_pulsed_measurement


def rotate_IQ(a):
    # want last element of Q to be fully imaginary
    last = a[-1]
    return a * 1j * np.conjugate(last) / np.abs(last)


def figure2bc(*args):
    # for this experiment

    omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi = args

    epsilon_m = np.sqrt(kappa / 2)
    omega_m = omega_r - chi

    omega_s = omega_a + 3 * chi
    delta_as = omega_a - omega_s
    delta_rm = omega_r - omega_m

    Omega = 0.05 * 2 * np.pi

    args = delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi

    # initial variables

    # steady state
    a0 = -epsilon_m / (delta_rm - chi - 1j * kappa / 2)
    sigmaz0 = -1
    sigmax0 = 0
    sigmay0 = 0
    a_sigmaz0 = a0 * sigmaz0
    a_sigmax0 = a0 * sigmax0
    a_sigmay0 = a0 * sigmay0
    adagger_a0 = -2 * epsilon_m / kappa * np.imag(a0)

    print(f"a0 ss = {a0}")
    print(f"adagger_a0 ss = {adagger_a0}")

    y0 = np.array([a0, sigmaz0, sigmay0, sigmax0, a_sigmaz0, a_sigmay0, a_sigmax0, adagger_a0])

    num_points = 2000
    tspan = (0, 2000)
    times = np.linspace(*tspan, num_points)

    result = solve_ivp(cavity_bloch_equations_resonant_time_drive_short_pulse, tspan, y0, t_eval=times, args=args,
                       max_step=1)

    y = result.y

    a = y[0, :]
    sigmaz = np.real(y[1, :])
    sigmax = np.real(y[2, :])
    sigmay = np.real(y[3, :])
    a_sigmaz = y[4, :]
    a_sigmax = y[5, :]
    a_sigmay = y[6, :]
    adagger_a = np.real(y[7, :])

    # plt.plot(result.t, sigmaz, label="sigma_z")
    # plt.plot(result.t, adagger_a, label="adag a")
    plt.plot(result.t, np.real(a), label="I")
    plt.plot(result.t, np.imag(a), label="Q")

    # plt.plot(result.t, sigmax, label="sigmax")
    # plt.plot(result.t, sigmay, label="sigmay")
    # plt.plot(result.t, np.imag(a_sigmax), label="imag(a_sigmax)")
    # plt.plot(result.t, np.imag(a_sigmay), label="imag(a_sigmay)")
    # plt.plot(result.t, np.imag(a_sigmaz), label="imag(a_sigmaz)")

    plt.xlabel("t (ns)")
    plt.ylabel("Populations")

    plt.title("Pi Pulse with decay")

    plt.legend()
    plt.show()

    plt.show()


def figure2cd(*args):
    omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi = args

    epsilon_m = np.sqrt(kappa / 2)

    omega_s = omega_a + 3 * chi
    delta_as = omega_a - omega_s

    Omega = 0.05 * 2 * np.pi

    num_points = 2501
    tspan = (-500, 2000)
    times = np.linspace(*tspan, num_points)

    delta_rms = np.linspace(5 * chi, -4 * chi, 100)

    sweep_data = np.zeros((len(delta_rms), num_points), dtype=complex)

    for i in range(len(delta_rms)):
        delta_rm = delta_rms[i]

        args = delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi

        # initial variables

        # steady state
        a0 = -epsilon_m / (delta_rm - chi - 1j * kappa / 2)
        sigmaz0 = -1
        sigmax0 = 0
        sigmay0 = 0
        a_sigmaz0 = a0 * sigmaz0
        a_sigmax0 = a0 * sigmax0
        a_sigmay0 = a0 * sigmay0
        adagger_a0 = -2 * epsilon_m / kappa * np.imag(a0)

        y0 = np.array([a0, sigmaz0, sigmay0, sigmax0, a_sigmaz0, a_sigmay0, a_sigmax0, adagger_a0])

        result = solve_ivp(cavity_bloch_equations_resonant_time_drive_short_pulse, tspan, y0, t_eval=times, args=args,
                           max_step=1)

        y = result.y

        a = y[0, :]
        sigmaz = np.real(y[1, :])
        sigmax = np.real(y[2, :])
        sigmay = np.real(y[3, :])
        a_sigmaz = y[4, :]
        a_sigmax = y[5, :]
        a_sigmay = y[6, :]
        adagger_a = np.real(y[7, :])

        a = rotate_IQ(a)

        sweep_data[i, :] = a

        # # plt.plot(result.t, sigmaz, label="sigma_z")
        # # plt.plot(result.t, adagger_a, label="adag a")
        # plt.plot(result.t, np.abs(np.real(a)), label="I")
        # plt.plot(result.t, np.abs(np.imag(a)), label="Q")
        #
        # plt.xlabel("t (ns)")
        # plt.ylabel("Populations")
        #
        # plt.title("Pi Pulse with decay")
        #
        # plt.legend()
        # plt.show()
        #
        # plt.show()

    t = np.copy(result.t) / 1000
    t_step = t[1] - t[0]
    delta_rm_step = (delta_rms[-1] - delta_rms[0]) / (len(delta_rms) - 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(np.abs(np.imag(sweep_data)), cmap='Reds', vmin=-1, vmax=16, aspect='auto', origin='lower',
                   extent=(t[0] - t_step / 2, t[-1] + t_step / 2, (delta_rms[0] - delta_rm_step / 2) / chi,
                           (delta_rms[-1] + delta_rm_step / 2) / chi), )
    # plt.colorbar()
    axes[0].set_xlabel("Times $[\mu s]$")
    axes[0].set_xticks(np.arange(0, 1.6, 0.5))
    axes[0].set_ylabel("Detuning $\Delta_{rm}/\chi$")
    axes[0].set_title("Q")

    axes[1].imshow(np.abs(np.real(sweep_data)), cmap='Reds', vmin=0, vmax=14, aspect='auto', origin='lower',
                   extent=(t[0] - t_step / 2, t[-1] + t_step / 2, (delta_rms[0] - delta_rm_step / 2) / chi,
                           (delta_rms[-1] + delta_rm_step / 2) / chi))
    # plt.colorbar()
    axes[1].set_xlabel("Times $[\mu s]$")
    axes[1].set_xticks(np.arange(0, 1.6, 0.5))
    axes[1].set_title("I")
    axes[1].set_yticklabels([])
    axes[1].tick_params(axis='y', direction="inout", length=10)

    plt.title("Cavity response to a weak measurement")

    fig.subplots_adjust(wspace=0)
    plt.show()


def figure3(*args):
    # for this experiment

    omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi = args

    epsilon_m = np.sqrt(kappa / 2)
    omega_m = omega_r - chi

    omega_s = omega_a + 3 * chi
    delta_as = omega_a - omega_s

    Omega = 0.05 * 2 * np.pi

    tspan = (-500, 2000)

    # evaluate at these times
    t_eval = [-100, 180, 740]

    delta_rms = np.linspace(5 * chi, -4 * chi, 100)
    # delta_rms = [-chi, 0, chi]

    I_data = np.zeros((4, len(delta_rms)))
    Q_data = np.zeros((4, len(delta_rms)))

    for i in range(len(delta_rms)):
        delta_rm = delta_rms[i]

        args = delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi

        # initial variables

        # steady state
        a0 = -epsilon_m / (delta_rm - chi - 1j * kappa / 2)
        sigmaz0 = -1
        sigmax0 = 0
        sigmay0 = 0
        a_sigmaz0 = a0 * sigmaz0
        a_sigmax0 = a0 * sigmax0
        a_sigmay0 = a0 * sigmay0
        adagger_a0 = -2 * epsilon_m / kappa * np.imag(a0)

        y0 = np.array([a0, sigmaz0, sigmay0, sigmax0, a_sigmaz0, a_sigmay0, a_sigmax0, adagger_a0])

        result = solve_ivp(cavity_bloch_equations_resonant_time_drive_short_pulse, tspan, y0, t_eval=t_eval, args=args,
                           max_step=1)

        y = result.y

        a = y[0, :]
        sigmaz = np.real(y[1, :])
        sigmax = np.real(y[2, :])
        sigmay = np.real(y[3, :])
        a_sigmaz = y[4, :]
        a_sigmax = y[5, :]
        a_sigmay = y[6, :]
        adagger_a = np.real(y[7, :])

        I_data[:-1, i] = np.real(a)
        Q_data[:-1, i] = np.imag(a)

        # add |e> steady state values

        I_data[-1, i] = -epsilon_m * (delta_rm + chi) / (np.power(delta_rm + chi, 2) + np.power(kappa / 2, 2))
        Q_data[-1, i] = -epsilon_m * (kappa / 2) / (np.power(delta_rm + chi, 2) + np.power(kappa / 2, 2))

    fig, axes = plt.subplots(2, 1, figsize=(8, 5))

    labels = ["0 ns", "180 ns", "740 ns", "excited state infinite lifetime"]
    colors = ["red", "blue", "lime", "black"]

    for i in range(I_data.shape[0]):
        axes[0].plot(-delta_rms, -Q_data[i, :], color=colors[i], label=labels[i])
        axes[1].plot(-delta_rms, I_data[i, :], color=colors[i], label=labels[i])

    axes[0].set_xlabel("Detuning $-\Delta_{rm}/\chi$")
    axes[0].set_ylabel("Q")
    axes[0].legend()

    axes[1].set_xlabel("Detuning $-\Delta_{rm}/\chi$")
    axes[1].set_ylabel("I")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def figure4(*args):
    # for this experiment

    omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi = args

    epsilon_m = np.sqrt(kappa / 2)
    omega_m = omega_r - chi

    omega_s = omega_a + chi
    delta_as = omega_a - omega_s
    delta_rm = omega_r - omega_m

    Omega = 0.05 * 2 * np.pi

    # epsilon_m = 0
    # gamma_1 = 0
    # gamma_phi = 0

    Omega = 0

    args = delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi

    # initial variables

    # alpha = 1j
    alpha = 2j

    # single photon fock state
    # a0 = (-1j * (delta_rm + epsilon_m) * (1j * delta_rm + gamma_1 + kappa / 2) + chi * epsilon_m) / (
    #             kappa / 2 * (1j * delta_rm + gamma_1 + kappa / 2) - 1j * chi * (1j * chi + gamma_1))
    a0 = 1j
    sigmaz0 = -1
    sigmax0 = 0
    sigmay0 = 0
    a_sigmaz0 = a0 * sigmaz0
    a_sigmax0 = a0 * sigmax0
    a_sigmay0 = a0 * sigmay0
    # adagger_a0 = -2 * epsilon_m / kappa * np.imag(a0)
    adagger_a0 = 0

    y0 = np.array([a0, sigmaz0, sigmay0, sigmax0, a_sigmaz0, a_sigmay0, a_sigmax0, adagger_a0])

    num_points = 2000
    tspan = (0, 1500)
    times = np.linspace(*tspan, num_points)

    result = solve_ivp(cavity_bloch_equations_resonant_time_drive_pulsed_measurement, tspan, y0, t_eval=times,
                       args=args, max_step=1)

    # result = solve_ivp(cavity_bloch_equations_short_pulse, tspan, y0, t_eval=times, args=args, max_step=1)

    y = result.y

    print(y.shape)

    a = y[0, :]
    sigmaz = np.real(y[1, :])
    sigmax = np.real(y[2, :])
    sigmay = np.real(y[3, :])
    a_sigmaz = y[4, :]
    a_sigmax = y[5, :]
    a_sigmay = y[6, :]
    adagger_a = np.real(y[7, :])

    plt.plot(result.t, sigmaz, label="sigma_z")
    # plt.plot(result.t, adagger_a, label="adag a")
    plt.plot(result.t, np.real(a), label="I")
    plt.plot(result.t, np.imag(a), label="Q")

    plt.xlabel("t (ns)")
    plt.ylabel("Populations")

    plt.title("Strong Measurement")

    plt.legend()
    plt.show()


def cavity_bloch_numerical():
    # all in GHz

    omega_r = 6.44252 * 2 * np.pi
    kappa = 0.00169 * 2 * np.pi

    omega_a = 4.009 * 2 * np.pi

    g = 0.134 * 2 * np.pi

    chi = -0.00069 * 2 * np.pi

    gamma_1 = 0.00019 * 2 * np.pi

    # gamma_phi not given
    gamma_phi = 2 * gamma_1

    # figure2cd(omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi)
    figure3(omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi)
    # figure4(omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi)


if __name__ == "__main__":
    cavity_bloch_numerical()
