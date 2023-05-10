import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import solve_ivp
import qutip as qt

from cavity_bloch_util import cavity_bloch_equations_resonant_time_drive_short_pulse, \
    cavity_bloch_equations_resonant_time_drive_pulsed_measurement, \
    cavity_bloch_equations_resonant_time_drive_variable_pulse_length, \
    cavity_bloch_equations_resonant_time_drive, rotate_IQ
from qutip_simulations_rotating_frame import get_Hamiltonian


def figure2de_qutip(delta_rms, num_levels, num_points, times, *args):
    omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi = args

    epsilon_m = np.sqrt(kappa / 2) / 10
    omega_m = omega_r - chi

    # pick omega so that pulse lasts 10 ns
    Omega = 0.025 * 2 * np.pi
    # lamb shift
    omega_s = omega_a + chi

    delta_as = omega_a - omega_s

    t1 = 0
    t2 = t1 + 10

    args = {
        "epsilon_m": epsilon_m,
        "omega_m": omega_m,
        "Omega": Omega,
        "omega_s": omega_s,
        "t1": t1,
        "t2": t2
    }

    def H_qubit_drive(t, args):
        Omega = args["Omega"]
        t1 = args["t1"]
        t2 = args["t2"]
        if t1 < t <= t2:
            return Omega
        else:
            return 0

    a = qt.tensor(qt.destroy(num_levels), qt.qeye(2))
    sigma_m = qt.tensor(qt.qeye(num_levels), qt.destroy(2))
    sigma_z = qt.tensor(qt.qeye(num_levels), -qt.sigmaz())

    collapse_operators = [
        np.sqrt(kappa) * a,
        np.sqrt(gamma_1) * sigma_m,
        np.sqrt(gamma_phi) * sigma_z,
    ]

    delta_rms = np.linspace(-5 * chi, 4 * chi, 50)

    # num_points = 501
    # times = np.linspace(-250, 2000, num_points)

    sweep_data = np.zeros((len(delta_rms), num_points), dtype=complex)

    for i in range(len(delta_rms)):
        delta_rm = delta_rms[i]

        H = get_Hamiltonian(num_levels, delta_rm, delta_as, chi, epsilon_m, Omega, qubit_drive=H_qubit_drive)

        steady_state_a = -epsilon_m / (delta_rm - chi - 1j * kappa / 2)

        alpha = steady_state_a
        psi0 = qt.tensor(qt.coherent(num_levels, alpha), qt.basis(2, 0))

        options = qt.Options(max_step=1)
        result = qt.mesolve(H, psi0, times, collapse_operators, [a], args, options)

        a_expectation = result.expect[0]

        a_expectation = rotate_IQ(a_expectation)

        sweep_data[i, :] = a_expectation

    return times, sweep_data
def figure2de_cavity_bloch(delta_rms, num_points, times, tspan, *args):
    omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi = args

    epsilon_m = np.sqrt(kappa / 2)

    omega_s = omega_a + 3 * chi
    delta_as = omega_a - omega_s

    Omega = 0.05 * 2 * np.pi


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

    return result.t, sweep_data


def figure2de(num_levels, *args):

    omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi = args

    delta_rms = np.linspace(-5 * chi, 4 * chi, 50)

    num_points = 1001
    tspan = (-500, 2000)
    times = np.linspace(*tspan, num_points)

    # cavity bloch data
    t_cb, sweep_data_cb = figure2de_cavity_bloch(delta_rms, num_points, times, tspan, omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi)

    # qutip data
    t_qt, sweep_data_qt = figure2de_qutip(delta_rms, num_levels, num_points, times, omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi)

    # normalize
    print(np.max(sweep_data_qt))
    print(np.max(sweep_data_cb))
    sweep_data_qt *= np.max(sweep_data_cb)/np.max(sweep_data_qt)
    sweep_data = sweep_data_cb - sweep_data_qt

    t = np.copy(t_cb) / 1000
    t_step = t[1] - t[0]
    delta_rm_step = (delta_rms[-1] - delta_rms[0]) / (len(delta_rms) - 1)
    delta_rms *= -1

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(np.abs(np.imag(sweep_data)), cmap='seismic', vmin=-1, vmax=16, aspect='auto', origin='lower',
                   extent=(t[0] - t_step / 2, t[-1] + t_step / 2, (delta_rms[0] - delta_rm_step / 2) / chi,
                           (delta_rms[-1] + delta_rm_step / 2) / chi), )
    # plt.colorbar()
    axes[0].set_xlabel("Times $[\mu s]$")
    axes[0].set_xticks(np.arange(0, 1.6, 0.5))
    axes[0].set_ylabel("Detuning $\Delta_{mr}/\chi$")
    axes[0].set_title("Q")

    axes[1].imshow(np.abs(np.real(sweep_data)), cmap='seismic', vmin=0, vmax=14, aspect='auto', origin='lower',
                   extent=(t[0] - t_step / 2, t[-1] + t_step / 2, (delta_rms[0] - delta_rm_step / 2) / chi,
                           (delta_rms[-1] + delta_rm_step / 2) / chi))
    # plt.colorbar()
    axes[1].set_xlabel("Times $[\mu s]$")
    axes[1].set_xticks(np.arange(0, 1.6, 0.5))
    axes[1].set_title("I")
    axes[1].set_yticklabels([])
    axes[1].tick_params(axis='y', direction="inout", length=10)

    fig.suptitle("Cavity response to a weak measurement")

    fig.subplots_adjust(wspace=0)
    plt.show()

if __name__ == "__main__":
    # all in GHz

    omega_r = 6.44252 * 2 * np.pi
    kappa = 0.00169 * 2 * np.pi

    omega_a = 4.009 * 2 * np.pi

    g = 0.134 * 2 * np.pi

    chi = -0.00069 * 2 * np.pi

    gamma_1 = 0.00019 * 2 * np.pi

    # gamma_phi not given
    gamma_phi = 2 * gamma_1

    num_levels = 10

    figure2de(omega_r, num_levels, kappa, omega_a, g, chi, gamma_1, gamma_phi)
