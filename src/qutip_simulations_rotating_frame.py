import matplotlib.pyplot as plt
import numpy as np
import qutip as qt


def get_Hamiltonian(num_levels, delta_rm, delta_as, chi, epsilon_m=None, Omega=None, measurement_drive=None, qubit_drive=None):
    """

    :param num_levels:
    :param omega_r:
    :param omega_a:
    :param chi:
    :param measurement_drive: function that defines time dependent coefficient (amplitude) of measurement drive term
    :param qubit_drive: function that defines time dependent coefficient (amplitude) of qubit control drive term
    :return:
    """

    a = qt.tensor(qt.destroy(num_levels), qt.qeye(2))
    sigma_z = qt.tensor(qt.qeye(num_levels), -qt.sigmaz())
    sigma_m = qt.tensor(qt.qeye(num_levels), qt.destroy(2))
    sigma_p = qt.tensor(qt.qeye(num_levels), qt.create(2))

    hbar = 1
    H_disp = (
        hbar * (delta_rm + chi * sigma_z) * a.dag() * a
        + hbar / 2 * (delta_as + chi) * sigma_z
    )

    H = [H_disp]

    H_drive_m = hbar * (a.dag() + a)

    H_drive_s = hbar * (sigma_p + sigma_m)

    if not measurement_drive is None:
        H.append([H_drive_m, measurement_drive])
    else:
        H[0] += epsilon_m * H_drive_m

    if not qubit_drive is None:
        H.append([H_drive_s, qubit_drive])
    else:
        H[0] += Omega * H_drive_s

    if measurement_drive is None and qubit_drive is None:
        return H[0]
    else:
        return H

def rabi_oscillations():

    omega_r = 6.44252 * 2 * np.pi
    omega_a = 4.009 * 2 * np.pi

    chi = -0.00069 * 2 * np.pi

    epsilon_m = 0
    omega_m = omega_r - chi

    # pick omega so that pulse lasts 10 ns
    Omega = 0.025 * 2 * np.pi
    # lamb shift
    omega_s = omega_a + chi

    delta_rm = omega_r - omega_m
    delta_as = omega_a - omega_s

    args = {
        "epsilon_m": epsilon_m,
        "omega_m": omega_m,
        "Omega": Omega,
        "omega_s": omega_s,
    }

    num_levels = 3
    H = get_Hamiltonian(num_levels, delta_rm, delta_as, chi, epsilon_m, Omega)

    times = np.linspace(0, 30, 500)

    psi0 = qt.tensor(qt.basis(num_levels, 0), qt.basis(2, 0))

    a = qt.tensor(qt.destroy(num_levels), qt.qeye(2))
    sigma_m = qt.tensor(qt.qeye(num_levels), qt.destroy(2))

    result = qt.sesolve(H, psi0, times, [sigma_m.dag() * sigma_m, a.dag() * a], args)

    plt.plot(times, result.expect[0], label="qubit population")
    # plt.plot(times, result.expect[1], label="cavity photon population")
    plt.axvline(10, color="red", label="10 ns")
    plt.axvline(np.pi / Omega, color="orange", label="Pi Pulse")

    plt.ylabel("Population")
    plt.xlabel("t (ns)")

    plt.title("Rabi Oscillations")

    plt.legend()
    plt.show()

def pi_pulse():

    omega_r = 6.44252 * 2 * np.pi
    omega_a = 4.009 * 2 * np.pi

    chi = -0.00069 * 2 * np.pi

    epsilon_m = 0
    omega_m = omega_r - chi

    # pick omega so that pulse lasts 10 ns
    Omega = 0.025 * 2 * np.pi
    # lamb shift
    omega_s = omega_a + chi

    delta_rm = omega_r - omega_m
    delta_as = omega_a - omega_s

    t1 = 10
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

    num_levels = 3
    H = get_Hamiltonian(num_levels, delta_rm, delta_as, chi, epsilon_m, Omega, qubit_drive=H_qubit_drive)

    times = np.linspace(0, 30, 500)

    psi0 = qt.tensor(qt.basis(num_levels, 0), qt.basis(2, 0))

    a = qt.tensor(qt.destroy(num_levels), qt.qeye(2))
    sigma_m = qt.tensor(qt.qeye(num_levels), qt.destroy(2))

    options = qt.Options(max_step=1)
    result = qt.sesolve(H, psi0, times, [sigma_m.dag() * sigma_m, a.dag() * a], args, options)

    plt.plot(times, result.expect[0], label="qubit population")
    # plt.plot(times, result.expect[1], label="cavity photon population")
    plt.axvline(10, color="red", label="10 ns")
    plt.axvline(np.pi / Omega, color="orange", label="Pi Pulse")

    plt.ylabel("Population")
    plt.xlabel("t (ns)")

    plt.title("Pi Pulse")

    plt.legend()
    plt.show()

def pi_pulse_decay():

    omega_r = 6.44252 * 2 * np.pi
    omega_a = 4.009 * 2 * np.pi

    chi = -0.00069 * 2 * np.pi

    kappa = 0.00169 * 2 * np.pi
    gamma_1 = 0.00019 * 2 * np.pi

    # gamma_phi not given
    gamma_phi = 2 * gamma_1

    epsilon_m = 0
    omega_m = omega_r - chi

    # pick omega so that pulse lasts 10 ns
    Omega = 0.025 * 2 * np.pi
    # lamb shift
    omega_s = omega_a + chi

    delta_rm = omega_r - omega_m
    delta_as = omega_a - omega_s

    t1 = 10
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

    num_levels = 3
    H = get_Hamiltonian(num_levels, delta_rm, delta_as, chi, epsilon_m, Omega, qubit_drive=H_qubit_drive)

    times = np.linspace(0, 200, 400)

    psi0 = qt.tensor(qt.basis(num_levels, 0), qt.basis(2, 0))

    a = qt.tensor(qt.destroy(num_levels), qt.qeye(2))
    sigma_m = qt.tensor(qt.qeye(num_levels), qt.destroy(2))
    sigma_z = qt.tensor(qt.qeye(num_levels), -qt.sigmaz())

    collapse_operators = [
        np.sqrt(kappa) * a,
        np.sqrt(gamma_1) * sigma_m,
        np.sqrt(gamma_phi) * sigma_z,
    ]

    options = qt.Options(max_step=1)
    result = qt.mesolve(H, psi0, times, collapse_operators, [sigma_m.dag() * sigma_m, a.dag() * a], args, options)

    plt.plot(times, result.expect[0], label="qubit population")
    # plt.plot(times, result.expect[1], label="cavity photon population")
    plt.axvline(10, color="red", label="10 ns")
    plt.axvline(np.pi / Omega, color="orange", label="Pi Pulse")

    plt.ylabel("Population")
    plt.xlabel("t (ns)")

    plt.title("Pi Pulse with decay")

    plt.legend()
    plt.show()

def weak_measurement():

    omega_r = 6.44252 * 2 * np.pi
    omega_a = 4.009 * 2 * np.pi

    chi = -0.00069 * 2 * np.pi

    kappa = 0.00169 * 2 * np.pi
    gamma_1 = 0.00019 * 2 * np.pi

    # gamma_phi not given
    gamma_phi = 2 * gamma_1

    epsilon_m = np.sqrt(kappa/2)/100
    omega_m = omega_r - chi

    # pick omega so that pulse lasts 10 ns
    Omega = 0.025 * 2 * np.pi
    # lamb shift
    omega_s = omega_a + chi

    delta_rm = omega_r - omega_m
    delta_as = omega_a - omega_s

    t1 = 10
    t2 = t1 + 10

    args = {
        "epsilon_m": epsilon_m,
        "omega_m": omega_m,
        "Omega": Omega,
        "omega_s": omega_s,
        "t1": t1,
        "t2": t2
    }
    def H_measurement_drive(t, args):
        epsilon_m = args["epsilon_m"]
        t1 = args["t1"]
        t2 = args["t2"]
        if t1 < t <= t2:
            return epsilon_m
        else:
            return 0
    def H_qubit_drive(t, args):
        Omega = args["Omega"]
        t1 = args["t1"]
        t2 = args["t2"]
        if t1 < t <= t2:
            return Omega
        else:
            return 0

    num_levels = 50
    H = get_Hamiltonian(num_levels, delta_rm, delta_as, chi, epsilon_m, Omega, qubit_drive=H_qubit_drive)

    times = np.linspace(0, 1000, 500)

    steady_state_a = -epsilon_m / (delta_rm - chi - 1j * kappa / 2)
    steady_state_n = -2 * epsilon_m / kappa * np.imag(steady_state_a)

    # psi0 = qt.tensor(qt.basis(num_levels, 0), qt.basis(2, 0))
    alpha = steady_state_a
    psi0 = qt.tensor(qt.coherent(num_levels, alpha), qt.basis(2, 0))

    a = qt.tensor(qt.destroy(num_levels), qt.qeye(2))
    sigma_m = qt.tensor(qt.qeye(num_levels), qt.destroy(2))
    sigma_z = qt.tensor(qt.qeye(num_levels), -qt.sigmaz())

    collapse_operators = [
        np.sqrt(kappa) * a,
        np.sqrt(gamma_1) * sigma_m,
        np.sqrt(gamma_phi) * sigma_z,
    ]

    options = qt.Options(max_step=1)
    result = qt.mesolve(H, psi0, times, collapse_operators, [sigma_m.dag() * sigma_m, a.dag() * a, a], args, options)

    plt.plot(times, result.expect[0], label="qubit population")
    plt.plot(times, result.expect[1], label="cavity photon population")
    plt.plot(times, np.real(result.expect[2]), label="I")
    plt.plot(times, np.imag(result.expect[2]), label="Q")

    # plt.axhline(steady_state_n, color='red', linestyle=':')

    plt.ylabel("Population")
    plt.xlabel("t (ns)")

    plt.title("Weak measurement")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    # rabi_oscillations()
    # pi_pulse()
    # pi_pulse_decay()
    weak_measurement()
    # pi_pulse_strong_measurement()
