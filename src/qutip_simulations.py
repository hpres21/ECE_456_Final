import matplotlib.pyplot as plt
import numpy as np
import qutip as qt


def get_Hamiltonian(num_levels, omega_r, omega_a, chi):

    a = qt.tensor(qt.destroy(num_levels), qt.qeye(2))
    sigma_z = qt.tensor(qt.qeye(num_levels), -qt.sigmaz())
    sigma_m = qt.tensor(qt.qeye(num_levels), qt.destroy(2))
    sigma_p = qt.tensor(qt.qeye(num_levels), qt.create(2))

    hbar = 1
    H_disp = (
        hbar * (omega_r + chi * sigma_z) * a.dag() * a
        + hbar / 2 * (omega_a + chi) * sigma_z
    )

    H_drive_m_plus = a.dag()
    H_drive_m_minus = a

    H_drive_s_plus = sigma_p
    H_drive_s_minus = sigma_m

    def H_drive_m_plus_coeff(t, args):
        epsilon_m = args["epsilon_m"]
        omega_m = args["omega_m"]
        return epsilon_m * np.exp(-1j * omega_m * t)

    def H_drive_m_minus_coeff(t, args):
        epsilon_m = args["epsilon_m"]
        omega_m = args["omega_m"]
        return epsilon_m * np.exp(1j * omega_m * t)

    def H_drive_s_plus_coeff(t, args):
        Omega = args["Omega"]
        omega_s = args["omega_s"]
        return Omega * np.exp(-1j * omega_s * t)

    def H_drive_s_minus_coeff(t, args):
        Omega = args["Omega"]
        omega_s = args["omega_s"]
        return Omega * np.exp(1j * omega_s * t)

    return [
        H_disp,
        [H_drive_m_plus, H_drive_m_plus_coeff],
        [H_drive_m_minus, H_drive_m_minus_coeff],
        [H_drive_s_plus, H_drive_s_plus_coeff],
        [H_drive_s_minus, H_drive_s_minus_coeff],
    ]


def get_Hamiltonian_pulse(num_levels, omega_r, omega_a, chi, drive_coeffs):

    a = qt.tensor(qt.destroy(num_levels), qt.qeye(2))
    sigma_z = qt.tensor(qt.qeye(num_levels), -qt.sigmaz())
    sigma_m = qt.tensor(qt.qeye(num_levels), qt.destroy(2))
    sigma_p = qt.tensor(qt.qeye(num_levels), qt.create(2))

    hbar = 1
    H_disp = (
        hbar * (omega_r + chi * sigma_z) * a.dag() * a
        + hbar / 2 * (omega_a + chi) * sigma_z
    )

    H_drive_m_plus = a.dag()
    H_drive_m_minus = a

    H_drive_s_plus = sigma_p
    H_drive_s_minus = sigma_m

    return [
        H_disp,
        [H_drive_m_plus, drive_coeffs[0]],
        [H_drive_m_minus, drive_coeffs[1]],
        [H_drive_s_plus, drive_coeffs[2]],
        [H_drive_s_minus, drive_coeffs[3]],
    ]


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
    args = {
        "epsilon_m": epsilon_m,
        "omega_m": omega_m,
        "Omega": Omega,
        "omega_s": omega_s,
    }

    num_levels = 3
    H = get_Hamiltonian(num_levels, omega_r, omega_a, chi)

    times = np.linspace(0, 30, 500)

    psi0 = qt.tensor(qt.basis(num_levels, 0), qt.basis(2, 1))

    a = qt.tensor(qt.destroy(num_levels), qt.qeye(2))
    sigma_m = qt.tensor(qt.qeye(num_levels), qt.destroy(2))

    result = qt.sesolve(H, psi0, times, [sigma_m.dag() * sigma_m, a.dag() * a], args)

    plt.plot(times, result.expect[0], label="qubit population")
    plt.plot(times, result.expect[1], label="cavity photon population")
    plt.axvline(10, color="red", label="10 ns")
    plt.axvline(np.pi / Omega, color="orange", label="Pi Pulse")

    plt.ylabel("Population")
    plt.xlabel("t (ns)")

    plt.title("Rabi Oscillations")

    plt.legend()
    plt.show()

def rabi_oscillations_with_measurement_and_decay():
    omega_r = 6.44252 * 2 * np.pi
    omega_a = 4.009 * 2 * np.pi

    chi = -0.00069 * 2 * np.pi

    kappa = 0.00169 * 2 * np.pi
    gamma_1 = 0.00019 * 2 * np.pi * 0

    # gamma_phi not given
    gamma_phi = 2 * gamma_1

    epsilon_m = np.sqrt(kappa/2)
    omega_m = omega_r - chi

    # pick omega so that pulse lasts 10 ns
    Omega = 0.025 * 2 * np.pi
    # lamb shift
    omega_s = omega_a + chi
    args = {
        "epsilon_m": epsilon_m,
        "omega_m": omega_m,
        "Omega": Omega,
        "omega_s": omega_s,
    }

    num_levels = 10
    H = get_Hamiltonian(num_levels, omega_r, omega_a, chi)
    # H = get_Hamiltonian_pulse(num_levels, omega_r, omega_a, chi)

    times = np.linspace(0, 1500, 10000)

    psi0 = qt.tensor(qt.basis(num_levels, 0), qt.basis(2, 1))

    a = qt.tensor(qt.destroy(num_levels), qt.qeye(2))
    sigma_m = qt.tensor(qt.qeye(num_levels), qt.destroy(2))
    sigma_z = qt.tensor(qt.qeye(num_levels), -qt.sigmaz())

    collapse_operators = [
        np.sqrt(kappa) * a,
        np.sqrt(gamma_1) * sigma_m,
        np.sqrt(gamma_phi) * sigma_z,
    ]
    result = qt.mesolve(H, psi0, times, collapse_operators, [sigma_m.dag() * sigma_m, a.dag() * a], args)

    plt.plot(times, result.expect[0], label="qubit population")
    plt.plot(times, result.expect[1], label="cavity photon population")

    plt.ylabel("Population")
    plt.xlabel("t (ns)")

    plt.title("Rabi Oscillations")

    plt.legend()
    plt.show()



def short_pi_pulse():
    omega_r = 6.44252 * 2 * np.pi
    omega_a = 4.009 * 2 * np.pi

    chi = -0.00069 * 2 * np.pi

    epsilon_m = 0
    omega_m = omega_r - chi

    # pick omega so that pulse lasts 10 ns
    Omega = 0.025 * 2 * np.pi
    # lamb shift
    omega_s = omega_a + chi
    args = {
        "epsilon_m": epsilon_m,
        "omega_m": omega_m,
        "Omega": Omega,
        "omega_s": omega_s,
        "t1": 10,
        "t2": 20,
    }

    num_levels = 3

    def H_drive_m_plus_coeff(t, args):
        return 0

    def H_drive_m_minus_coeff(t, args):
        return 0

    def H_drive_s_plus_coeff(t, args):
        Omega = args["Omega"]
        omega_s = args["omega_s"]
        t1 = args["t1"]
        t2 = args["t2"]
        if t1 < t <= t2:
            return Omega * np.exp(-1j * omega_s * t)
        else:
            return 0

    def H_drive_s_minus_coeff(t, args):
        Omega = args["Omega"]
        omega_s = args["omega_s"]
        t1 = args["t1"]
        t2 = args["t2"]
        if t1 < t <= t2:
            return Omega * np.exp(1j * omega_s * t)
        else:
            return 0

    H = get_Hamiltonian_pulse(
        num_levels,
        omega_r,
        omega_a,
        chi,
        [
            H_drive_m_plus_coeff,
            H_drive_m_minus_coeff,
            H_drive_s_plus_coeff,
            H_drive_s_minus_coeff,
        ],
    )

    times = np.linspace(0, 50, 500)

    psi0 = qt.tensor(qt.basis(num_levels, 0), qt.basis(2, 0))

    a = qt.tensor(qt.destroy(num_levels), qt.qeye(2))
    sigma_m = qt.tensor(qt.qeye(num_levels), qt.destroy(2))

    result = qt.sesolve(H, psi0, times, [sigma_m.dag() * sigma_m, a.dag() * a], args)

    plt.plot(times, result.expect[0], label="qubit population")
    plt.plot(times, result.expect[1], label="cavity photon population")
    plt.axvline(args["t1"], color="red", label=f"{args['t1']} ns")
    plt.axvline(args["t2"], color="red", label=f"{args['t2']} ns")

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
    args = {
        "epsilon_m": epsilon_m,
        "omega_m": omega_m,
        "Omega": Omega,
        "omega_s": omega_s,
        "t1": 10,
        "t2": 20,
    }

    num_levels = 3

    def H_drive_m_plus_coeff(t, args):
        return 0

    def H_drive_m_minus_coeff(t, args):
        return 0

    def H_drive_s_plus_coeff(t, args):
        Omega = args["Omega"]
        omega_s = args["omega_s"]
        t1 = args["t1"]
        t2 = args["t2"]
        if t1 < t <= t2:
            return Omega * np.exp(-1j * omega_s * t)
        else:
            return 0

    def H_drive_s_minus_coeff(t, args):
        Omega = args["Omega"]
        omega_s = args["omega_s"]
        t1 = args["t1"]
        t2 = args["t2"]
        if t1 < t <= t2:
            return Omega * np.exp(1j * omega_s * t)
        else:
            return 0

    H = get_Hamiltonian_pulse(
        num_levels,
        omega_r,
        omega_a,
        chi,
        [
            H_drive_m_plus_coeff,
            H_drive_m_minus_coeff,
            H_drive_s_plus_coeff,
            H_drive_s_minus_coeff,
        ],
    )

    times = np.linspace(0, 200, 1000)

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
    result = qt.mesolve(
        H,
        psi0,
        times,
        collapse_operators,
        [sigma_m.dag() * sigma_m],
        args,
        options,
    )

    plt.plot(times, result.expect[0], label="qubit population")
    plt.axvline(args["t1"], color="red", label=f"{args['t1']} ns")
    plt.axvline(args["t2"], color="red", label=f"{args['t2']} ns")

    plt.ylabel("Population")
    plt.xlabel("t (ns)")

    plt.title("Pi Pulse with decay")

    plt.legend()
    plt.show()


def pi_pulse_during_measurement():
    omega_r = 6.44252 * 2 * np.pi
    omega_a = 4.009 * 2 * np.pi

    chi = -0.00069 * 2 * np.pi

    kappa = 0.00169 * 2 * np.pi
    gamma_1 = 0.00019 * 2 * np.pi

    # gamma_phi not given
    gamma_phi = 2 * gamma_1

    epsilon_m = np.sqrt(kappa / 2)
    omega_m = omega_r - chi

    # pick omega so that pulse lasts 10 ns
    Omega = 0.025 * 2 * np.pi
    # lamb shift
    omega_s = omega_a + chi
    args = {
        "epsilon_m": epsilon_m,
        "omega_m": omega_m,
        "Omega": Omega * 0,
        "omega_s": omega_s,
        "t1": 1000,
        "t2": 1010,
    }

    num_levels = 50

    def H_drive_m_plus_coeff(t, args):
        epsilon_m = args["epsilon_m"]
        omega_m = args["omega_m"]
        return epsilon_m * np.exp(-1j * omega_m * t)

    def H_drive_m_minus_coeff(t, args):
        epsilon_m = args["epsilon_m"]
        omega_m = args["omega_m"]
        return epsilon_m * np.exp(1j * omega_m * t)

    def H_drive_s_plus_coeff(t, args):
        Omega = args["Omega"]
        omega_s = args["omega_s"]
        t1 = args["t1"]
        t2 = args["t2"]
        if t1 < t <= t2:
            return Omega * np.exp(-1j * omega_s * t)
        else:
            return 0

    def H_drive_s_minus_coeff(t, args):
        Omega = args["Omega"]
        omega_s = args["omega_s"]
        t1 = args["t1"]
        t2 = args["t2"]
        if t1 < t <= t2:
            return Omega * np.exp(1j * omega_s * t)
        else:
            return 0

    H = get_Hamiltonian_pulse(
        num_levels,
        omega_r,
        omega_a,
        chi,
        [
            H_drive_m_plus_coeff,
            H_drive_m_minus_coeff,
            H_drive_s_plus_coeff,
            H_drive_s_minus_coeff,
        ],
    )

    times = np.linspace(0, 1000, 2000)

    # photon_state

    delta_rm = omega_r - omega_m
    alpha = -epsilon_m / (delta_rm - chi - 1j * kappa / 2)

    psi0 = qt.tensor(qt.coherent(num_levels, alpha), qt.basis(2, 0))
    # psi0 = qt.tensor(qt.basis(num_levels, 0), qt.basis(2, 0))

    a = qt.tensor(qt.destroy(num_levels), qt.qeye(2))
    sigma_m = qt.tensor(qt.qeye(num_levels), qt.destroy(2))
    sigma_z = qt.tensor(qt.qeye(num_levels), -qt.sigmaz())

    collapse_operators = [
        np.sqrt(kappa) * a,
        np.sqrt(gamma_1) * sigma_m,
        np.sqrt(gamma_phi) * sigma_z,
    ]

    # options = qt.Options(max_step=1, nsteps=1000)
    options = qt.Options(nsteps=5000)
    result = qt.mesolve(
        H,
        psi0,
        times,
        collapse_operators,
        [],
        args,
        options
    )

    result_str = "result_test"

    qt.qsave(result, result_str)

    result = qt.qload(result_str)
    last_state = result.states[-1]



    expect_array = np.zeros((3, len(result.states)), dtype=complex)
    for i, op in enumerate([sigma_m.dag() * sigma_m, a.dag() * a, a]):
        for j in range(len(result.states)):
            expect_array[i, j] = qt.expect(op, result.states[j])

    plt.plot(times, np.abs(expect_array[0, :]), label="qubit population")
    plt.plot(times, np.abs(expect_array[1, :]), label="cavity photon population")
    plt.plot(times, np.real(expect_array[2, :]), label="I")
    plt.plot(times, np.imag(expect_array[2, :]), label="Q")
    # plt.axvline(args["t1"], color='red', label=f"{args['t1']} ns")
    # plt.axvline(args["t2"], color='red', label=f"{args['t2']} ns")

    plt.ylabel("Population")
    plt.xlabel("t (ns)")

    plt.title("Pi Pulse with measurement drive")

    plt.legend()
    plt.show()


    # new measurement starting at last state of old measurement

    args = {
        "epsilon_m": epsilon_m,
        "omega_m": omega_m,
        "Omega": Omega,
        "omega_s": omega_s,
        "t1": 100,
        "t2": 110,
    }

    psi0 = last_state
    result = qt.mesolve(
        H,
        psi0,
        times,
        collapse_operators,
        [sigma_m.dag() * sigma_m, a.dag() * a, a],
        args,
        options,
    )

    plt.plot(times, np.abs(result.expect[0]), label="qubit population")
    plt.plot(times, np.abs(result.expect[1]), label="cavity photon population")
    plt.plot(times, np.real(result.expect[2]), label="I")
    plt.plot(times, np.imag(result.expect[2]), label="Q")
    # plt.axvline(args["t1"], color='red', label=f"{args['t1']} ns")
    # plt.axvline(args["t2"], color='red', label=f"{args['t2']} ns")

    plt.ylabel("Population")
    plt.xlabel("t (ns)")

    plt.title("Pi Pulse with measurement drive")

    plt.legend()
    plt.show()

def pi_pulse_strong_measurement():
    omega_r = 6.44252 * 2 * np.pi
    omega_a = 4.009 * 2 * np.pi

    chi = -0.00069 * 2 * np.pi

    kappa = 0.00169 * 2 * np.pi
    gamma_1 = 0.00019 * 2 * np.pi

    # gamma_phi not given
    gamma_phi = 2 * gamma_1

    epsilon_m = np.sqrt(kappa / 2)
    omega_m = omega_r - chi

    # pick omega so that pulse lasts 10 ns
    Omega = 0.025 * 2 * np.pi
    # lamb shift
    omega_s = omega_a + chi
    args = {
        "epsilon_m": epsilon_m,
        "omega_m": omega_m,
        "Omega": Omega,
        "omega_s": omega_s,
        "t1": 200,
        "t2": 210,
    }

    num_levels = 5

    def H_drive_m_plus_coeff(t, args):
        epsilon_m = args["epsilon_m"]
        omega_m = args["omega_m"]
        t2 = args["t2"]
        if t > t2:
            return epsilon_m * np.exp(-1j * omega_m * t)
        else:
            return 0

    def H_drive_m_minus_coeff(t, args):
        epsilon_m = args["epsilon_m"]
        omega_m = args["omega_m"]
        t2 = args["t2"]
        if t > t2:
            return epsilon_m * np.exp(1j * omega_m * t)
        else:
            return 0

    def H_drive_s_plus_coeff(t, args):
        Omega = args["Omega"]
        omega_s = args["omega_s"]
        t1 = args["t1"]
        t2 = args["t2"]
        if t1 < t <= t2:
            return Omega * np.exp(-1j * omega_s * t)
        else:
            return 0

    def H_drive_s_minus_coeff(t, args):
        Omega = args["Omega"]
        omega_s = args["omega_s"]
        t1 = args["t1"]
        t2 = args["t2"]
        if t1 < t <= t2:
            return Omega * np.exp(1j * omega_s * t)
        else:
            return 0

    H = get_Hamiltonian_pulse(
        num_levels,
        omega_r,
        omega_a,
        chi,
        [
            H_drive_m_plus_coeff,
            H_drive_m_minus_coeff,
            H_drive_s_plus_coeff,
            H_drive_s_minus_coeff,
        ],
    )

    times = np.linspace(0, 2000, 1000)

    # alpha = 1j
    # psi0 = qt.tensor(qt.coherent(num_levels, alpha), qt.basis(2, 0))
    psi0 = qt.tensor(qt.basis(num_levels, 0), qt.basis(2, 0))

    a = qt.tensor(qt.destroy(num_levels), qt.qeye(2))
    sigma_m = qt.tensor(qt.qeye(num_levels), qt.destroy(2))
    sigma_z = qt.tensor(qt.qeye(num_levels), -qt.sigmaz())

    collapse_operators = [
        np.sqrt(kappa) * a,
        np.sqrt(gamma_1) * sigma_m,
        np.sqrt(gamma_phi) * sigma_z,
    ]

    options = qt.Options(max_step=1, nsteps=1000)
    # options = qt.Options()
    result = qt.mesolve(
        H,
        psi0,
        times,
        collapse_operators,
        [sigma_m.dag() * sigma_m, a.dag() * a, a],
        args,
        options,
    )

    plt.plot(times, result.expect[0], label="qubit population")
    plt.plot(times, result.expect[1], label="cavity photon population")
    plt.plot(times, np.real(result.expect[2]), label="I")
    plt.plot(times, np.imag(result.expect[2]), label="Q")
    # plt.axvline(args["t1"], color='red', label=f"{args['t1']} ns")
    # plt.axvline(args["t2"], color='red', label=f"{args['t2']} ns")

    plt.ylabel("Population")
    plt.xlabel("t (ns)")

    plt.title("Pi Pulse with measurement drive")

    plt.legend()
    plt.show()

if __name__ == "__main__":

    # rabi_oscillations_with_measurement_and_decay()
    # short_pi_pulse()
    # pi_pulse_decay()
    pi_pulse_during_measurement()
    # pi_pulse_strong_measurement()
