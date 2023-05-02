import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from qutip_simulations_rotating_frame import get_Hamiltonian


def rotate_IQ(a):
    # want last element of Q to be fully imaginary
    last = a[-1]
    return a * 1j * np.conjugate(last) / np.abs(last)


def figure2bc():
    omega_r = 6.44252 * 2 * np.pi
    omega_a = 4.009 * 2 * np.pi

    chi = -0.00069 * 2 * np.pi

    kappa = 0.00169 * 2 * np.pi
    gamma_1 = 0.00019 * 2 * np.pi

    # gamma_phi not given
    gamma_phi = 2 * gamma_1

    epsilon_m = np.sqrt(kappa / 2) / 10
    omega_m = omega_r - chi

    # pick omega so that pulse lasts 10 ns
    Omega = 0.025 * 2 * np.pi
    # lamb shift
    omega_s = omega_a + chi

    delta_rm = omega_r - omega_m
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

    num_levels = 10
    H = get_Hamiltonian(num_levels, delta_rm, delta_as, chi, epsilon_m, Omega, qubit_drive=H_qubit_drive)

    times = np.linspace(-250, 2000, 501)

    steady_state_a = -epsilon_m / (delta_rm - chi - 1j * kappa / 2)
    steady_state_n = -2 * epsilon_m / kappa * np.imag(steady_state_a)

    print(steady_state_n)

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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(result.times / 1000, 0.6 - np.imag(result.expect[2]), color='black')

    axes[0].set_xlabel("Times $[\mu s]$")
    axes[0].set_xticks(np.arange(0, 1.6, 0.5))
    axes[0].set_ylabel("Q [a.u.]")
    axes[0].set_ylim(0, 2)

    axes[1].plot(result.times / 1000, np.real(result.expect[2]), color='black')

    axes[1].set_xlabel("Times $[\mu s]$")
    axes[1].set_xticks(np.arange(0, 1.6, 0.5))

    # axes[1].set_yticklabels([])
    # axes[1].tick_params(axis='y', direction="inout", length=10)

    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.tick_right()
    axes[1].set_ylabel("I [a.u.]")
    axes[1].set_ylim(0, 2)

    fig.suptitle("Cavity response to a weak measurement for $\Delta_{mr} = -\chi$")

    fig.subplots_adjust(wspace=0)
    plt.show()

    plt.show()


def figure2de():
    omega_r = 6.44252 * 2 * np.pi
    omega_a = 4.009 * 2 * np.pi

    chi = -0.00069 * 2 * np.pi

    kappa = 0.00169 * 2 * np.pi
    gamma_1 = 0.00019 * 2 * np.pi

    # gamma_phi not given
    gamma_phi = 2 * gamma_1

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

    num_levels = 10

    a = qt.tensor(qt.destroy(num_levels), qt.qeye(2))
    sigma_m = qt.tensor(qt.qeye(num_levels), qt.destroy(2))
    sigma_z = qt.tensor(qt.qeye(num_levels), -qt.sigmaz())

    collapse_operators = [
        np.sqrt(kappa) * a,
        np.sqrt(gamma_1) * sigma_m,
        np.sqrt(gamma_phi) * sigma_z,
    ]

    delta_rms = np.linspace(-5 * chi, 4 * chi, 50)

    num_points = 501
    times = np.linspace(-250, 2000, num_points)

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

        # # plt.plot(times, result.expect[0], label="qubit population")
        # # plt.plot(times, result.expect[1], label="cavity photon population")
        # plt.plot(times, np.real(a_expectation), label="I")
        # plt.plot(times, -np.imag(a_expectation), label="Q")
        #
        # # plt.axhline(steady_state_n, color='red', linestyle=':')
        #
        # plt.ylabel("Population")
        # plt.xlabel("t (ns)")
        #
        # plt.title("Weak measurement")
        #
        # plt.legend()
        # plt.show()

        sweep_data[i, :] = a_expectation

    t = np.copy(result.times) / 1000
    t_step = t[1] - t[0]
    delta_rms *= -1
    delta_rm_step = (delta_rms[-1] - delta_rms[0]) / (len(delta_rms) - 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(np.abs(np.imag(sweep_data)), cmap='Reds', vmin=0, vmax=2, aspect='auto', origin='lower',
                   extent=(t[0] - t_step / 2, t[-1] + t_step / 2, (delta_rms[0] - delta_rm_step / 2) / chi,
                           (delta_rms[-1] + delta_rm_step / 2) / chi), )
    # plt.colorbar()
    axes[0].set_xlabel("Times $[\mu s]$")
    axes[0].set_xticks(np.arange(0, 1.6, 0.5))
    axes[0].set_ylabel("Detuning $\Delta_{mr}/\chi$")
    axes[0].set_title("Q")

    axes[1].imshow(np.abs(np.real(sweep_data)), cmap='Reds', vmin=0, vmax=2, aspect='auto', origin='lower',
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


def figure3():
    omega_r = 6.44252 * 2 * np.pi
    omega_a = 4.009 * 2 * np.pi

    chi = -0.00069 * 2 * np.pi

    kappa = 0.00169 * 2 * np.pi
    gamma_1 = 0.00019 * 2 * np.pi

    # gamma_phi not given
    gamma_phi = 2 * gamma_1

    epsilon_m = np.sqrt(kappa / 2) / 10
    omega_m = omega_r - chi

    # pick omega so that pulse lasts 10 ns
    Omega = 0.025 * 2 * np.pi
    # lamb shift
    omega_s = omega_a + chi

    delta_as = omega_a - omega_s

    t1 = 0
    t2 = t1 + 10

    def H_qubit_drive(t, args):
        Omega = args["Omega"]
        t1 = args["t1"]
        t2 = args["t2"]
        if t1 < t <= t2:
            return Omega
        else:
            return 0

    args = {
        "epsilon_m": epsilon_m,
        "omega_m": omega_m,
        "Omega": Omega,
        "omega_s": omega_s,
        "t1": t1,
        "t2": t2
    }

    num_levels = 10
    # times = np.linspace(-250, 2000, 501)
    times = [-100, 180, 740]

    a = qt.tensor(qt.destroy(num_levels), qt.qeye(2))
    sigma_m = qt.tensor(qt.qeye(num_levels), qt.destroy(2))
    sigma_z = qt.tensor(qt.qeye(num_levels), -qt.sigmaz())

    collapse_operators = [
        np.sqrt(kappa) * a,
        np.sqrt(gamma_1) * sigma_m,
        np.sqrt(gamma_phi) * sigma_z,
    ]

    options = qt.Options(max_step=1, nsteps=100000)

    delta_rms = np.linspace(-5 * chi, 4 * chi, 100)
    # delta_rms = np.array([-2, -1, 0, 1, 2])

    I_data = np.zeros((4, len(delta_rms)))
    Q_data = np.zeros((4, len(delta_rms)))

    for i in range(len(delta_rms)):
        delta_rm = delta_rms[i]

        H = get_Hamiltonian(num_levels, delta_rm, delta_as, chi, epsilon_m, Omega, qubit_drive=H_qubit_drive)

        steady_state_a = -epsilon_m / (delta_rm - chi - 1j * kappa / 2)

        alpha = steady_state_a
        psi0 = qt.tensor(qt.coherent(num_levels, alpha), qt.basis(2, 0))

        result = qt.mesolve(H, psi0, times, collapse_operators, [a], args, options)

        I_data[:-1, i] = np.real(result.expect[0])
        Q_data[:-1, i] = np.imag(result.expect[0])

        I_data[-1, i] = -epsilon_m * (delta_rm + chi) / (np.power(delta_rm + chi, 2) + np.power(kappa / 2, 2))
        Q_data[-1, i] = -epsilon_m * (kappa / 2) / (np.power(delta_rm + chi, 2) + np.power(kappa / 2, 2))

    fig, axes = plt.subplots(2, 1, figsize=(8, 5))

    labels = ["0 ns", "180 ns", "740 ns", "excited state infinite lifetime"]
    colors = ["red", "blue", "lime", "black"]

    for i in range(I_data.shape[0]):
        axes[0].plot(delta_rms/chi, -Q_data[i, :], color=colors[i], label=labels[i])
        axes[1].plot(delta_rms/chi, I_data[i, :], color=colors[i], label=labels[i])

    axes[0].set_xlabel("Detuning $-\Delta_{mr}/\chi$")
    axes[0].set_ylabel("Q [a.u.]")
    axes[0].legend()

    axes[1].set_xlabel("Detuning $-\Delta_{mr}/\chi$")
    axes[1].set_ylabel("I [a.u.]")
    axes[1].legend()

    fig.suptitle("Resonator Transmission Spectrum")

    plt.tight_layout()
    plt.show()


def figure4():
    omega_r = 6.44252 * 2 * np.pi
    omega_a = 4.009 * 2 * np.pi

    chi = -0.00069 * 2 * np.pi

    kappa = 0.00169 * 2 * np.pi
    gamma_1 = 0.00019 * 2 * np.pi

    # gamma_phi not given
    gamma_phi = 2 * gamma_1

    epsilon_m = np.sqrt(kappa / 2) / 10
    omega_m = omega_r - chi

    # no qubit drive for this figure
    Omega = 0
    # lamb shift
    omega_s = omega_a + chi

    delta_rm = omega_r - omega_m
    delta_as = omega_a - omega_s

    t1 = 0

    args = {
        "epsilon_m": epsilon_m,
        "omega_m": omega_m,
        "Omega": Omega,
        "omega_s": omega_s,
        "t1": t1,
    }

    def H_measurement_drive(t, args):
        epsilon_m = args["epsilon_m"]
        t1 = args["t1"]
        if t >= t1:
            return epsilon_m
        else:
            return 0

    num_levels = 10
    H = get_Hamiltonian(num_levels, delta_rm, delta_as, chi, epsilon_m, Omega, measurement_drive=None)

    times = np.linspace(0, 2000, 501)

    psi0_g = qt.tensor(qt.basis(num_levels, 0), qt.basis(2, 0))
    psi0_e = qt.tensor(qt.basis(num_levels, 0), qt.basis(2, 1))

    a = qt.tensor(qt.destroy(num_levels), qt.qeye(2))
    sigma_m = qt.tensor(qt.qeye(num_levels), qt.destroy(2))
    sigma_z = qt.tensor(qt.qeye(num_levels), -qt.sigmaz())

    collapse_operators = [
        np.sqrt(kappa) * a,
        np.sqrt(gamma_1) * sigma_m,
        np.sqrt(gamma_phi) * sigma_z,
    ]

    result_g = qt.mesolve(H, psi0_g, times, collapse_operators, [sigma_m.dag() * sigma_m, a.dag() * a, a], args)
    result_e = qt.mesolve(H, psi0_e, times, collapse_operators, [sigma_m.dag() * sigma_m, a.dag() * a, a], args)

    I_g = np.abs(np.real(result_g.expect[2]))
    Q_g = np.abs(np.imag(result_g.expect[2]))

    I_e = np.abs(np.real(result_e.expect[2]))
    Q_e = np.abs(np.imag(result_e.expect[2]))

    fig, ax = plt.subplots(3, 1, figsize=(6, 8))
    ax[0].plot(result_g.times / 1000, Q_e, label="|e>", c='red')
    ax[0].plot(result_g.times / 1000, Q_g, label="|g>", c='blue')
    ax[0].set_xlabel("Time (us)")
    ax[0].set_ylabel("Q [a.u.]")
    ax[0].legend()

    ax[1].plot(result_g.times / 1000, I_e, label='|e>', c='red')
    ax[1].plot(result_g.times / 1000, I_g, label='|g>', c='blue')
    ax[1].legend()
    ax[1].set_xlabel("Time (us)")
    ax[1].set_ylabel("I [a.u.]")
    ax[1].set_ylim(0, 1.4)

    ax[2].scatter(Q_e[::5], I_e[::5], label='|e>', marker='o', c='red')
    ax[2].scatter(Q_g[::5], I_g[::5], label='|g>', marker='x', c='blue')
    ax[2].legend()
    ax[2].set_xlabel('Q quadrature [a.u.]')
    ax[2].set_ylabel('I quadrature [a.u.]')

    fig.suptitle("Cavity Response for a Pulsed Measurement")

    plt.tight_layout()
    plt.show()


def figure5():
    omega_r = 6.44252 * 2 * np.pi
    omega_a = 4.009 * 2 * np.pi

    chi = -0.00069 * 2 * np.pi

    kappa = 0.00169 * 2 * np.pi
    gamma_1 = 0.00019 * 2 * np.pi

    # gamma_phi not given
    gamma_phi = 2 * gamma_1

    epsilon_m = np.sqrt(kappa / 2) / 10
    omega_m = omega_r - chi

    # np pi pulse for this figure
    Omega = 0
    # lamb shift
    omega_s = omega_a + chi

    delta_as = omega_a - omega_s

    args = {
        "epsilon_m": epsilon_m,
        "omega_m": omega_m,
        "Omega": Omega,
        "omega_s": omega_s,
    }

    num_levels = 10
    num_points = 501
    times = np.linspace(0, 2000, num_points)

    a = qt.tensor(qt.destroy(num_levels), qt.qeye(2))
    sigma_m = qt.tensor(qt.qeye(num_levels), qt.destroy(2))
    sigma_z = qt.tensor(qt.qeye(num_levels), -qt.sigmaz())

    collapse_operators = [
        np.sqrt(kappa) * a,
        np.sqrt(gamma_1) * sigma_m,
        np.sqrt(gamma_phi) * sigma_z,
    ]

    psi0_g = qt.tensor(qt.basis(num_levels, 0), qt.basis(2, 0))
    psi0_e = qt.tensor(qt.basis(num_levels, 0), qt.basis(2, 1))

    delta_rms = np.linspace(-5 * chi, 4 * chi, 100)
    # delta_rms = np.array([-2, -1, 0, 1, 2])

    sweep_data_ground = np.zeros((len(delta_rms), num_points), dtype=complex)
    sweep_data_excited = np.zeros((len(delta_rms), num_points), dtype=complex)

    for i in range(len(delta_rms)):
        delta_rm = delta_rms[i]

        H = get_Hamiltonian(num_levels, delta_rm, delta_as, chi, epsilon_m, Omega)

        result_g = qt.mesolve(H, psi0_g, times, collapse_operators, [a], args)
        result_e = qt.mesolve(H, psi0_e, times, collapse_operators, [a], args)

        a_ground = result_g.expect[0]
        a_excited = result_e.expect[0]

        a_ground = rotate_IQ(a_ground)
        a_excited = rotate_IQ(a_excited)

        sweep_data_ground[i, :] = a_ground
        sweep_data_excited[i, :] = a_excited

    # single traces
    delta_rms_trace = np.array([-1, 0.5, 2]) * chi
    single_trace_data_g = np.zeros((len(delta_rms_trace), num_points), dtype=complex)
    single_trace_data_e = np.zeros((len(delta_rms_trace), num_points), dtype=complex)
    for i in range(len(delta_rms_trace)):
        delta_rm = delta_rms_trace[i]

        H = get_Hamiltonian(num_levels, delta_rm, delta_as, chi, epsilon_m, Omega)

        result_g = qt.mesolve(H, psi0_g, times, collapse_operators, [a], args)
        result_e = qt.mesolve(H, psi0_e, times, collapse_operators, [a], args)

        a_ground = result_g.expect[0]
        a_excited = result_e.expect[0]

        a_ground = rotate_IQ(a_ground)
        a_excited = rotate_IQ(a_excited)

        single_trace_data_g[i, :] = a_ground
        single_trace_data_e[i, :] = a_excited

    t = np.copy(result_g.times) / 1000
    t_step = t[1] - t[0]
    delta_rm_step = (delta_rms[-1] - delta_rms[0]) / (len(delta_rms) - 1)
    delta_rms *= -1

    print(delta_rms)

    # Fig 5a ----------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1], 'width_ratios': [1, 1]})

    im1 = axes[0,0].imshow(np.imag(sweep_data_ground), cmap='seismic', vmin=-3, vmax=3, aspect='auto', origin='lower',
                         extent=(t[0] - t_step / 2, t[-1] + t_step / 2, (delta_rms[0] - delta_rm_step / 2) / chi,
                                 (delta_rms[-1] + delta_rm_step / 2) / chi), )
    # plt.colorbar(im1, ax=axes[0])
    axes[0,0].set_xlabel("Times $[\mu s]$")
    axes[0,0].set_xticks(np.arange(0, 1.6, 0.5))
    axes[0,0].set_ylabel("Detuning $\Delta_{rm}/\chi$")
    axes[0,0].set_title("Q")

    im2 = axes[0,1].imshow(-np.real(sweep_data_ground), cmap='seismic', vmin=-1, vmax=1, aspect='auto', origin='lower',
                         extent=(t[0] - t_step / 2, t[-1] + t_step / 2, (delta_rms[0] - delta_rm_step / 2) / chi,
                                 (delta_rms[-1] + delta_rm_step / 2) / chi))

    # plt.colorbar(im2, ax=axes[1])
    axes[0,1].set_xlabel("Times $[\mu s]$")
    axes[0,1].set_xticks(np.arange(0, 1.6, 0.5))
    axes[0,1].set_title("I")
    axes[0,1].set_yticklabels([])
    axes[0,1].tick_params(axis='y', direction="inout", length=10)

    colors = ['red', 'black', 'blue']
    for i in range(len(delta_rms_trace)):
        axes[0, 0].axhline(-delta_rms_trace[i] / chi, color=colors[i], linestyle=":")
        axes[0, 1].axhline(-delta_rms_trace[i] / chi, color=colors[i], linestyle=":")

        axes[1, 0].plot(t, np.imag(single_trace_data_g[i, :]), color=colors[i],
                        label=f"$\Delta_{{mr}} \\approx$ {-delta_rms_trace[i] / chi}$\chi$")
        axes[1, 1].plot(t, -np.real(single_trace_data_g[i, :]), color=colors[i],
                        label=f"$\Delta_{{mr}} \\approx$ {-delta_rms_trace[i] / chi}$\chi$")

    axes[1, 0].set_xticks(np.arange(0, 1.6, 0.5))
    axes[1, 0].set_xlabel("Times $(\mu s)$")
    axes[1, 0].set_ylabel("Q [a.u.]")
    axes[1, 0].set_ylim(-0.7, 1.4)
    axes[1, 0].legend()

    axes[1, 1].set_xticks(np.arange(0, 1.6, 0.5))
    axes[1, 1].set_xlabel("Times $(\mu s)$")
    axes[1, 1].yaxis.set_label_position("right")
    axes[1, 1].yaxis.tick_right()
    axes[1, 1].set_ylabel("I [a.u.]")
    axes[1, 1].set_ylim(-0.7, 1.4)
    axes[1, 1].legend()

    fig.suptitle("Cavity response to a strong measurement in ground state")

    fig.subplots_adjust(wspace=0)
    plt.show()

    # Fig 5b ----------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1], 'width_ratios': [1, 1]})

    im1 = axes[0,0].imshow(np.imag(sweep_data_excited), cmap='seismic', vmin=-3, vmax=3, aspect='auto', origin='lower',
                         extent=(t[0] - t_step / 2, t[-1] + t_step / 2, (delta_rms[0] - delta_rm_step / 2) / chi,
                                 (delta_rms[-1] + delta_rm_step / 2) / chi), )

    # plt.colorbar(im1, ax=axes[0])
    axes[0,0].set_xlabel("Times $[\mu s]$")
    axes[0,0].set_xticks(np.arange(0, 1.6, 0.5))
    axes[0,0].set_ylabel("Detuning $\Delta_{rm}/\chi$")
    axes[0,0].set_title("Q")

    im2 = axes[0,1].imshow(-np.real(sweep_data_excited), cmap='seismic', vmin=-1, vmax=1, aspect='auto', origin='lower',
                         extent=(t[0] - t_step / 2, t[-1] + t_step / 2, (delta_rms[0] - delta_rm_step / 2) / chi,
                                 (delta_rms[-1] + delta_rm_step / 2) / chi))

    # plt.colorbar(im2, ax=axes[1])
    axes[0,1].set_xlabel("Times $[\mu s]$")
    axes[0,1].set_xticks(np.arange(0, 1.6, 0.5))
    axes[0,1].set_title("I")
    axes[0,1].set_yticklabels([])
    axes[0,1].tick_params(axis='y', direction="inout", length=10)

    colors = ['red', 'black', 'blue']
    for i in range(len(delta_rms_trace)):
        axes[0, 0].axhline(-delta_rms_trace[i] / chi, color=colors[i], linestyle=":")
        axes[0, 1].axhline(-delta_rms_trace[i] / chi, color=colors[i], linestyle=":")

        axes[1, 0].plot(t, np.imag(single_trace_data_e[i, :]), color=colors[i],
                        label=f"$\Delta_{{mr}} \\approx$ {-delta_rms_trace[i] / chi}$\chi$")
        axes[1, 1].plot(t, -np.real(single_trace_data_e[i, :]), color=colors[i],
                        label=f"$\Delta_{{mr}} \\approx$ {-delta_rms_trace[i] / chi}$\chi$")

    axes[1, 0].set_xticks(np.arange(0, 1.6, 0.5))
    axes[1, 0].set_xlabel("Times $(\mu s)$")
    axes[1, 0].set_ylabel("Q [a.u.]")
    axes[1, 0].set_ylim(-0.7, 1.4)
    axes[1, 0].legend()

    axes[1, 1].set_xticks(np.arange(0, 1.6, 0.5))
    axes[1, 1].set_xlabel("Times $(\mu s)$")
    axes[1, 1].yaxis.set_label_position("right")
    axes[1, 1].yaxis.tick_right()
    axes[1, 1].set_ylabel("I [a.u.]")
    axes[1, 1].set_ylim(-0.7, 1.4)
    axes[1, 1].legend()

    fig.suptitle("Cavity response to a strong measurement in excited state")

    fig.subplots_adjust(wspace=0)
    plt.show()

    # Fig 5c ----------------------------------

    difference = sweep_data_excited - sweep_data_ground
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1], 'width_ratios': [1, 1]})

    im1 = axes[0,0].imshow(np.imag(difference), cmap='seismic', vmin=-1, vmax=1, aspect='auto', origin='lower',
                         extent=(t[0] - t_step / 2, t[-1] + t_step / 2, (delta_rms[0] - delta_rm_step / 2) / chi,
                                 (delta_rms[-1] + delta_rm_step / 2) / chi), )

    # plt.colorbar(im1, ax=axes[0])
    axes[0,0].set_xlabel("Times $[\mu s]$")
    axes[0,0].set_xticks(np.arange(0, 1.6, 0.5))
    axes[0,0].set_ylabel("Detuning $\Delta_{rm}/\chi$")
    axes[0,0].set_title("Q")

    im2 = axes[0,1].imshow(-np.real(difference), cmap='seismic', vmin=-1, vmax=1, aspect='auto', origin='lower',
                         extent=(t[0] - t_step / 2, t[-1] + t_step / 2, (delta_rms[0] - delta_rm_step / 2) / chi,
                                 (delta_rms[-1] + delta_rm_step / 2) / chi))

    # plt.colorbar(im2, ax=axes[1])
    axes[0,1].set_xlabel("Times $[\mu s]$")
    axes[0,1].set_xticks(np.arange(0, 1.6, 0.5))
    axes[0,1].set_title("I")
    axes[0,1].set_yticklabels([])
    axes[0,1].tick_params(axis='y', direction="inout", length=10)

    colors = ['red', 'black', 'blue']
    for i in range(len(delta_rms_trace)):
        axes[0, 0].axhline(-delta_rms_trace[i] / chi, color=colors[i], linestyle=":")
        axes[0, 1].axhline(-delta_rms_trace[i] / chi, color=colors[i], linestyle=":")

        axes[1, 0].plot(t, np.imag((single_trace_data_e - single_trace_data_g)[i, :]), color=colors[i],
                        label=f"$\Delta_{{mr}} \\approx$ {-delta_rms_trace[i] / chi}$\chi$")
        axes[1, 1].plot(t, -np.real((single_trace_data_e - single_trace_data_g)[i, :]), color=colors[i],
                        label=f"$\Delta_{{mr}} \\approx$ {-delta_rms_trace[i] / chi}$\chi$")

    axes[1, 0].set_xticks(np.arange(0, 1.6, 0.5))
    axes[1, 0].set_xlabel("Times $(\mu s)$")
    axes[1, 0].set_ylabel("Q [a.u.]")
    axes[1, 0].set_ylim(-0.7, 1.4)
    axes[1, 0].legend()

    axes[1, 1].set_xticks(np.arange(0, 1.6, 0.5))
    axes[1, 1].set_xlabel("Times $(\mu s)$")
    axes[1, 1].yaxis.set_label_position("right")
    axes[1, 1].yaxis.tick_right()
    axes[1, 1].set_ylabel("I [a.u.]")
    axes[1, 1].set_ylim(-0.7, 1.4)
    axes[1, 1].legend()

    fig.suptitle("Difference between excited and ground state sweeps")

    fig.subplots_adjust(wspace=0)
    plt.show()


def figure6():
    omega_r = 6.44252 * 2 * np.pi
    omega_a = 4.009 * 2 * np.pi

    chi = -0.00069 * 2 * np.pi

    kappa = 0.00169 * 2 * np.pi
    gamma_1 = 0.00019 * 2 * np.pi

    # gamma_phi not given
    gamma_phi = 2 * gamma_1

    epsilon_m = np.sqrt(kappa / 2) / 10
    omega_m = omega_r - 1 / 2 * chi

    # no qubit drive for getting baseline
    Omega = 0
    # lamb shift
    omega_s = omega_a + chi

    delta_rm = omega_r - omega_m
    delta_as = omega_a - omega_s

    meas_t1 = 0
    measurement_pulse_length = 500  # ns

    args = {
        "epsilon_m": epsilon_m,
        "omega_m": omega_m,
        "Omega": Omega,
        "omega_s": omega_s,
        "meas_t1": meas_t1,
        "meas_t2": measurement_pulse_length
    }

    def H_measurement_drive(t, args):
        epsilon_m = args["epsilon_m"]
        t1 = args["meas_t1"]
        t2 = args["meas_t2"]
        if t1 < t <= t2:
            return epsilon_m
        else:
            return 0

    def H_qubit_drive(t, args):
        Omega = args["Omega"]
        t1 = args["drive_t1"]
        t2 = args["drive_t2"]
        if t1 < t <= t2:
            return Omega
        else:
            return 0

    num_levels = 10
    H = get_Hamiltonian(num_levels, delta_rm, delta_as, chi, epsilon_m, Omega)

    times = np.linspace(0, measurement_pulse_length, 501)

    psi0_g = qt.tensor(qt.basis(num_levels, 0), qt.basis(2, 0))
    psi0_e = qt.tensor(qt.basis(num_levels, 0), qt.basis(2, 1))

    a = qt.tensor(qt.destroy(num_levels), qt.qeye(2))
    sigma_m = qt.tensor(qt.qeye(num_levels), qt.destroy(2))
    sigma_z = qt.tensor(qt.qeye(num_levels), -qt.sigmaz())

    collapse_operators = [
        np.sqrt(kappa) * a,
        np.sqrt(gamma_1) * sigma_m,
        np.sqrt(gamma_phi) * sigma_z,
    ]

    result_g = qt.mesolve(H, psi0_g, times, collapse_operators, [sigma_m.dag() * sigma_m, a.dag() * a, a], args)
    result_e = qt.mesolve(H, psi0_e, times, collapse_operators, [sigma_m.dag() * sigma_m, a.dag() * a, a], args)

    a_expectation_g = result_g.expect[2]
    a_expectation_e = result_e.expect[2]

    # a_expectation_g = rotate_IQ(a_expectation_g)
    # a_expectation_e = rotate_IQ(a_expectation_e)

    I_g = np.real(a_expectation_g)
    Q_g = np.imag(a_expectation_g)

    I_e = np.real(a_expectation_e)
    Q_e = np.imag(a_expectation_e)

    # print(Q_e - Q_g)
    # print(I_e - I_g)
    #
    # plt.plot(result_g.times, Q_e - Q_g, label="Q difference")
    # plt.plot(result_e.times, I_e - I_g, label="I difference")
    #
    # plt.legend()
    # plt.show()

    # unknown state measurement

    # turn qubit drive on
    Omega = 0.025 * 2 * np.pi
    args["Omega"] = Omega

    pulse_lengths = np.linspace(0, 50, 101)
    # pulse_lengths = [0, 10]

    num_points = 601

    sweep_data = np.zeros((len(pulse_lengths), num_points), dtype=complex)

    H = get_Hamiltonian(num_levels, delta_rm, delta_as, chi, epsilon_m, Omega, measurement_drive=H_measurement_drive,
                        qubit_drive=H_qubit_drive)

    for i in range(len(pulse_lengths)):
        pulse_length = pulse_lengths[i]

        times = np.linspace(-100, measurement_pulse_length, num_points)

        args["drive_t1"] = -pulse_length
        args["drive_t2"] = 0

        psi0 = qt.tensor(qt.basis(num_levels, 0), qt.basis(2, 0))

        options = qt.Options(max_step=1)
        result = qt.mesolve(H, psi0, times, collapse_operators, [a, sigma_z], args, options)

        a_expectation = result.expect[0]

        # plt.plot(result.times, result.expect[1], label="sigma_z")
        # plt.plot(result.times, np.imag(result.expect[0]), label="Q")
        # plt.plot(result.times, np.real(result.expect[0]), label="I")
        # plt.title(f"pulse length: {pulse_length}")
        # plt.legend()
        # plt.show()

        # a_expectation = rotate_IQ(a_expectation)

        sweep_data[i, :] = a_expectation

    dt = result.times[1] - result.times[0]

    print(I_g.shape)

    p_e_array = np.zeros((2, len(pulse_lengths)))

    # skip first terms because they're small

    start_index = 100

    I_g = I_g[start_index:]
    Q_g = Q_g[start_index:]

    I_e = I_e[start_index:]
    Q_e = Q_e[start_index:]

    for i in range(len(pulse_lengths)):
        s_rho_I = np.real(sweep_data[i, 100 + start_index:])
        s_rho_Q = np.imag(sweep_data[i, 100 + start_index:])

        p_e_I = np.sum(np.divide((s_rho_I - I_g), (I_e - I_g))) * dt / (measurement_pulse_length - start_index * dt)
        p_e_Q = np.sum(np.divide((s_rho_Q - Q_g), (Q_e - Q_g))) * dt / (measurement_pulse_length  - start_index * dt)

        # plt.plot(s_rho_I - I_g, label="measurement I")
        # plt.plot(I_e - I_g, label="baseline I")
        # # plt.plot((s_rho_I - I_g)/(I_e - I_g), label="ratio I")
        #
        # plt.plot(s_rho_Q - Q_g, label="measurement Q")
        # plt.plot(Q_e - Q_g, label="baseline Q")
        #
        # plt.title(f"pulse length: {pulse_lengths[i]}"
        #           f"")
        # plt.legend()
        # plt.show()

        p_e_array[:, i] = np.array([p_e_Q, p_e_I])

    plt.plot(pulse_lengths, p_e_array[0, :], color='blue', marker='o', linestyle='', label="Q data reconstruction")
    # plt.plot(pulse_lengths, p_e_array[1, :], label="I")

    # get expected sigma_z behavior
    H = get_Hamiltonian(num_levels, delta_rm, delta_as, chi, 0, Omega)
    result_expected = qt.mesolve(H, psi0_g, pulse_lengths, collapse_operators, [1/2*(1+sigma_z)])

    plt.plot(pulse_lengths, result_expected.expect[0], color='black', linestyle=':', label="Expected Rabi oscillation")

    plt.ylabel("$P_e$")
    plt.xlabel("Pulse length (ns)")

    plt.title("Rabi oscillation reconstruction using integrated Q data")

    plt.legend()
    plt.show()

    # print(sweep_data)
    # print(result.times[100:])
    # print(sweep_data.shape)




if __name__ == "__main__":
    # figure2bc()
    # figure2de()
    # figure3()
    # figure4()
    # figure5()
    figure6()
