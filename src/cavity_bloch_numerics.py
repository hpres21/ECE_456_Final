import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from cavity_bloch_util import cavity_bloch_equations_resonant_time_drive_short_pulse, \
    cavity_bloch_equations_resonant_time_drive_pulsed_measurement, \
    cavity_bloch_equations_resonant_time_drive_variable_pulse_length, \
    cavity_bloch_equations_resonant_time_drive, rotate_IQ



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


    # # plt.plot(result.t, sigmaz, label="sigma_z")
    # # plt.plot(result.t, adagger_a, label="adag a")
    # plt.plot(result.t, np.real(a), label="I")
    # plt.plot(result.t, np.imag(a), label="Q")
    #
    # # plt.plot(result.t, sigmax, label="sigmax")
    # # plt.plot(result.t, sigmay, label="sigmay")
    # # plt.plot(result.t, np.imag(a_sigmax), label="imag(a_sigmax)")
    # # plt.plot(result.t, np.imag(a_sigmay), label="imag(a_sigmay)")
    # # plt.plot(result.t, np.imag(a_sigmaz), label="imag(a_sigmaz)")
    #
    # plt.xlabel("t (ns)")
    # plt.ylabel("Populations")
    #
    # plt.title("Pi Pulse with decay")
    #
    # plt.legend()
    # plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(result.t/1000, -4-np.imag(a), color='black')
    axes[0].set_ylim(0, 10)

    axes[0].set_xlabel("Times $[\mu s]$")
    axes[0].set_xticks(np.arange(0, 1.6, 0.5))
    axes[0].set_ylabel("Q [a.u.]")

    axes[1].plot(result.t/1000, np.real(a), color='black')

    axes[1].set_xlabel("Times $[\mu s]$")
    axes[1].set_xticks(np.arange(0, 1.6, 0.5))
    axes[1].set_ylim(0, 10)

    # axes[1].set_yticklabels([])
    # axes[1].tick_params(axis='y', direction="inout", length=10)

    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.tick_right()
    axes[1].set_ylabel("I [a.u.]")

    fig.suptitle("Cavity response to a weak measurement for $\Delta_{mr} = -\chi$")

    fig.subplots_adjust(wspace=0)
    plt.show()



def figure2de(*args):
    omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi = args

    epsilon_m = np.sqrt(kappa / 2)

    omega_s = omega_a + 3 * chi
    delta_as = omega_a - omega_s

    Omega = 0.05 * 2 * np.pi

    num_points = 1001
    tspan = (-500, 2000)
    times = np.linspace(*tspan, num_points)

    delta_rms = np.linspace(-5 * chi, 4 * chi, 50)

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
    delta_rms *= -1

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(np.abs(np.imag(sweep_data)), cmap='Reds', vmin=-1, vmax=16, aspect='auto', origin='lower',
                   extent=(t[0] - t_step / 2, t[-1] + t_step / 2, (delta_rms[0] - delta_rm_step / 2) / chi,
                           (delta_rms[-1] + delta_rm_step / 2) / chi), )
    # plt.colorbar()
    axes[0].set_xlabel("Times $[\mu s]$")
    axes[0].set_xticks(np.arange(0, 1.6, 0.5))
    axes[0].set_ylabel("Detuning $\Delta_{mr}/\chi$")
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

    fig.suptitle("Cavity response to a weak measurement")

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

    delta_rms = np.linspace(-5 * chi, 4 * chi, 100)
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
        axes[0].plot(delta_rms / chi, -Q_data[i, :], color=colors[i], label=labels[i])
        axes[1].plot(delta_rms / chi, I_data[i, :], color=colors[i], label=labels[i])

    axes[0].set_xlabel("Detuning $-\Delta_{mr}/\chi$")
    axes[0].set_ylabel("Q [a.u.]")
    axes[0].legend()

    axes[1].set_xlabel("Detuning $-\Delta_{mr}/\chi$")
    axes[1].set_ylabel("I [a.u.]")
    axes[1].legend()

    fig.suptitle("Resonator Transmission Spectrum")

    plt.tight_layout()
    plt.show()


def figure4(*args):
    omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi = args

    epsilon_m = np.sqrt(kappa / 2)
    omega_m = omega_r - chi
    # omega_m = omega_r + chi

    omega_s = omega_a + chi
    delta_as = omega_a - omega_s
    delta_rm = omega_r - omega_m
    Omega = 0

    # initial variables
    alpha = 1j

    a0 = 0j
    sigmaz0 = 1
    sigmax0 = 0
    sigmay0 = 0
    a_sigmaz0 = a0 * sigmaz0
    a_sigmax0 = a0 * sigmax0
    a_sigmay0 = a0 * sigmay0
    adagger_a0 = 0

    y0 = np.array([a0, sigmaz0, sigmay0, sigmax0, a_sigmaz0, a_sigmay0, a_sigmax0, adagger_a0])
    y1 = np.array([a0, -1, sigmay0, sigmax0, a_sigmaz0, a_sigmay0, a_sigmax0, adagger_a0])

    args = delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi

    num_points = 2000
    tspan = (-100, 2000)
    times = np.linspace(*tspan, num_points)

    result = solve_ivp(cavity_bloch_equations_resonant_time_drive_pulsed_measurement, tspan, y0, t_eval=times,
                       args=args, max_step=1)
    result2 = solve_ivp(cavity_bloch_equations_resonant_time_drive_pulsed_measurement, tspan, y1, t_eval=times,
                        args=args, max_step=1)
    y = result.y
    y2 = result2.y

    a = y[0, :]
    sigmaz = np.real(y[1, :])
    sigmax = np.real(y[2, :])
    sigmay = np.real(y[3, :])
    a_sigmaz = y[4, :]
    a_sigmax = y[5, :]
    a_sigmay = y[6, :]
    adagger_a = np.real(y[7, :])
    new_a = rotate_IQ(a)

    a2 = y2[0, :]
    sigmaz2 = np.real(y2[1, :])
    sigmax2 = np.real(y2[2, :])
    sigmay2 = np.real(y2[3, :])
    a_sigmaz2 = y2[4, :]
    a_sigmax2 = y2[5, :]
    a_sigmay2 = y2[6, :]
    adagger_a2 = np.real(y2[7, :])
    new_a2 = rotate_IQ(a2)

    fig, ax = plt.subplots(3, 1, figsize=(6, 8))
    ax[0].plot(result.t / 1000, np.abs(np.imag(new_a)), label="|e>", c='red')
    ax[0].plot(result.t / 1000, np.abs(np.imag(new_a2)), label="|g>", c='blue')
    ax[0].set_xlabel("Time (us)")
    ax[0].set_ylabel("Q [a.u.]")
    ax[0].legend()

    ax[1].plot(result.t / 1000, np.abs(np.real(new_a)), label='|e>', c='red')
    ax[1].plot(result.t / 1000, np.abs(np.real(new_a2)), label='|g>', c='blue')
    ax[1].legend()
    ax[1].set_xlabel("Time (us)")
    ax[1].set_ylabel("I [a.u.]")
    ax[1].set_ylim(0, 14)

    ax[2].scatter(np.abs(np.imag(new_a)[::5]), np.abs(np.real(new_a)[::5]), label='|e>', marker='o', c='red')
    ax[2].scatter(np.abs(np.imag(new_a2)[::5]), np.abs(np.real(new_a2)[::5]), label='|g>', marker='x', c='blue')
    ax[2].legend()
    ax[2].set_xlabel('Q quadrature [a.u.]')
    ax[2].set_ylabel('I quadrature [a.u.]')

    fig.suptitle("Cavity Response for a Pulsed Measurement")

    plt.tight_layout()
    plt.show()


def figure5(*args):
    # for this experiment

    omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi = args

    epsilon_m = np.sqrt(kappa / 2)
    omega_m = omega_r - chi

    omega_s = omega_a + chi
    delta_as = omega_a - omega_s
    delta_rm = omega_r - omega_m

    Omega = 0.05 * 2 * np.pi * 0

    # epsilon_m = 0
    # gamma_1 = 0
    # gamma_phi = 0

    # initial variables
    a0 = 0j
    sigmaz0 = -1
    sigmax0 = 0
    sigmay0 = 0
    a_sigmaz0 = a0 * sigmaz0
    a_sigmax0 = a0 * sigmax0
    a_sigmay0 = a0 * sigmay0
    adagger_a0 = 0

    y0_ground = np.array([a0, -1, sigmay0, sigmax0, a_sigmaz0, a_sigmay0, a_sigmax0, adagger_a0])
    y0_excited = np.array([a0, 1, sigmay0, sigmax0, a_sigmaz0, a_sigmay0, a_sigmax0, adagger_a0])

    num_points = 1000
    tspan = (-250, 1720)
    times = np.linspace(*tspan, num_points)

    delta_rms = np.linspace(-5 * chi, 4 * chi, 100)

    sweep_data_ground = np.zeros((len(delta_rms), num_points), dtype=complex)
    sweep_data_excited = np.zeros((len(delta_rms), num_points), dtype=complex)

    for i in range(len(delta_rms)):
        delta_rm = delta_rms[i]

        args = delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi

        result_ground = solve_ivp(cavity_bloch_equations_resonant_time_drive_pulsed_measurement, tspan, y0_ground, t_eval=times,
                           args=args, max_step=1)

        result_excited = solve_ivp(cavity_bloch_equations_resonant_time_drive_pulsed_measurement, tspan, y0_excited,
                                  t_eval=times,
                                  args=args, max_step=1)

        y_ground = result_ground.y
        y_excited = result_excited.y

        a_ground = y_ground[0, :]
        a_excited = y_excited[0, :]

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

        args = delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi

        result_ground = solve_ivp(cavity_bloch_equations_resonant_time_drive_pulsed_measurement, tspan, y0_ground,
                                  t_eval=times,
                                  args=args, max_step=1)

        result_excited = solve_ivp(cavity_bloch_equations_resonant_time_drive_pulsed_measurement, tspan, y0_excited,
                                   t_eval=times,
                                   args=args, max_step=1)

        y_ground = result_ground.y
        y_excited = result_excited.y

        a_ground = y_ground[0, :]
        a_excited = y_excited[0, :]

        a_ground = rotate_IQ(a_ground)
        a_excited = rotate_IQ(a_excited)

        single_trace_data_g[i, :] = a_ground
        single_trace_data_e[i, :] = a_excited

    t = np.copy(result_ground.t) / 1000
    t_step = t[1] - t[0]
    delta_rm_step = (delta_rms[-1] - delta_rms[0]) / (len(delta_rms) - 1)


    # Fig 5a ----------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1], 'width_ratios': [1, 1]})

    im1 = axes[0, 0].imshow(np.imag(sweep_data_ground), cmap='seismic', vmin=-30, vmax=30, aspect='auto',
                            origin='lower',
                            extent=(t[0] - t_step / 2, t[-1] + t_step / 2, (delta_rms[0] - delta_rm_step / 2) / chi,
                                    (delta_rms[-1] + delta_rm_step / 2) / chi), )
    # plt.colorbar(im1, ax=axes[0])
    axes[0, 0].set_xlabel("Times $(\mu s)$")
    axes[0, 0].set_xticks(np.arange(0, 1.6, 0.5))
    axes[0, 0].set_ylabel("Detuning $-\Delta_{rm}/\chi$")
    axes[0, 0].set_title("Q")

    im2 = axes[0, 1].imshow(-np.real(sweep_data_ground), cmap='seismic', vmin=-10, vmax=10, aspect='auto',
                            origin='lower',
                            extent=(t[0] - t_step / 2, t[-1] + t_step / 2, (delta_rms[0] - delta_rm_step / 2) / chi,
                                    (delta_rms[-1] + delta_rm_step / 2) / chi))

    # plt.colorbar(im2, ax=axes[1])
    axes[0, 1].set_xlabel("Times $(\mu s)$")
    axes[0, 1].set_xticks(np.arange(0, 1.6, 0.5))
    axes[0, 1].set_title("I")
    axes[0, 1].set_yticklabels([])
    axes[0, 1].tick_params(axis='y', direction="inout", length=10)

    colors = ['red', 'black', 'blue']
    for i in range(len(delta_rms_trace)):
        axes[0, 0].axhline(delta_rms_trace[i] / chi, color=colors[i], linestyle=":")
        axes[0, 1].axhline(delta_rms_trace[i] / chi, color=colors[i], linestyle=":")

        axes[1, 0].plot(t, np.imag(single_trace_data_g[i, :]), color=colors[i],
                        label=f"$\Delta_{{mr}} \\approx$ {-delta_rms_trace[i] / chi}$\chi$")
        axes[1, 1].plot(t, -np.real(single_trace_data_g[i, :]), color=colors[i],
                        label=f"$\Delta_{{mr}} \\approx$ {-delta_rms_trace[i] / chi}$\chi$")

    axes[1, 0].set_xticks(np.arange(0, 1.6, 0.5))
    axes[1, 0].set_xlabel("Times $(\mu s)$")
    axes[1, 0].set_ylabel("Q [a.u.]")
    axes[1, 0].set_ylim(-7, 14)
    axes[1, 0].legend()

    axes[1, 1].set_xticks(np.arange(0, 1.6, 0.5))
    axes[1, 1].set_xlabel("Times $(\mu s)$")
    axes[1, 1].yaxis.set_label_position("right")
    axes[1, 1].yaxis.tick_right()
    axes[1, 1].set_ylabel("I [a.u.]")
    axes[1, 1].set_ylim(-7, 14)
    axes[1, 1].legend()

    fig.suptitle("Cavity response to a strong measurement in ground state")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0)
    plt.show()

    # Fig 5b ----------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1], 'width_ratios': [1, 1]})

    im1 = axes[0, 0].imshow(np.imag(sweep_data_excited), cmap='seismic', vmin=-30, vmax=30, aspect='auto',
                            origin='lower',
                            extent=(t[0] - t_step / 2, t[-1] + t_step / 2, (delta_rms[0] - delta_rm_step / 2) / chi,
                                    (delta_rms[-1] + delta_rm_step / 2) / chi), )

    # plt.colorbar(im1, ax=axes[0])
    axes[0, 0].set_xlabel("Times $[\mu s]$")
    axes[0, 0].set_xticks(np.arange(0, 1.6, 0.5))
    axes[0, 0].set_ylabel("Detuning $-\Delta_{rm}/\chi$")
    axes[0, 0].set_title("Q")

    im2 = axes[0, 1].imshow(-np.real(sweep_data_excited), cmap='seismic', vmin=-12, vmax=12, aspect='auto',
                            origin='lower',
                            extent=(t[0] - t_step / 2, t[-1] + t_step / 2, (delta_rms[0] - delta_rm_step / 2) / chi,
                                    (delta_rms[-1] + delta_rm_step / 2) / chi))

    # plt.colorbar(im2, ax=axes[1])
    axes[0, 1].set_xlabel("Times $[\mu s]$")
    axes[0, 1].set_xticks(np.arange(0, 1.6, 0.5))
    axes[0, 1].set_title("I")
    axes[0, 1].set_yticklabels([])
    axes[0, 1].tick_params(axis='y', direction="inout", length=10)

    colors = ['red', 'black', 'blue']
    for i in range(len(delta_rms_trace)):
        axes[0, 0].axhline(delta_rms_trace[i] / chi, color=colors[i], linestyle=":")
        axes[0, 1].axhline(delta_rms_trace[i] / chi, color=colors[i], linestyle=":")

        axes[1, 0].plot(t, np.imag(single_trace_data_e[i, :]), color=colors[i],
                        label=f"$\Delta_{{mr}} \\approx$ {-delta_rms_trace[i] / chi}$\chi$")
        axes[1, 1].plot(t, -np.real(single_trace_data_e[i, :]), color=colors[i],
                        label=f"$\Delta_{{mr}} \\approx$ {-delta_rms_trace[i] / chi}$\chi$")

    axes[1, 0].set_xticks(np.arange(0, 1.6, 0.5))
    axes[1, 0].set_xlabel("Times $(\mu s)$")
    axes[1, 0].set_ylabel("Q [a.u.]")
    axes[1, 0].set_ylim(-7, 14)
    axes[1, 0].legend()

    axes[1, 1].set_xticks(np.arange(0, 1.6, 0.5))
    axes[1, 1].set_xlabel("Times $(\mu s)$")
    axes[1, 1].yaxis.set_label_position("right")
    axes[1, 1].yaxis.tick_right()
    axes[1, 1].set_ylabel("I [a.u.]")
    axes[1, 1].set_ylim(-7, 14)
    axes[1, 1].legend()

    fig.suptitle("Cavity response to a strong measurement in excited state")

    fig.subplots_adjust(wspace=0)
    plt.show()

    # Fig 5c ----------------------------------

    difference = sweep_data_excited - sweep_data_ground
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1], 'width_ratios': [1, 1]})

    im1 = axes[0, 0].imshow(np.imag(difference), cmap='seismic', vmin=-10, vmax=10, aspect='auto', origin='lower',
                            extent=(t[0] - t_step / 2, t[-1] + t_step / 2, (delta_rms[0] - delta_rm_step / 2) / chi,
                                    (delta_rms[-1] + delta_rm_step / 2) / chi), )

    # plt.colorbar(im1, ax=axes[0])
    axes[0, 0].set_xlabel("Times $[\mu s]$")
    axes[0, 0].set_xticks(np.arange(0, 1.6, 0.5))
    axes[0, 0].set_ylabel("Detuning $-\Delta_{rm}/\chi$")
    axes[0, 0].set_title("Q")

    im2 = axes[0, 1].imshow(-np.real(difference), cmap='seismic', vmin=-10, vmax=10, aspect='auto', origin='lower',
                            extent=(t[0] - t_step / 2, t[-1] + t_step / 2, (delta_rms[0] - delta_rm_step / 2) / chi,
                                    (delta_rms[-1] + delta_rm_step / 2) / chi))

    # plt.colorbar(im2, ax=axes[1])
    axes[0, 1].set_xlabel("Times $[\mu s]$")
    axes[0, 1].set_xticks(np.arange(0, 1.6, 0.5))
    axes[0, 1].set_title("I")
    axes[0, 1].set_yticklabels([])
    axes[0, 1].tick_params(axis='y', direction="inout", length=10)

    colors = ['red', 'black', 'blue']
    for i in range(len(delta_rms_trace)):
        axes[0, 0].axhline(delta_rms_trace[i] / chi, color=colors[i], linestyle=":")
        axes[0, 1].axhline(delta_rms_trace[i] / chi, color=colors[i], linestyle=":")

        axes[1, 0].plot(t, np.imag((single_trace_data_e - single_trace_data_g)[i, :]), color=colors[i],
                        label=f"$\Delta_{{mr}} \\approx$ {-delta_rms_trace[i] / chi}$\chi$")
        axes[1, 1].plot(t, -np.real((single_trace_data_e - single_trace_data_g)[i, :]), color=colors[i],
                        label=f"$\Delta_{{mr}} \\approx$ {-delta_rms_trace[i] / chi}$\chi$")

    axes[1, 0].set_xticks(np.arange(0, 1.6, 0.5))
    axes[1, 0].set_xlabel("Times $(\mu s)$")
    axes[1, 0].set_ylabel("Q [a.u.]")
    axes[1, 0].set_ylim(-7, 14)
    axes[1, 0].legend()

    axes[1, 1].set_xticks(np.arange(0, 1.6, 0.5))
    axes[1, 1].set_xlabel("Times $(\mu s)$")
    axes[1, 1].yaxis.set_label_position("right")
    axes[1, 1].yaxis.tick_right()
    axes[1, 1].set_ylabel("I [a.u.]")
    axes[1, 1].set_ylim(-7, 14)
    axes[1, 1].legend()

    fig.suptitle("Difference between excited and ground state sweeps")

    fig.subplots_adjust(wspace=0)
    plt.show()

def figure6(*args):
    omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi = args

    epsilon_m = np.sqrt(kappa / 2)
    omega_m = omega_r - chi

    omega_s = omega_a + chi
    delta_as = omega_a - omega_s
    delta_rm = omega_r - omega_m
    Omega = 0

    # initial variables
    a0 = 0j
    sigmaz0 = 1
    sigmax0 = 0
    sigmay0 = 0
    a_sigmaz0 = a0 * sigmaz0
    a_sigmax0 = a0 * sigmax0
    a_sigmay0 = a0 * sigmay0
    adagger_a0 = 0

    y0_g = np.array([a0, -1, sigmay0, sigmax0, a_sigmaz0, a_sigmay0, a_sigmax0, adagger_a0])
    y0_e = np.array([a0, 1, sigmay0, sigmax0, a_sigmaz0, a_sigmay0, a_sigmax0, adagger_a0])

    args = delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi

    measurement_pulse_length = 500
    num_points = 501
    tspan = (0, measurement_pulse_length)
    times = np.linspace(*tspan, num_points)

    result_g = solve_ivp(cavity_bloch_equations_resonant_time_drive_pulsed_measurement, tspan, y0_g, t_eval=times,
                       args=args, max_step=1)
    result_e = solve_ivp(cavity_bloch_equations_resonant_time_drive_pulsed_measurement, tspan, y0_e, t_eval=times,
                        args=args, max_step=1)
    y_g = result_g.y
    y_e = result_e.y

    a_g = y_g[0, :]
    a_e = y_e[0, :]

    I_g = np.real(a_g)
    Q_g = np.imag(a_g)

    I_e = np.real(a_e)
    Q_e = np.imag(a_e)

    # turn qubit drive on
    Omega = 0.05 * 2 * np.pi

    pulse_lengths = np.linspace(0, 50, 101)
    # pulse_lengths = [0, 10]

    num_points = 501

    sweep_data = np.zeros((len(pulse_lengths), num_points), dtype=complex)

    tspan = (-100, measurement_pulse_length)

    for i in range(len(pulse_lengths)):
        pulse_length = pulse_lengths[i]

        t_eval = np.linspace(0, measurement_pulse_length, num_points)

        args = delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi, pulse_length

        result = solve_ivp(cavity_bloch_equations_resonant_time_drive_variable_pulse_length, tspan, y0_g, t_eval=t_eval,
                             args=args, max_step=1)

        y = result.y
        a = y[0, :]
        sigma_z = y[1, :]

        # plt.plot(result.t, sigma_z, label="sigma_z")
        # plt.plot(result.t, np.imag(a), label="Q")
        # plt.plot(result.t, np.real(a), label="I")
        # plt.title(f"pulse length: {pulse_length}")
        # plt.legend()
        # plt.show()

        sweep_data[i, :] = a

    dt = result.t[1] - result.t[0]

    print(I_g.shape)

    p_e_array = np.zeros((2, len(pulse_lengths)))

    # skip first terms because they're small

    start_index = 100

    I_g = I_g[start_index:]
    Q_g = Q_g[start_index:]

    I_e = I_e[start_index:]
    Q_e = Q_e[start_index:]

    for i in range(len(pulse_lengths)):
        s_rho_I = np.real(sweep_data[i, start_index:])
        s_rho_Q = np.imag(sweep_data[i, start_index:])

        p_e_I = np.sum(np.divide((s_rho_I - I_g), (I_e - I_g))) * dt / (measurement_pulse_length - start_index*dt)
        p_e_Q = np.sum(np.divide((s_rho_Q - Q_g), (Q_e - Q_g))) * dt / (measurement_pulse_length - start_index*dt)

        p_e_array[:, i] = np.array([p_e_Q, p_e_I])

    plt.plot(pulse_lengths, p_e_array[0, :], color='blue', marker='o', linestyle='', label="Q data reconstruction")
    # plt.plot(pulse_lengths, p_e_array[1, :], label="I")

    # get expected sigma_z behavior

    # initial variables
    a0 = 0j
    sigmaz0 = -1
    sigmax0 = 0
    sigmay0 = 0
    a_sigmaz0 = a0 * sigmaz0
    a_sigmax0 = a0 * sigmax0
    a_sigmay0 = a0 * sigmay0
    adagger_a0 = 0

    epsilon_m = 0
    args = delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi

    tspan = (pulse_lengths[0], pulse_lengths[-1])
    result = solve_ivp(cavity_bloch_equations_resonant_time_drive, tspan, y0_g, t_eval=pulse_lengths,
                       args=args)
    plt.plot(pulse_lengths, np.real(1/2*(1+result.y[1])), color='black', linestyle=':', label="Expected Rabi oscillation")

    plt.ylabel("$P_e$")
    plt.xlabel("Pulse length (ns)")

    plt.title("Rabi oscillation reconstruction using integrated Q data")

    plt.legend()
    plt.show()


def show_steady_state(*args):
    omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi = args

    epsilon_m = np.sqrt(kappa / 2) / 10
    omega_m = omega_r - chi

    omega_s = omega_a + 3 * chi
    delta_as = omega_a - omega_s
    delta_rm = omega_r - omega_m

    Omega = 0

    args = delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi

    # initial variables

    # vacuum state
    a0 = 0j
    sigmaz0 = -1
    sigmax0 = 0
    sigmay0 = 0
    a_sigmaz0 = a0 * sigmaz0
    a_sigmax0 = a0 * sigmax0
    a_sigmay0 = a0 * sigmay0
    adagger_a0 = 0

    y0 = np.array([a0, sigmaz0, sigmay0, sigmax0, a_sigmaz0, a_sigmay0, a_sigmax0, adagger_a0])

    num_points = 2000
    tspan = (0, 2000)
    times = np.linspace(*tspan, num_points)

    result_vacuum = solve_ivp(cavity_bloch_equations_resonant_time_drive, tspan, y0, t_eval=times, args=args)

    y = result_vacuum.y

    a = y[0, :]
    sigmaz = np.real(y[1, :])
    sigmax = np.real(y[2, :])
    sigmay = np.real(y[3, :])
    a_sigmaz = y[4, :]
    a_sigmax = y[5, :]
    a_sigmay = y[6, :]
    adagger_a = np.real(y[7, :])

    plt.plot(result_vacuum.t, sigmaz, label="Qubit population")
    plt.plot(result_vacuum.t, adagger_a, label="Cavity population")
    # plt.plot(result_vacuum.t, np.real(a), label="I")
    # plt.plot(result_vacuum.t, np.imag(a), label="Q")
    plt.axhline(10, color='black', linestyle=':', label='Number of cavity levels simulated in QuTiP')


    plt.xlabel("t (ns)")
    plt.ylabel("Populations")

    plt.title("Steady state under constant measurement drive")

    plt.legend()
    plt.show()

    # initial variables

    args = delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi

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

    result_ss = solve_ivp(cavity_bloch_equations_resonant_time_drive_short_pulse, tspan, y0, t_eval=times, args=args,
                       max_step=1)

    y = result_ss.y

    a = y[0, :]
    sigmaz = np.real(y[1, :])
    sigmax = np.real(y[2, :])
    sigmay = np.real(y[3, :])
    a_sigmaz = y[4, :]
    a_sigmax = y[5, :]
    a_sigmay = y[6, :]
    adagger_a = np.real(y[7, :])

    plt.plot(result_vacuum.t, sigmaz, label="Qubit population")
    plt.plot(result_vacuum.t, adagger_a, label="Cavity population")
    plt.plot(result_vacuum.t, np.real(a), label="I")
    plt.plot(result_vacuum.t, np.imag(a), label="Q")


    plt.xlabel("t (ns)")
    plt.ylabel("Populations")

    plt.title("System steady state")

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

    figure2bc(omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi)
    # figure2de(omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi)
    # figure3(omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi)
    # figure4(omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi)
    # figure5(omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi)
    # figure6(omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi)
    # show_steady_state(omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi)


if __name__ == "__main__":
    cavity_bloch_numerical()
