import numpy as np
from scipy.integrate import solve_ivp

def rotate_IQ(a):
    # want last element of Q to be fully imaginary
    last = a[-1]
    return a * 1j * np.conjugate(last) / np.abs(last)
def cavity_bloch_equations(t, y, *args):
    """

    Constant qubit and measurement drives

    :param y:
    :param t:
    :param args: (delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi)
    :return:
    """

    delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi = args

    a = y[0]
    sigmaz = y[1]
    sigmax = y[2]
    sigmay = y[3]
    a_sigmaz = y[4]
    a_sigmax = y[5]
    a_sigmay = y[6]
    adagger_a = y[7]

    ydot = np.array([0j] * len(y))

    ydot[0] = -1j * delta_rm * a - 1j * chi * a_sigmaz - 1j * epsilon_m - kappa / 2 * a
    ydot[1] = Omega * sigmay - gamma_1 * (1 + sigmaz)
    ydot[2] = -(delta_as + 2 * chi * (adagger_a + 1 / 2)) * sigmay - (gamma_1 / 2 + gamma_phi) * sigmax
    ydot[3] = (delta_as + 2 * chi * (adagger_a + 1 / 2)) * sigmax - (gamma_1 / 2 + gamma_phi) * sigmay - Omega * sigmaz
    ydot[
        4] = -1j * delta_rm * a_sigmaz - 1j * chi * a + Omega * a_sigmay - 1j * epsilon_m * sigmaz - gamma_1 * a - (
            gamma_1 + kappa / 2) * a_sigmaz
    ydot[5] = -1j * delta_rm * a_sigmax - (
            delta_as + 2 * chi * (adagger_a + 1)) * a_sigmay - 1j * epsilon_m * sigmax - (
                      gamma_1 / 2 + gamma_phi + kappa / 2) * a_sigmax
    ydot[6] = -1j * delta_rm * a_sigmay + (
            delta_as + 2 * chi * (adagger_a + 1)) * a_sigmax - 1j * epsilon_m * sigmay - (
                      gamma_1 / 2 + gamma_phi + kappa / 2) * a_sigmay - Omega * a_sigmaz
    ydot[7] = -2 * epsilon_m * np.imag(a) - kappa * adagger_a

    return ydot


def cavity_bloch_equations_short_pulse(t, y, *args):
    """

    Short qubit control pulse between t1 and t2

    :param y:
    :param t:
    :param args: (delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi)
    :return:
    """

    delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi = args

    t1 = 100
    t2 = t1 + 10
    if not t1 < t < t2:
        Omega = 0

    a = y[0]
    sigmaz = y[1]
    sigmax = y[2]
    sigmay = y[3]
    a_sigmaz = y[4]
    a_sigmax = y[5]
    a_sigmay = y[6]
    adagger_a = y[7]

    ydot = np.array([0j] * len(y))

    ydot[0] = -1j * delta_rm * a - 1j * chi * a_sigmaz - 1j * epsilon_m - kappa / 2 * a
    ydot[1] = Omega * sigmay - gamma_1 * (1 + sigmaz)
    ydot[2] = -(delta_as + 2 * chi * (adagger_a + 1 / 2)) * sigmay - (gamma_1 / 2 + gamma_phi) * sigmax
    ydot[3] = (delta_as + 2 * chi * (adagger_a + 1 / 2)) * sigmax - (gamma_1 / 2 + gamma_phi) * sigmay - Omega * sigmaz
    ydot[
        4] = -1j * delta_rm * a_sigmaz - 1j * chi * a + Omega * a_sigmay - 1j * epsilon_m * sigmaz - gamma_1 * a - (
            gamma_1 + kappa / 2) * a_sigmaz
    ydot[5] = -1j * delta_rm * a_sigmax - (
            delta_as + 2 * chi * (adagger_a + 1)) * a_sigmay - 1j * epsilon_m * sigmax - (
                      gamma_1 / 2 + gamma_phi + kappa / 2) * a_sigmax
    ydot[6] = -1j * delta_rm * a_sigmay + (
            delta_as + 2 * chi * (adagger_a + 1)) * a_sigmax - 1j * epsilon_m * sigmay - (
                      gamma_1 / 2 + gamma_phi + kappa / 2) * a_sigmay - Omega * a_sigmaz
    ydot[7] = -2 * epsilon_m * np.imag(a) - kappa * adagger_a

    return ydot


def cavity_bloch_equations_resonant_time_drive(t, y, *args):
    """

    Uses delta_as = - 2*chi*(adagger_a + 1/2) with constant drives

    :param y:
    :param t:
    :param args: (delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi)
    :return:
    """

    delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi = args

    a = y[0]
    sigmaz = y[1]
    sigmax = y[2]
    sigmay = y[3]
    a_sigmaz = y[4]
    a_sigmax = y[5]
    a_sigmay = y[6]
    adagger_a = y[7]

    ydot = np.array([0j] * len(y))

    ydot[0] = -1j * delta_rm * a - 1j * chi * a_sigmaz - 1j * epsilon_m - kappa / 2 * a
    ydot[1] = Omega * sigmay - gamma_1 * (1 + sigmaz)
    ydot[2] = - (gamma_1 / 2 + gamma_phi) * sigmax
    ydot[3] = - (gamma_1 / 2 + gamma_phi) * sigmay - Omega * sigmaz
    ydot[
        4] = -1j * delta_rm * a_sigmaz - 1j * chi * a + Omega * a_sigmay - 1j * epsilon_m * sigmaz - gamma_1 * a - (
            gamma_1 + kappa / 2) * a_sigmaz
    ydot[5] = -1j * delta_rm * a_sigmax - chi * a_sigmay - 1j * epsilon_m * sigmax - (
            gamma_1 / 2 + gamma_phi + kappa / 2) * a_sigmax
    ydot[6] = -1j * delta_rm * a_sigmay + chi * a_sigmax - 1j * epsilon_m * sigmay - (
            gamma_1 / 2 + gamma_phi + kappa / 2) * a_sigmay - Omega * a_sigmaz
    ydot[7] = -2 * epsilon_m * np.imag(a) - kappa * adagger_a

    return ydot


def cavity_bloch_equations_resonant_time_drive_short_pulse(t, y, *args):
    """

    uses delta_as = - 2*chi*(adagger_a + 1/2)
    Pi pulse only happens between t1 and t2

    :param y:
    :param t:
    :param args: (delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi)
    :return:
    """

    delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi = args

    t1 = 100
    t2 = t1 + 10
    if not t1 < t < t2:
        Omega = 0

    a = y[0]
    sigmaz = y[1]
    sigmax = y[2]
    sigmay = y[3]
    a_sigmaz = y[4]
    a_sigmax = y[5]
    a_sigmay = y[6]
    adagger_a = y[7]

    ydot = np.array([0j] * len(y))

    ydot[0] = -1j * delta_rm * a - 1j * chi * a_sigmaz - 1j * epsilon_m - kappa / 2 * a
    ydot[1] = Omega * sigmay - gamma_1 * (1 + sigmaz)
    ydot[2] = - (gamma_1 / 2 + gamma_phi) * sigmax
    ydot[3] = - (gamma_1 / 2 + gamma_phi) * sigmay - Omega * sigmaz
    ydot[
        4] = -1j * delta_rm * a_sigmaz - 1j * chi * a + Omega * a_sigmay - 1j * epsilon_m * sigmaz - gamma_1 * a - (
            gamma_1 + kappa / 2) * a_sigmaz
    ydot[5] = -1j * delta_rm * a_sigmax - chi * a_sigmay - 1j * epsilon_m * sigmax - (
            gamma_1 / 2 + gamma_phi + kappa / 2) * a_sigmax
    ydot[6] = -1j * delta_rm * a_sigmay + chi * a_sigmax - 1j * epsilon_m * sigmay - (
            gamma_1 / 2 + gamma_phi + kappa / 2) * a_sigmay - Omega * a_sigmaz
    ydot[7] = -2 * epsilon_m * np.imag(a) - kappa * adagger_a

    # print(f"Omega: {Omega}")
    # print(f"sigma y0; {sigmay}")
    # print(ydot)

    return ydot


def cavity_bloch_equations_resonant_time_drive_pulsed_measurement(t, y, *args):
    """

    uses delta_as = - 2*chi*(adagger_a + 1/2)
    Pi pulse only happens between t1 and t2
    Measurement drive starts at t2

    :param y:
    :param t:
    :param args: (delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi)
    :return:
    """

    delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi = args

    t1 = -10
    t2 = t1 + 10
    # qubit control drive on in between t1 and t2
    if not t1 < t < t2:
        Omega = 0

    # resonator measurement pulse on after t2
    if t <= t2:
        epsilon_m = 0

    a = y[0]
    sigmaz = y[1]
    sigmax = y[2]
    sigmay = y[3]
    a_sigmaz = y[4]
    a_sigmax = y[5]
    a_sigmay = y[6]
    adagger_a = y[7]

    ydot = np.array([0j] * len(y))

    ydot[0] = -1j * delta_rm * a - 1j * chi * a_sigmaz - 1j * epsilon_m - kappa / 2 * a
    ydot[1] = Omega * sigmay - gamma_1 * (1 + sigmaz)
    ydot[2] = - (gamma_1 / 2 + gamma_phi) * sigmax
    ydot[3] = - (gamma_1 / 2 + gamma_phi) * sigmay - Omega * sigmaz
    ydot[
        4] = -1j * delta_rm * a_sigmaz - 1j * chi * a + Omega * a_sigmay - 1j * epsilon_m * sigmaz - gamma_1 * a - (
            gamma_1 + kappa / 2) * a_sigmaz
    ydot[5] = -1j * delta_rm * a_sigmax - chi * a_sigmay - 1j * epsilon_m * sigmax - (
            gamma_1 / 2 + gamma_phi + kappa / 2) * a_sigmax
    ydot[6] = -1j * delta_rm * a_sigmay + chi * a_sigmax - 1j * epsilon_m * sigmay - (
            gamma_1 / 2 + gamma_phi + kappa / 2) * a_sigmay - Omega * a_sigmaz
    ydot[7] = -2 * epsilon_m * np.imag(a) - kappa * adagger_a

    return ydot

def cavity_bloch_equations_resonant_time_drive_variable_pulse_length(t, y, *args):
    """

    uses delta_as = - 2*chi*(adagger_a + 1/2)
    Pi pulse only happens between t1 and t2
    Measurement drive starts at t2

    :param y:
    :param t:
    :param args: (delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi)
    :return:
    """

    delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi, pulse_length = args

    t1 = -pulse_length
    t2 = 0
    # qubit control drive on in between -pulse_length and 0
    if not t1 < t < t2:
        Omega = 0

    # resonator measurement pulse on after t2
    if t <= t2:
        epsilon_m = 0

    a = y[0]
    sigmaz = y[1]
    sigmax = y[2]
    sigmay = y[3]
    a_sigmaz = y[4]
    a_sigmax = y[5]
    a_sigmay = y[6]
    adagger_a = y[7]

    ydot = np.array([0j] * len(y))

    ydot[0] = -1j * delta_rm * a - 1j * chi * a_sigmaz - 1j * epsilon_m - kappa / 2 * a
    ydot[1] = Omega * sigmay - gamma_1 * (1 + sigmaz)
    ydot[2] = - (gamma_1 / 2 + gamma_phi) * sigmax
    ydot[3] = - (gamma_1 / 2 + gamma_phi) * sigmay - Omega * sigmaz
    ydot[
        4] = -1j * delta_rm * a_sigmaz - 1j * chi * a + Omega * a_sigmay - 1j * epsilon_m * sigmaz - gamma_1 * a - (
            gamma_1 + kappa / 2) * a_sigmaz
    ydot[5] = -1j * delta_rm * a_sigmax - chi * a_sigmay - 1j * epsilon_m * sigmax - (
            gamma_1 / 2 + gamma_phi + kappa / 2) * a_sigmax
    ydot[6] = -1j * delta_rm * a_sigmay + chi * a_sigmax - 1j * epsilon_m * sigmay - (
            gamma_1 / 2 + gamma_phi + kappa / 2) * a_sigmay - Omega * a_sigmaz
    ydot[7] = -2 * epsilon_m * np.imag(a) - kappa * adagger_a

    return ydot
