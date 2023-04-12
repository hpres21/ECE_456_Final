import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from odeintw import odeintw

def cavity_bloch_equations(y, t, args):
    """

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

    ydot = np.zeros(len(y))

    ydot[0] = -1j * delta_rm * a - 1j * chi * a_sigmaz - 1j * epsilon_m - kappa/2 * a
    ydot[1] = Omega*sigmay - gamma_1 * (1+sigmaz)
    ydot[2] = -(delta_as + 2*chi*(adagger_a + 1/2))*sigmay - (gamma_1/2 + gamma_phi)*sigmax
    ydot[3] = (delta_as + 2*chi*(adagger_a + 1/2))*sigmax - (gamma_1/2 + gamma_phi)*sigmay - Omega*sigmaz
    ydot[4] = -1j * delta_rm * a_sigmaz - 1j * 1j*chi*a + Omega*a_sigmay - 1j*epsilon_m*sigmaz - gamma_1*a - (gamma_1 + kappa/2)*a_sigmaz
    ydot[5] = -1j * delta_rm * a_sigmax - (delta_as + 2*chi*(adagger_a+1))*a_sigmay - 1j*epsilon_m*sigmax - (gamma_1/2 + gamma_phi + kappa/2)*a_sigmax
    ydot[6] = -1j * delta_rm * a_sigmay + (delta_as + 2*chi*(adagger_a+1))*a_sigmax - 1j*epsilon_m*sigmay - (gamma_1/2 + gamma_phi + kappa/2)*a_sigmay - Omega*a_sigmaz
    ydot[7] = -2*epsilon_m*np.imag(a) - kappa*adagger_a

    return ydot


def cavity_bloch_equations_resonant_time_drive(y, t, *args):
    """

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

    ydot = np.array([0j]*len(y))

    ydot[0] = -1j * delta_rm * a - 1j * chi * a_sigmaz - 1j * epsilon_m - kappa / 2 * a
    ydot[1] = Omega * sigmay - gamma_1 * (1 + sigmaz)
    ydot[2] = - (gamma_1 / 2 + gamma_phi) * sigmax
    ydot[3] = - (gamma_1 / 2 + gamma_phi) * sigmay - Omega * sigmaz
    ydot[
        4] = -1j * delta_rm * a_sigmaz - 1j * 1j * chi * a + Omega * a_sigmay - 1j * epsilon_m * sigmaz - gamma_1 * a - (
                gamma_1 + kappa / 2) * a_sigmaz
    ydot[5] = -1j * delta_rm * a_sigmax - chi * a_sigmay - 1j * epsilon_m * sigmax - (
                          gamma_1 / 2 + gamma_phi + kappa / 2) * a_sigmax
    ydot[6] = -1j * delta_rm * a_sigmay + chi * a_sigmax - 1j * epsilon_m * sigmay - (
                          gamma_1 / 2 + gamma_phi + kappa / 2) * a_sigmay - Omega * a_sigmaz
    ydot[7] = -2 * epsilon_m * np.imag(a) - kappa * adagger_a

    return ydot

def cavity_bloch_equations_resonant_time_drive_short_pulse(y, t, *args):
    """

    :param y:
    :param t:
    :param args: (delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi)
    :return:
    """

    delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi = args

    t1 = 100
    t2 = t1 + 70
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

    ydot = np.array([0j]*len(y))

    ydot[0] = -1j * delta_rm * a - 1j * chi * a_sigmaz - 1j * epsilon_m - kappa / 2 * a
    ydot[1] = Omega * sigmay - gamma_1 * (1 + sigmaz)
    ydot[2] = - (gamma_1 / 2 + gamma_phi) * sigmax
    ydot[3] = - (gamma_1 / 2 + gamma_phi) * sigmay - Omega * sigmaz
    ydot[
        4] = -1j * delta_rm * a_sigmaz - 1j * 1j * chi * a + Omega * a_sigmay - 1j * epsilon_m * sigmaz - gamma_1 * a - (
                gamma_1 + kappa / 2) * a_sigmaz
    ydot[5] = -1j * delta_rm * a_sigmax - chi * a_sigmay - 1j * epsilon_m * sigmax - (
                          gamma_1 / 2 + gamma_phi + kappa / 2) * a_sigmax
    ydot[6] = -1j * delta_rm * a_sigmay + chi * a_sigmax - 1j * epsilon_m * sigmay - (
                          gamma_1 / 2 + gamma_phi + kappa / 2) * a_sigmay - Omega * a_sigmaz
    ydot[7] = -2 * epsilon_m * np.imag(a) - kappa * adagger_a

    return ydot

def figure2(*args):
    # for this experiment

    omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi = args

    epsilon_m = np.sqrt(kappa / 2) * 0
    omega_m = omega_r - chi

    omega_s = 0
    delta_as = omega_a - omega_s
    delta_rm = omega_r - omega_m

    Omega = 0.025 * 2 * np.pi

    args = delta_rm, chi, epsilon_m, kappa, Omega, gamma_1, delta_as, gamma_phi

    a0 = 0j
    sigmaz0 = -1
    sigmax0 = 0
    sigmay0 = 0
    a_sigmaz0 = 0j
    a_sigmax0 = 0j
    a_sigmay0 = 0j
    adagger_a0 = 0

    y0 = np.array([a0, sigmaz0, sigmay0, sigmax0, a_sigmaz0, a_sigmay0, a_sigmax0, adagger_a0])


    times = np.linspace(0, 500, 5000)
    y, infodict = odeintw(cavity_bloch_equations_resonant_time_drive_short_pulse, y0, times, args=args, full_output=True)

    print(y.shape)

    a = y[:,0]
    sigmaz = y[:,1]
    sigmax = y[:,2]
    sigmay = y[:,3]
    a_sigmaz = y[:,4]
    a_sigmax = y[:,5]
    a_sigmay = y[:,6]
    adagger_a = y[:,7]

    plt.plot(times, sigmaz, label="sigma_z")
    plt.plot(times, adagger_a, label="adag a")
    plt.plot(times, np.real(a), label="I")
    plt.plot(times, np.imag(a), label="Q")

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


    figure2(omega_r, kappa, omega_a, g, chi, gamma_1, gamma_phi)


if __name__ == "__main__":
    cavity_bloch_numerical()
