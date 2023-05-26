import numpy as np

'''
    Numerical solution of the integral in the optical model of the Tauk-Lorentz-Drude. 
    
    get_optics_func(*args)
    
    args = dict, key - number of the oscillator, key - dict with the parameters of the oscillator. 
    e_inf should be specified only for the first oscillator. 
    
    Returns the optics function of wavelength.
    
    Jellison and Modine, APL 69, 371-373 (1996)
'''



e_inf = 0
A = []
E_0 = []
G = []
E_g = []
N = 0
mu = 0
m_star = 0
rho = 0
tau = 0





def a_ln(E, i):
    t1 = (E_g[i] ** 2 - E_0[i] ** 2) * E ** 2
    t2 = E_g[i] ** 2 * G[i] ** 2
    t3 = -E_0[i] ** 2 * (E_0[i] ** 2 + 3 * E_g[i] ** 2)
    return t1 + t2 + t3

def a_atan(E, i):
    t1 = (E ** 2 - E_0[i] ** 2) * (E_0[i] ** 2 + E_g[i] ** 2)
    t2 = E_g[i] ** 2 * G[i] ** 2
    return t1 + t2

def ksi(E, i):
    t1 = np.power(np.power(E, 2) - gamma(i) ** 2, 2)
    t2 = 0.25 * alpha(i) ** 2 * G[i] ** 2
    return np.power(t1 + t2, 0.25)

def alpha(i):
    return (4 * E_0[i]**2 - G[i]**2)**0.5

def gamma(i):
    return (E_0[i]**2 - 0.5 * G[i]**2)**0.5

"""
    Real part of dielectric function.
"""
def e_re(A, E_0, G, E_g, E, i):
    t1 = A * G * a_ln(E, i) / (2 * np.pi * ksi(E, i) ** 4 * alpha(i) * E_0) * np.log(
        (E_0 ** 2 + E_g ** 2 + alpha(i) * E_g) / (E_0 ** 2 + E_g ** 2 - alpha(i) * E_g))
    t2 = -A * a_atan(E, i) / (np.pi * ksi(E, i) ** 4 * E_0) * (
                np.pi - np.arctan(1 / G * (2 * E_g + alpha(i))) + np.arctan(1 / G * (alpha(i) - 2 * E_g)))
    t3 = 4 * A * E_0 * E_g * (E ** 2 - gamma(i) ** 2) / (np.pi * ksi(E, i) ** 4 * alpha(i)) * (
                np.arctan(1 / G * (alpha(i) + 2 * E_g)) + np.arctan(1 / G * (alpha(i) - 2 * E_g)))
    t4 = -A * E_0 * G * (E ** 2 + E_g ** 2) / (np.pi * ksi(E, i) ** 4 * E) * np.log(np.fabs(E - E_g) / (E + E_g))
    t5 = 2 * A * E_0 * G * E_g / (np.pi * ksi(E, i) ** 4) * np.log(
        np.fabs(E - E_g) * (E + E_g) / ((E_0 ** 2 - E_g ** 2) ** 2 + E_g ** 2 * G ** 2) ** 0.5)
    return t1 + t2 + t3 + t4 + t5

def calc_real_parts(E, e_inf):
    sum_real_parts = e_inf
    for i in range(len(A)):
        sum_real_parts = sum_real_parts + e_re(A[i], E_0[i], G[i], E_g[i], E, i)
    return sum_real_parts

"""
    Imaginary part of dielectric function.
"""
def e_im(A, E_0, G, E_g, E):
    result = 1 / E * A * E_0 * G * (E - E_g)**2 / ((E**2 - E_0**2)**2 + G**2 * E**2)
    out = np.where(E > E_g, result, 0)
    return out


def calc_imag_parts(E):
    sum_imag_parts = 0
    for i in range(len(A)):
        sum_imag_parts = sum_imag_parts + e_im(A[i], E_0[i], G[i], E_g[i], E)
    return sum_imag_parts

"""
    Contribution of the Drude oscillator
"""
def drude(E):
    """
        N [ 1/сm^3 ]
        mu [cm^2/(V*s)]
        m_star [unitless]

    """
    h = 6.582e-16  # eV * s
    m_e = 9.109e-31  # kg
    q_e = 1.602e-19  # Coulomb
    e_0 = 8.854e-12  # F/m

    return -(((h*q_e)**2)*N*mu)/(e_0*(mu*m_e*m_star*((E)**2)+complex(0, 1)*q_e*h*E))

def get_model_params(*args):
    global optics_params, e_inf

    oscillators = args[0]

    if type(oscillators) != dict:
        print("Неверный формат слоя!")
        return

    for key, values in oscillators.items():
        if 'Tauc-Lorentz' in key:
            for param, value in values.items():
                if param == 'e_inf':
                    e_inf = value
                elif param == 'A':
                    A.append(value)
                elif param == 'E_0':
                    E_0.append(value)
                elif param == 'G':
                    G.append(value)
                elif param == 'E_g':
                    E_g.append(value)
        elif 'Drude' in key:
            global N, m_star, mu
            for param, value in values.items():
                if param == 'N':
                    N = value*10**6
                elif param == 'mu':
                    mu = value*10**(-4)
                elif param == 'm_star':
                    m_star = value

        elif 'Drude_RT' in key:
            global rho, tau
            for param, value in values.items():
                if param == 'rho':
                    rho = value * 10 ** -2
                elif param == 'mu':
                    tau = value * 10 ** (-15)



        def optics_params(lamda, output='RI'):
            global e_inf
            h = 4.135667662e-15
            c = 299792458
            E = h * c / (lamda*1e-9)

            re = calc_real_parts(E, e_inf)
            im = calc_imag_parts(E)

            drude_part = drude(E)
            re = re + drude_part.real
            im = im + drude_part.imag

            if output == 'RI':
                func_n = np.real_if_close(np.sqrt(0.5 * (np.sqrt(re**2 + im**2) + re)))
                func_k = np.real_if_close(np.sqrt(0.5 * (np.sqrt(re**2 + im**2) - re)))
                return complex(func_n, func_k)

            elif output == 'dielectric_function':
                return complex(re, im)

            else:
                return complex(0, 0)

    return optics_params


"""
    After you finish working with the optics of the layer, you need to clear the data
"""
def clear():
    global A, E_0, G, E_g, N, tau, rho, mu, m_star
    A.clear()
    E_0.clear()
    G.clear()
    E_g.clear()
    N = 0
    tau = 0
    rho = 0
    m_star = 0
    mu = 0







# params =  get_model_params( {'Tauc-Lorentz_1': {'e_inf': 2.957, 'A': 56.0033, 'E_0': 4.931, 'G': 4.028, 'E_g': 2.094},
#                  'Tauc-Lorentz_2': {'A': 191.332, 'E_0': 3.251, 'G': 0.253, 'E_g': 3.251},
#                  'Tauc-Lorentz_3': {'A': 66.0704, 'E_0': 2.629, 'G': 1.533, 'E_g': 2.096},
#                  'Drude': {'N': 2.56E22, 'mu': 0.556, 'm_star': 1.122}})   #{'Tauc-Lorentz_1': {'e_inf': 1, 'A': 120, 'E_0': 4.5, 'G':2.2, 'E_g': 1.3}, 'd': 1})  #{'Tauc-Lorentz_1': {'e_inf': 1, 'A': 120, 'E_0': 4.5, 'G':2.2, 'E_g': 1.3} ,
#
# w = [i for i in range(250, 850, 10)]
# for i in w:
#     n= params(i).imag
#     print(i, '\t', n)