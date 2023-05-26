from random import choice, randint
import tauc_lorentz as tl
import numpy as np
import pandas as pd
import math
import cmath
from scipy import interpolate
from functools import reduce
import os


class reflectometry_tool:

    def __init__(self):
        self.constant = 1
        self.Wavelengths=(250, 801, 10)

    def upload_data_Si_RI(self, path='Si 1983 0,21-0,83 Full.xlsx', Wavelengths=(250, 801, 1)):
        # доделать ссылку на подложку!!!
        Si_data = pd.read_excel('Si 1983 0,21-0,83 Full.xlsx', engine='openpyxl')
        SiModel = np.reshape(Si_data.values, (-1, 3))
        fn = interpolate.interp1d([i * 1000 for i in SiModel[:, 0]], [i for i in SiModel[:, 1]])
        fk = interpolate.interp1d([i * 1000 for i in SiModel[:, 0]], [i for i in SiModel[:, 2]])
        Si = [complex(fn(i), -fk(i)) for i in range(*Wavelengths)]
        return Si

    def Im(self, r_mj, t_mj):
        I = np.array([[1 / t_mj, r_mj / t_mj], [r_mj / t_mj, 1 / t_mj]], dtype=complex)
        return I

    def Lm(self, z_mj):
        L = np.array([[z_mj, 0], [0, 1 / z_mj]], dtype=complex)
        return L

    def rms(self, data, wave):
        rs, ts, zs, S = [], [], [], []
        rs = [[(data[i][0][j] * cmath.cos(data[i][2][j] * math.pi / 180) - data[i + 1][0][j] * cmath.cos(
            data[i + 1][2][j] * math.pi / 180)) / (
                       data[i][0][j] * cmath.cos(data[i][2][j] * math.pi / 180) + data[i + 1][0][j] * cmath.cos(
                   data[i + 1][2][j] * math.pi / 180)) for i in range(len(data) - 1)] for j in range(len(wave))]

        ts = [[2 * data[i][0][j] * cmath.cos(data[i][2][j] * math.pi / 180) / (
                data[i][0][j] * cmath.cos(data[i][2][j] * math.pi / 180) + data[i + 1][0][j] * cmath.cos(
            data[i + 1][2][j] * math.pi / 180)) for i in range(len(data) - 1)] for j in range(len(wave))]

        zs = [[cmath.exp(
            (2 * math.pi * data[i][1][j] * data[i][0][j] * complex(0, 1) * cmath.cos(data[i][2][j] * math.pi / 180)) /
            wave[j]) for i in range(len(data) - 1)] for j in range(len(wave))]

        for i in range(len(data) - 1):
            S = [[[self.Lm(zs[i][j]), self.Im(rs[i][j], ts[i][j])] for j in range(len(data) - 1)] for i in range(len(rs))]
        S = [np.concatenate(j) for j in S]
        return S


    def rmp(self, data, wave):
        rp, tp, zp, S = [], [], [], []
        rp = [[(data[i + 1][0][j] * cmath.cos(data[i][2][j] * math.pi / 180) - data[i][0][j] * cmath.cos(
            data[i + 1][2][j] * math.pi / 180)) / (
                       data[i + 1][0][j] * cmath.cos(data[i][2][j] * math.pi / 180) + data[i][0][j] * cmath.cos(
                   data[i + 1][2][j] * math.pi / 180)) for i in range(len(data) - 1)] for j in range(len(wave))]
        tp = [[2 * data[i][0][j] * cmath.cos(data[i][2][j] * math.pi / 180) / (
                data[i + 1][0][j] * cmath.cos(data[i][2][j] * math.pi / 180) + data[i][0][j] * cmath.cos(
            data[i + 1][2][j] * math.pi / 180)) for i in range(len(data) - 1)] for j in range(len(wave))]
        zp = [[cmath.exp(
            2 * math.pi * data[i][1][j] * data[i][0][j] * cmath.cos(data[i][2][j] * math.pi / 180) * complex(0, 1) /
            wave[j]) for i in range(len(data) - 1)] for j in range(len(wave))]
        for i in range(len(data) - 1):
            S = [[[self.Lm(zp[i][j]), self.Im(rp[i][j], tp[i][j])] for j in range(len(data) - 1)] for i in range(len(rp))]
            S = [np.concatenate(j) for j in S]
        return S

    """
        Calculation of angles according to Snellius' law
    """
    def sinm(self, data):
        for i in range(len(data) - 1):
            for j in range(len(data[i][2])):
                data[i + 1][2][j] = 180 * (cmath.asin(
                    data[i][0][j] / data[i + 1][0][j] * cmath.sin(cmath.pi * (data[i][2][j]) / 180))) / math.pi
        return data


    """
        Returns the optical wavelength function for a layer without absorption
    """
    def Couchy_Layer(self, layer):
        A, B, C = [x[1] for x in layer['Couchy'].items()]
        func = lambda x: complex(A + B / (x/1000) ** 2 + C / (x/1000) ** 4, -0)
        return func





    """
        OpticLayers - слои на подложке. Пишутся в порядке от верхнего слоя к нижниму.
        Тип данных list, тип данных слоя - словарь. В качестве значений словоря выступают тоже словарь с параметрами осциллятора. 
        Также можно передать передать уже готовую callable функцию от длины волны, которая вернёт комплексный показатель преломления. 
        Возможные варианты ключей в слое: Couchy, Tauc-Lorentz_n, Drude и optic_function . В случае нескольких осциляторов для каждого ключа Tauc-Lorentz_n
        указать словрь с параметрами осциллятора.  Для n>1 e_inf указать только для первого осциллятора.
        Примеры:
                1) Прозрачный слой:
                {'Couchy' : {'A': 1.452, 'B': 0, 'C': 0, 'd': 200}}    (defolt)
                
                2) Слой с поглощением:
                {'Tauc-Lorentz_1': {'e_inf': 1, 'A': 128.6, 'E_0': 2.52, 'G': 3.93, 'E_g': 0.67}, 'Tauc-Lorentz_2': {'A': 153.7, 'E_0': 3.32, 'G': 3.93, 'E_g': 0.67}, 'd': 25}
                
                3)Металлический слой, с поглащением на свободных электронах:
                {'Tauc-Lorentz_1': {'e_inf': 1, 'A': 128.6, 'E_0': 2.52, 'G': 3.93, 'E_g': 0.67}, 'Drude':  {'N': 2.2e+22, 'mu': 5, 'm_star': 0.8}, 'd': 25}
                
                4)Слой, с заданной оптикой
                {'optic_function': функиця_от_длины_волны, 'd': d}
                
            
        LayerPodl - название файла формата .xlsx с оптикой подлжожки. Столбцы: Длина волны, коэффициент преломления (n), коэффициент затухания (k)
        tetta - угол падения, тип данных float
        Wavelengths - диапазон длин волн, (начало, конец, шаг). Последнее значение - не войдет в диапазон (См. функцию range())
        output_data - формат данных:
            R_mix - коэффициент отражения для смежанных s и p поляризации
            Rs - коэффициент отражения для s поляризации
            Rp - коэффициент отражения для p поляризации
            r_mix - энергетический коэффициент отражения для смежанных s и p поляризации
            r_s - энергетический коэффициент отражения для s поляризации
            r_p - энергетический коэффициент отражения для p поляризации
            Phi_delta - эллипсометрические углы
            Full - эллипсометрические углы и коэффициент отражения для смежанных s и p поляризации
    """
    def calc_refl_coef_for_diff_PS_polarisation(self, OpticLayers=[{'Couchy' : {'A': 1.452, 'B': 0, 'C': 0}, 'd': 200}],
                                                       LayerPodl='Si 1983 0,21-0,83 Full.xlsx',
                                                       tetta=float(0),
                                                       Wavelengths=(250, 810, 10),
                                                       output_data = "R_mix",
                                                       save_data = False):



        Si = np.array(self.upload_data_Si_RI(LayerPodl, Wavelengths=Wavelengths))
        thicknesses = []
        layer_optica = []

        for i in OpticLayers:
            if 'Couchy' in i.keys():
                optica = self.Couchy_Layer(i['Couchy'])
                d = i['d']
                thicknesses.append(d)
                layer_optica.append(optica)
            elif 'Tauc-Lorentz_1' in i.keys() or 'Drude' in i.keys():
                tl.clear()
                optica = tl.get_model_params(i)
                d = i['d']
                thicknesses.append(d)
                layer_optica.append(optica)
            elif 'optic_function' in i.keys():
                optica = i['optic_function']
                d = i['d']
                thicknesses.append(d)
                layer_optica.append(optica)

        LayerFin = []
        LayerFin.append(
            np.array([[complex(1, -0)] * len(Si), [0] * len(Si), [tetta] * len(Si)]))  # Добавление нулевого слоя воздуха



        if (len(OpticLayers) > 0):

            for i in range(len(OpticLayers)):
                LayerFin.append(np.array([[layer_optica[i](j) for j in range(*Wavelengths)], [thicknesses[i]] * len(Si),
                                          [tetta] * len(Si)]))  # Добавление промежуточных слоёв

        LayerFin.append(np.array([Si, [0] * len(Si), [0] * len(Si)]))  # Добавление подложки

        if tetta != 0:
            LayerFin = self.sinm(LayerFin)

        # s-polarisation
        S_s = self.rms(LayerFin, [i for i in range(*Wavelengths)])
        S_s = [reduce(np.dot, i) for i in S_s]

        # p-polarisation
        S_p = self.rmp(LayerFin, [i for i in range(*Wavelengths)])
        S_p = [reduce(np.dot, i) for i in S_p]


        if output_data == 'R_mix':
            Rs = [(i[1][0] * i[1][0].conjugate()) / (i[0][0] * i[0][0].conjugate()) for i in S_s]
            Rp = [(i[1][0] * i[1][0].conjugate()) / (i[0][0] * i[0][0].conjugate()) for i in S_p]
            if save_data == True:
                self.save_data(np.array([(Rs[i].real + Rp[i].real) / 2 for i in range(len(Rs))]), waves=Wavelengths)
            return np.array([(Rs[i].real + Rp[i].real) / 2 for i in range(len(Rs))])
        elif output_data == 'R_p':
            Rp = [(i[1][0] * i[1][0].conjugate()) / (i[0][0] * i[0][0].conjugate()) for i in S_p]
            if save_data == True:
                self.save_data(Rp, waves=Wavelengths)
            return np.array(Rp)
        elif output_data == 'R_s':
            Rs = [(i[1][0] * i[1][0].conjugate()) / (i[0][0] * i[0][0].conjugate()) for i in S_s]
            if save_data == True:
                self.save_data(Rs, waves=Wavelengths)
            return np.array(Rs)
        elif output_data == 'r_mix':
            rs = [i[1][0] / i[0][0] for i in S_s]
            rp = [i[1][0] / i[0][0] for i in S_p]
            return np.array((rs[i]+rp[i])/2 for i in range(len(rs)))
        elif output_data == 'r_s':
            rs = [i[1][0] / i[0][0] for i in S_s]
            return np.array(rs)
        elif output_data == 'r_p':
            rp = [i[1][0]/ i[0][0] for i in S_p]
            return np.array(rp)
        elif output_data == 'Phi_delta':
            rp = [i[1][0]/ i[0][0] for i in S_p]
            rs = [i[1][0] / i[0][0] for i in S_s]
            Phi, delta = self.calculate_phi_delata(rs, rp)
            return Phi, delta
        elif output_data == 'Full':
            rp = [i[1][0] / i[0][0] for i in S_p]
            rs = [i[1][0] / i[0][0] for i in S_s]
            Phi, delta = self.calculate_phi_delata(rs, rp)
            Rs = [(i[1][0] * i[1][0].conjugate()) / (i[0][0] * i[0][0].conjugate()) for i in S_s]
            Rp = [(i[1][0] * i[1][0].conjugate()) / (i[0][0] * i[0][0].conjugate()) for i in S_p]
            return Phi, delta, np.array([(Rs[i].real + Rp[i].real) / 2 for i in range(len(Rs))])


    """
        RMSE is a metric for using optimization methods to find the best model parameters.
    """
    def root_mean_square_error(self, target, pred, Wavelength):
        Rtarget = interpolate.interp1d(target[:, 0], target[:, 1])
        Wavelengths = list(range(*Wavelength))
        RMSE = (sum([(Rtarget(Wavelengths[i]) - pred[i]) ** 2 for i in range(len(Wavelengths))]) / len(pred)) ** (1 / 2)
        return RMSE


    # Целевая функция МЕТОДА ОПТИМИЗАЦИИ
    def minimise_function(self, Layer, Wavelength, data):
        A, B, C, d = Layer
        R = self.calc_refl_coef_for_diff_PS_polarisation(OpticLayers=[{'Couchy': {'A': A, 'B': B, 'C': C}, 'd': d}], Wavelengths=Wavelength)
        MSE = self.root_mean_square_error(data, R, Wavelength)
        return MSE

    def calculate_lambda_RI(self, data, lambda0):
        lambda0 = float(lambda0)
        Wavelength1 = []
        nn = []
        kk = []
        for i in range(len(data)):
            Wavelength1.append(data[i][0] * 1000)
            nn.append(data[i][1])
            kk.append(data[i][2])
        fn = interpolate.interp1d(Wavelength1, nn)
        fk = interpolate.interp1d(Wavelength1, kk)
        n3 = fn(lambda0)
        k3 = fk(lambda0)
        return float(n3), float(k3)


    """
        Calculation of ellipsometry angles
    """
    def calculate_phi_delata(self, rs, rp):
        Phi = []
        delta = []
        for i in range(len(rs)):
          rho = rp[i] / rs[i]
          Phi.append(math.degrees(math.atan(abs(rp[i]) / abs(rs[i]))))
          if (rho.real > 0):
              delta.append(math.degrees(math.atan(rho.imag / rho.real)))
          elif (rho.real < 0 and rho.imag >= 0):
              delta.append(math.degrees(math.atan(rho.imag / rho.real)) + 180)
          elif (rho.real < 0 and rho.imag < 0):
              delta.append(math.degrees(math.atan(rho.imag / rho.real)) + 180)
          elif (rho.real == 0 and rho.imag > 0):
              delta.append(90)
          elif (rho.real == 0 and rho.imag < 0):
              delta.append(-90)
        return np.array(Phi), np.array(delta)

    """
        Saves data to a file
    """
    def save_data(self, data, waves=(250, 800, 5)):
        with open('data.txt', 'w') as output:
            wave = list(range(*waves))
            for number in range(len(data)):
                print(str(wave[number]), str(data[number]), file=output, sep='\t')


    """
        Creating a dataset for absorbing films with random oscillator parameters and thickness.
    """
    def random_dataset_for_ML_tauc_lorentz(self, count: int, oscillators: dict, positive_e2=False,  *args):
        columns = [str(w)+'Phi' for w in range(*self.Wavelengths)] + [str(w)+'delta' for w in range(*self.Wavelengths)] + [str(w)+'R' for w in range(*self.Wavelengths)] + \
                             [str(w)+'n' for w in range(*self.Wavelengths)] + [str(w)+'k' for w in range(*self.Wavelengths)] + \
                             ['e_inf', 'A_1', 'E_0_1', 'G_1', 'E_g_1', 'A_2', 'E_0_2', 'G_2', 'E_g_2', 'A_3', 'E_0_3', 'G_3', 'E_g_3', 'N', 'mu', 'm_star', 'd']
        full_data = pd.DataFrame(columns=columns)



        e_inf = 0
        A = []
        E_0 = []
        G = []
        E_g = []
        N = 0
        mu = 0
        m_star = 0

        if type(oscillators) != dict:
            print("Неверный формат!")
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

                for param, value in values.items():
                    if param == 'N':
                        N = value
                    elif param == 'mu':
                        mu = value
                    elif param == 'm_star':
                        m_star = value

        random_value = lambda x, y=10: x + randint(-y, y)/100*x

        A_new = [0 for _ in range(len(A))]
        E_0_new = [0 for _ in range(len(A))]
        E_g_new = [0 for _ in range(len(A))]
        G_new = [0 for _ in range(len(A))]
        N_new, mu_new, m_star_new = 0, 0, 0
        e_inf_new = 1
        for _ in range(count):

            e_inf_new = random_value(e_inf, 20)
            d = randint(10, 1000)/10
            for i in range(len(A)):
                A_new[i] = random_value(A[i])
                E_0_new[i] = random_value(E_0[i])
                G_new[i] = random_value(G[i])
                E_g_new[i] = random_value(E_g[i])
                N_new = random_value(N)
                mu_new = random_value(mu, 20)
                m_star_new = random_value(m_star, 20)

                while 4*E_0_new[i]**2 - G_new[i]**2 < 0 or E_0_new[i]**2-G_new[i]**2/2 < 0:
                    E_0_new[i] = random_value(E_0[i])
                    G_new[i] = random_value(G[i])

            print(e_inf_new, A_new[0], E_0_new[0], G_new[0], E_g_new[0],A_new[1], E_0_new[1], G_new[1], E_g_new[1],
                  A_new[2], E_0_new[2], G_new[2], E_g_new[2], N_new, mu_new, m_star_new, d)

            tl.clear()
            optics = tl.get_model_params(
                {'Tauc-Lorentz_1': {'e_inf': e_inf_new, 'A': A_new[0], 'E_0': E_0_new[0], 'G': G_new[0], 'E_g': E_g_new[0]},
                 'Tauc-Lorentz_2': {'A': A_new[1], 'E_0': E_0_new[1], 'G': G_new[1], 'E_g': E_g_new[1]},
                 'Tauc-Lorentz_3': {'A': A_new[2], 'E_0': E_0_new[2], 'G': G_new[2], 'E_g': E_g_new[2]},
                 'Drude': {'N': N_new, 'mu': mu_new, 'm_star': m_star_new}})

            if positive_e2 == False:
                Phi, delta, Rmix = self.calc_refl_coef_for_diff_PS_polarisation(tetta=float(65),
                                                                Wavelengths=self.Wavelengths,
                                                                OpticLayers=[{'optic_function': optics, 'd': d}],
                                                                LayerPodl='Si',
                                                                output_data="Full",
                                                                save_data=False)


                n = np.array([optics(i, output='RI') for i in range(*self.Wavelengths)])
                n, k = n.real, n.imag
                small_data = pd.DataFrame(np.concatenate([Phi, delta, Rmix, n, k, [e_inf_new, A_new[0], E_0_new[0], G_new[0], E_g_new[0],
                                                            A_new[1], E_0_new[1], G_new[1], E_g_new[1], A_new[2], E_0_new[2], G_new[2], E_g_new[2],
                                                                                   N_new, mu_new, m_star_new, d]])).T
                small_data.columns = columns
                full_data = pd.concat([full_data, small_data])
            else:
                flag = True
                for wave in range(250, 801, 25):
                    if optics(wave, output='dielectric_function').imag < 0:
                        flag = False
                        break
                if flag == True:
                    n = np.array([optics(i, output='RI') for i in range(*self.Wavelengths)])
                    n, k = n.real, n.imag

                    Phi, delta, Rmix = self.calc_refl_coef_for_diff_PS_polarisation(tetta=float(65),
                                                                                           Wavelengths=self.Wavelengths,
                                                                                           OpticLayers=[{'optic_function': optics,
                                                                                                            'd': d}],
                                                                                           LayerPodl='Si',
                                                                                           output_data="Full",
                                                                                           save_data=False)

                    small_data = pd.DataFrame(
                    np.concatenate([Phi, delta, Rmix, n, k, [e_inf_new, A_new[0], E_0_new[0], G_new[0], E_g_new[0],
                                                             A_new[1], E_0_new[1], G_new[1], E_g_new[1], A_new[2],
                                                             E_0_new[2], G_new[2], E_g_new[2],
                                                             N_new, mu_new, m_star_new, d]])).T
                    small_data.columns = columns
                    full_data = pd.concat([full_data, small_data])


        return full_data

    '''
        Creating a dataset for a Cauchy model with discrete parameters.
    '''
    def discrete_dataset_for_ML_couchy(self, *args):
        columns = [str(w)+'Phi' for w in range(*self.Wavelengths)] + [str(w)+'delta' for w in range(*self.Wavelengths)] + [str(w)+'R' for w in range(*self.Wavelengths)] + ['A', 'B', 'C', 'd']
        full_data = pd.DataFrame(columns=columns)
        for A in range(13, 25, 1):
            for B in range(0, 201, 10):
                for C in range(0, 201, 10):
                    for d in list(range(1, 10, 2))+list(range(10, 551, 10)):
                        Phi, delta, Rmix = self.calc_refl_coef_for_diff_PS_polarisation(tetta=float(65),
                                                            Wavelengths=self.Wavelengths,
                                                            OpticLayers=[{'Couchy': {'A': A/10, 'B': (B-100)/10000, 'C': (C-100)/100000}, 'd': d}],
                                                            LayerPodl='Si',
                                                            output_data="Full",
                                                            save_data=False)
                        small_data = pd.DataFrame(np.concatenate([Phi, delta, Rmix, [A/10, (B-100)/10000, (C-100)/100000, d]])).T
                        small_data.columns = columns
                        full_data = pd.concat([full_data, small_data])
        return full_data


    '''
        Creating a dataset for a Cauchy model with random parameters.
    '''
    def random_dataset_for_ML_couchy(self, count, *args):
        columns = [str(w)+'Phi' for w in range(*self.Wavelengths)] + \
                  [str(w)+'delta' for w in range(*self.Wavelengths)] + \
                  [str(w)+'R' for w in range(*self.Wavelengths)] + \
                  [str(w)+'n' for w in range(*self.Wavelengths)] + \
                  [str(w)+'k' for w in range(*self.Wavelengths)] + \
                  ['A', 'B', 'C', 'd']
        full_data = pd.DataFrame(columns=columns)

        A_all = list(range(125, 251, 1))
        B_all = list(range(-800, 800, 2))
        C_all = list(range(-800, 800, 2))
        d_all = list(range(1, 5500, 2))

        for _ in range(count):
            A = choice(A_all)/100
            B = choice(B_all) / 10000
            C = choice(C_all) / 100000
            d = choice(d_all)/10
            optics = lambda x: complex(A + B/(x/1000)**2 + C/(x/1000)**4, 0)
            Phi, delta, Rmix = self.calc_refl_coef_for_diff_PS_polarisation(tetta=float(65),
                                                            Wavelengths=self.Wavelengths,
                                                            OpticLayers=[{'optic_function': optics,  'd': d}],
                                                            LayerPodl='Si',
                                                            output_data="Full",
                                                            save_data=False)
            RI = np.array([optics(i) for i in range(*self.Wavelengths)])
            small_data = pd.DataFrame(np.concatenate([Phi, delta, Rmix, RI.real, RI.imag, [A, B, C, d]])).T

            small_data.columns = columns

            if len(full_data.values) == 0:
                full_data = pd.concat([full_data, small_data])
            else:
                full_data = pd.DataFrame(
                    np.concatenate([full_data.values, small_data.values], axis=0),
                    columns=columns
                )

        return full_data

    def RI_to_params(self, file, d):
        columns = [str(w) + 'Phi' for w in range(*self.Wavelengths)] + \
                  [str(w) + 'delta' for w in range(*self.Wavelengths)] + \
                  [str(w) + 'R' for w in range(*self.Wavelengths)] + \
                  [str(w) + 'n' for w in range(*self.Wavelengths)] + \
                  [str(w) + 'k' for w in range(*self.Wavelengths)] + ['d']

        input_data = pd.read_excel(file, engine='openpyxl')
        input_data = np.reshape(input_data.values, (-1, 3))
        fn = interpolate.interp1d([i for i in input_data[:, 0]], [i for i in input_data[:, 1]])
        fk = interpolate.interp1d([i for i in input_data[:, 0]], [i for i in input_data[:, 2]])
        input_RI = lambda i: complex(fn(i), -fk(i))
        Phi, delta, Rmix = self.calc_refl_coef_for_diff_PS_polarisation(tetta=float(65),
                                                                        Wavelengths=self.Wavelengths,
                                                                        OpticLayers=[{'optic_function': input_RI,'d': d}],
                                                                        LayerPodl='Si',
                                                                        output_data="Full",
                                                                        save_data=False)
        n = np.array([input_RI(i) for i in range(*self.Wavelengths)])
        n, k = n.real, -n.imag
        out_data = pd.DataFrame(np.concatenate([Phi, delta, Rmix, n, k, [d]])).T
        out_data.columns = columns
        return out_data

    def dirRI_to_params(self, path):
        columns = [str(w) + 'Phi' for w in range(*self.Wavelengths)] + \
                  [str(w) + 'delta' for w in range(*self.Wavelengths)] + \
                  [str(w) + 'R' for w in range(*self.Wavelengths)] + \
                  [str(w) + 'n' for w in range(*self.Wavelengths)] + \
                  [str(w) + 'k' for w in range(*self.Wavelengths)] + ['d']
        files = os.listdir(path)
        final_data = pd.DataFrame(columns=columns)


        for name in files:
            print(name)
            for d in range(1, 100):
                data = self.RI_to_params(f'{path}\\{name}', d)
                final_data = pd.concat([final_data, data])

        return final_data


# a = reflectometry_tool()
# a.Wavelengths = (250, 801, 2)
# #aa = a.dirRI_to_params(path = 'D:\\Проекты\\ReflectML\\Оптика Нитриды\\TiN')
# params = {'Tauc-Lorentz_1': {'e_inf': 2.993, 'A': 166.5276, 'E_0': 2.151, 'G': 1.912, 'E_g': 3.748},
#                  'Tauc-Lorentz_2': {'A': 162.4371, 'E_0': 3.284, 'G': 0.196, 'E_g': 3.284},
#                  'Tauc-Lorentz_3': {'A': 38.3279, 'E_0': 5.076, 'G': 3.618, 'E_g': 2.051},
#                  'Drude': {'N': 1.6478E22, 'mu': 0.854, 'm_star': 0.721}}

# #aa = a.random_dataset_for_ML_tauc_lorentz(count = 5000, positive_e2=True, oscillators= params)
# #aa.to_csv('data_TL_random_optics_TiN_1_5000.csv')


# aa = a.random_dataset_for_ML_couchy(10000)
# aa.to_csv('data_random_optics_Couchy_10000_2.csv')

