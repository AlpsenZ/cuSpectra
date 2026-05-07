"""
cuLBL v2.4
2025/9/1
Added grid striding
Obtained temperature-pressure grid
Output spectra
Fixed typo in Humlicek q8 coefficient
"""
import numpy as np
import math
from time import time
import matplotlib.pylab as plb
import scipy
from numba import cuda, float64
from hapi import db_begin, getColumns, PYTIPS, molecularMass, transmittanceSpectrum
from scipy.io import savemat

h = 6.62607015e-27  # erg s
c = 2.99792458e10  # cm s−1
k = 1.380649e-16  # erg K−1
c2 = 1.4387769  # cm K
Na = 6.02214129e23  # mol-1

cLn2 = 0.6931471805599
cSqrtLn2 = 0.8325546111577
cSqrtLn2divSqrtPi = 0.469718639319144059835

Tref = 296  # K
Pref = 1.  # atm


def volumeConcentration(p, T):
    return (p / 9.869233e-7) / (k * T)


def sw_pt(sw, T, v, El, m_id, i_id):
    return sw * PYTIPS(m_id, i_id, Tref) / PYTIPS(m_id, i_id, T) * np.exp(-c2 * El / T) / np.exp(-c2 * El / Tref) * (
            1 - np.exp(-c2 * v / T) / (1 - np.exp(-c2 * v / Tref)))


def doppler_hwhm(v, T, M):
    return v / c * np.sqrt(2 * Na * k * T * cLn2 / M)


def lorentz_hwhm(P, T, g_air, g_self, n_a, P_s):
    return np.power(Tref / T, n_a) * (g_air * (P - P_s) + g_self * P_s)


@cuda.jit(device=True)
def voigt_profile(delta, g_d, g_l):
    x = delta * cSqrtLn2 / g_d
    y = g_l * cSqrtLn2 / g_d
    return cSqrtLn2divSqrtPi / g_d * voigt_func(x, y)


@cuda.jit(device=True)
def voigt_func(x, y):
    if abs(x) + y > 15:
        a1 = 0.2820948 * y + 0.5641896 * pow(y, 3)
        b1 = 0.5641896 * y
        a2 = 0.5 + pow(y, 2) + pow(y, 4)
        b2 = -1 + 2 * pow(y, 2)
        return (a1 + b1 * pow(x, 2)) / (a2 + b2 * pow(x, 2) + pow(x, 4))
    if 15 > abs(x) + y > 5.5:
        a3 = 1.05786 * y + 4.65456 * pow(y, 3) + 3.10304 * pow(y, 5) + 0.56419 * pow(y, 7)
        b3 = 2.962 * y + 0.56419 * pow(y, 3) + 1.69257 * pow(y, 5)
        c3 = 1.69257 * pow(y, 3) - 2.53885 * y
        d3 = 0.56419 * y
        a4 = 0.5625 + 4.5 * pow(y, 2) + 10.5 * pow(y, 4) + 6 * pow(y, 6) + pow(y, 8)
        b4 = -4.5 + 9 * pow(y, 2) + 6 * pow(y, 4) + 4 * pow(y, 6)
        c4 = 10.5 - 6 * pow(y, 2) + 6 * pow(y, 4)
        d4 = -6 + 4 * pow(y, 2)
        return (a3 + b3 * pow(x, 2) + c3 * pow(x, 4) + d3 * pow(x, 6)) / (
                a4 + b4 * pow(x, 2) + c4 * pow(x, 4) + d4 * pow(x, 6) + pow(x, 8))
    if abs(x) + y < 5.5 and y > 0.195 * abs(x) - 0.176:
        a5 = 272.102 + 973.778 * y + 1629.76 * pow(y, 2) + 1678.33 * pow(y, 3) + 1174.8 * pow(y, 4) + 581.746 * pow(y,
                                                                                                                    5) + 204.501 * pow(
            y, 6) + 49.5213 * pow(y, 7) + 7.55895 * pow(y, 8) + 0.564224 * pow(y, 9)
        b5 = -60.5644 - 2.34403 * y + 220.843 * pow(y, 2) + 336.364 * pow(y, 3) + 247.198 * pow(y, 4) + 100.705 * pow(y,
                                                                                                                      5) + 22.6778 * pow(
            y, 6) + 2.25689 * pow(y, 7)
        c5 = 4.58029 + 18.546 * y + 42.5683 * pow(y, 2) + 52.8454 * pow(y, 3) + 22.6798 * pow(y, 4) + 3.38534 * pow(y,
                                                                                                                    5)
        d5 = -0.128922 + 1.66203 * y + 7.56186 * pow(y, 2) + 2.25689 * pow(y, 3)
        e5 = 0.000971457 + 0.564224 * y
        a6 = 272.102 + 1280.83 * y + 2802.87 * pow(y, 2) + 3764.97 * pow(y, 3) + 3447.63 * pow(y, 4) + 2256.98 * pow(y,
                                                                                                                     5) + 1074.41 * pow(
            y, 6) + 369.199 * pow(y, 7) + 88.2674 * pow(y, 8) + 13.3988 * pow(y, 9) + pow(y, 10)
        b6 = 211.678 + 902.306 * y + 1758.34 * pow(y, 2) + 2037.31 * pow(y, 3) + 1549.68 * pow(y, 4) + 793.427 * pow(y,
                                                                                                                     5) + 266.299 * pow(
            y, 6) + 53.5952 * pow(y, 7) + 5 * pow(y, 8)
        c6 = 78.866 + 308.186 * y + 497.302 * pow(y, 2) + 479.258 * pow(y, 3) + 269.292 * pow(y, 4) + 80.3928 * pow(y,
                                                                                                                    5) + 10 * pow(
            y, 6)
        d6 = 22.0353 + 55.02931 * y + 92.7568 * pow(y, 2) + 53.5952 * pow(y, 3) + 10 * pow(y, 4)
        e6 = 1.49645 + 13.3988 * y + 5 * pow(y, 2)
        return (a5 + b5 * pow(x, 2) + c5 * pow(x, 4) + d5 * pow(x, 6) + e5 * pow(x, 8)) / (
                a6 + b6 * pow(x, 2) + c6 * pow(x, 4) + d6 * pow(x, 6) + e6 * pow(x, 8) + pow(x, 10))
    if abs(x) + y < 5.5 and y < 0.195 * abs(x) - 0.176:
        a7 = 1.16028e9 * y - 9.86604e8 * pow(y, 3) + 4.56662e8 * pow(y, 5) - 1.53575e8 * pow(y, 7) + 4.08168e7 * pow(y,
                                                                                                                     9) - 9.69463e6 * pow(
            y, 11) + 1.6841e6 * pow(y, 13) - 320772 * pow(y, 15) + 40649.2 * pow(y, 17) - 5860.68 * pow(y,
                                                                                                        19) + 571.687 * pow(
            y, 21) - 72.9359 * pow(y, 23) + 2.35944 * pow(y, 25) - 0.56419 * pow(y, 27)
        b7 = -5.60505e8 * y - 9.85386e8 * pow(y, 3) + 8.06985e8 * pow(y, 5) - 2.91876e8 * pow(y, 7) + 8.64829e7 * pow(y,
                                                                                                                      9) - 7.72359e6 * pow(
            y, 11) + 3.59915e6 * pow(y, 13) - 234417 * pow(y, 15) + 45251.3 * pow(y, 17) - 2269.19 * pow(y,
                                                                                                         19) - 234.143 * pow(
            y, 21) + 23.0312 * pow(y, 23) - 7.33447 * pow(y, 25)
        c7 = -6.51523e8 * y + 2.47157e8 * pow(y, 3) + 2.94262e8 * pow(y, 5) - 2.04467e8 * pow(y, 7) + 2.29302e7 * pow(y,
                                                                                                                      9) - 2.3818e7 * pow(
            y, 11) + 576054 * pow(y, 13) + 98079.1 * pow(y, 15) - 25338.3 * pow(y, 17) + 1097.77 * pow(y,
                                                                                                       19) + 97.6203 * pow(
            y, 21) - 44.0068 * pow(y, 23)
        d7 = -2.63894e8 * y + 2.70167e8 * pow(y, 3) - 9.96224e7 * pow(y, 5) - 4.15013e7 * pow(y, 7) + 3.83112e7 * pow(y,
                                                                                                                      9) + 2.2404e6 * pow(
            y, 11) - 303569 * pow(y, 13) - 66431.2 * pow(y, 15) + 8381.97 * pow(y, 17) + 228.563 * pow(y,
                                                                                                       19) - 161.358 * pow(
            y, 21)
        e7 = -6.31771e7 * y + 1.40677e8 * pow(y, 3) + 5.56965e6 * pow(y, 5) + 2.46201e7 * pow(y, 7) + 468142 * pow(y,
                                                                                                                   9) - 1.003e6 * pow(
            y, 11) - 66212.1 * pow(y, 13) + 23507.6 * pow(y, 15) + 296.38 * pow(y, 17) - 403.396 * pow(y, 19)
        f7 = - 1.69846e7 * y + 4.07382e6 * pow(y, 3) - 3.32896e7 * pow(y, 5) - 1.93114e6 * pow(y, 7) - 934717 * pow(y,
                                                                                                                    9) + 8820.94 * pow(
            y, 11) + 37544.8 * pow(y, 13) + 125.591 * pow(y, 15) - 726.113 * pow(y, 17)
        g7 = - 1.23165e6 * y + 7.52883e6 * pow(y, 3) - 900010 * pow(y, 5) - 186682 * pow(y, 7) + 79902.5 * pow(y,
                                                                                                               9) + 37371.9 * pow(
            y, 11) - 260.198 * pow(y, 13) - 968.15 * pow(y, 15)
        h7 = -610622 * y + 86407.6 * pow(y, 3) + 153468 * pow(y, 5) + 72520.9 * pow(y, 7) + 23137.1 * pow(y,
                                                                                                          9) - 571.645 * pow(
            y, 11) - 968.15 * pow(y, 13)
        o7 = -23586.5 * y + 49883.8 * pow(y, 3) + 26538.5 * pow(y, 5) + 8073.15 * pow(y, 7) - 575.164 * pow(y,
                                                                                                            9) - 726.113 * pow(
            y, 11)
        p7 = -8009.1 * y + 2198.86 * pow(y, 3) + 953.655 * pow(y, 5) - 352.467 * pow(y, 7) - 403.396 * pow(y, 9)
        q7 = -622.056 * y - 271.202 * pow(y, 3) - 134.792 * pow(y, 5) - 161.358 * pow(y, 7)
        r7 = - 77.0535 * y - 29.7896 * pow(y, 3) - 44.0068 * pow(y, 5)
        s7 = -2.92264 * y - 7.33447 * pow(y, 3)
        t7 = -0.56419 * y
        a8 = 1.02827e9 - 1.5599e9 * pow(y, 2) + 1.17022e9 * pow(y, 4) - 5.79099e8 * pow(y, 6) + 2.11107e8 * pow(y,
                                                                                                                8) - 6.11148e7 * pow(
            y, 10) + 1.44647e7 * pow(y, 12) - 2.85721e6 * pow(y, 14) + 483737 * pow(y, 16) - 70946.1 * pow(y,
                                                                                                           18) + 9504.65 * pow(
            y, 20) - 955.194 * pow(y, 22) + 126.532 * pow(y, 24) - 3.68288 * pow(y, 26) + pow(y, 28)
        b8 = 1.5599e9 - 2.28855e9 * pow(y, 2) + 1.66421e9 * pow(y, 4) - 7.53828e8 * pow(y, 6) + 2.89676e8 * pow(y,
                                                                                                                8) - 7.01358e7 * pow(
            y, 10) + 1.39465e7 * pow(y, 12) - 2.84954e6 * pow(y, 14) + 498334 * pow(y, 16) - 55600 * pow(y,
                                                                                                         18) + 3058.26 * pow(
            y, 20) + 533.254 * pow(y, 22) - 40.5117 * pow(y, 24) + 14 * pow(y, 26)
        c8 = 1.17022e9 - 1.66421e9 * pow(y, 2) + 1.06002e9 * pow(y, 4) - 6.60078e8 * pow(y, 6) + 6.33496e7 * pow(y,
                                                                                                                 8) - 4.60396e7 * pow(
            y, 10) + 1.4841e7 * pow(y, 12) - 1.06352e6 * pow(y, 14) - 217801 * pow(y, 16) + 48153.3 * pow(y,
                                                                                                          18) - 1500.17 * pow(
            y, 20) - 198.876 * pow(y, 22) + 91 * pow(y, 24)
        d8 = 5.79099e8 - 7.53828e8 * pow(y, 2) + 6.60078e8 * pow(y, 4) + 5.40367e7 * pow(y, 6) + 1.99846e8 * pow(y,
                                                                                                                 8) - 6.87656e6 * pow(
            y, 10) - 6.89002e6 * pow(y, 12) + 280428 * pow(y, 14) + 161461 * pow(y, 16) - 16493.7 * pow(y,
                                                                                                        18) - 567.164 * pow(
            y, 20) + 364 * pow(y, 22)
        e8 = 2.11107e8 - 2.89676e8 * pow(y, 2) + 6.33496e7 * pow(y, 4) - 1.99846e8 * pow(y, 6) - 5.01017e7 * pow(y,
                                                                                                                 8) - 5.25722e6 * pow(
            y, 10) + 1.9547e6 * pow(y, 12) + 240373 * pow(y, 14) - 55582 * pow(y, 16) - 1012.79 * pow(y,
                                                                                                      18) + 1001 * pow(
            y, 20)
        f8 = 6.11148e7 - 7.01358e7 * pow(y, 2) + 4.60396e7 * pow(y, 4) - 6.87656e6 * pow(y, 6) + 5.25722e6 * pow(y,
                                                                                                                 8) + 3.04316e6 * pow(
            y, 10) + 123052 * pow(y, 12) - 106663 * pow(y, 14) - 1093.82 * pow(y, 16) + 2002 * pow(y, 18)
        g8 = 1.44647e7 - 1.39465e7 * pow(y, 2) + 1.4841e7 * pow(y, 4) + 6.89002e6 * pow(y, 6) + 1.9547e6 * pow(y,
                                                                                                               8) - 123052 * pow(
            y, 10) - 131337 * pow(y, 12) - 486.14 * pow(y, 14) + 3003 * pow(y, 16)
        h8 = 2.85721e6 - 2.84954e6 * pow(y, 2) + 1.06352e6 * pow(y, 4) + 280428 * pow(y, 6) - 240373 * pow(y,
                                                                                                           8) - 106663 * pow(
            y, 10) + 486.14 * pow(y, 12) + 3432 * pow(y, 14)
        o8 = 483737 - 498334 * pow(y, 2) - 217801 * pow(y, 4) - 161461 * pow(y, 6) - 55582 * pow(y, 8) + 1093.82 * pow(
            y, 10) + 3003 * pow(y, 12)
        p8 = 70946.1 - 55600 * pow(y, 2) - 48153.3 * pow(y, 4) - 16493.7 * pow(y, 6) + 1012.79 * pow(y, 8) + 2002 * pow(
            y, 10)
        q8 = 9504.65 - 3058.26 * pow(y, 2) - 1500.17 * pow(y, 4) + 567.164 * pow(y, 6) + 1001 * pow(y, 8)
        r8 = 955.194 + 533.254 * pow(y, 2) + 198.876 * pow(y, 4) + 364 * pow(y, 6)
        s8 = 126.532 + 40.5117 * pow(y, 2) + 91 * pow(y, 4)
        t8 = 3.68288 + 14 * pow(y, 2)
        return math.exp(pow(y, 2) - pow(x, 2)) * math.cos(2 * x * y) - (
                a7 + b7 * pow(x, 2) + c7 * pow(x, 4) + d7 * pow(x, 6) + e7 * pow(x, 8) + f7 * pow(x, 10) + g7 * pow(
            x, 12) + h7 * pow(x, 14) + o7 * pow(x, 16) + p7 * pow(x, 18) + q7 * pow(x, 20) + r7 * pow(x,
                                                                                                      22) + s7 * pow(
            x, 24) + t7 * pow(x, 26)) / (
                a8 + b8 * pow(x, 2) + c8 * pow(x, 4) + d8 * pow(x, 6) + e8 * pow(x, 8) + f8 * pow(x, 10) + g8 * pow(x,
                                                                                                                    12) + h8 * pow(
            x, 14) + o8 * pow(x, 16) + p8 * pow(x, 18) + q8 * pow(x, 20) + r8 * pow(x, 22) + s8 * pow(x, 24) + t8 * pow(
            x, 26) + pow(x, 28))


@cuda.jit
def cuLBL_calcu(Sw, Gamma_D, Gamma_L, Nu, wave_start, wave_end, wing_cut, sampling_delta, unit_k, sample_point,
                absor_coef, coef):
    """
    CUDA kernel function
    :param Sw: Temperature-corrected line strength
    :param Gamma_D: Doppler HWHM
    :param Gamma_L: Lorentz HWHM
    :param Nu: Spectral line wavenumber
    :param wave_start: Band start wavenumber (cm^-1)
    :param wave_end: Band end wavenumber (cm^-1)
    :param wing_cut: Line wing cutoff (cm^-1)
    :param sampling_delta: Sampling interval (cm^-1)
    :param unit_k: Volume concentration factor
    :param sample_point: Sampling points
    :param absor_coef: Output absorption coefficient placeholder, do not remove or a bug occurs
    """
    ib = cuda.blockIdx.x
    it = cuda.threadIdx.x
    ibd = cuda.blockDim.x
    igd = cuda.gridDim.x
    idxWithinGrid = it + ib * ibd
    gridStride = igd * ibd
    for idx in range(idxWithinGrid, len(sample_point), gridStride):
        # sample_point[idx] = wave_start + idx * sampling_delta
        # if sample_point[idx] <= wave_end:
        for i in range(len(Nu)):
            delta_nu = sample_point[idx] - Nu[i]
            if abs(delta_nu) < wing_cut:
                coef[idx] += unit_k * Sw[i] * voigt_profile(delta_nu, Gamma_D[i], Gamma_L[i])


if __name__ == '__main__':
    begin_time = time()
    # CPU
    db_begin('data')

    molec_id = 2
    local_iso_id = 1
    table_name = 'CO2_1900_3500'
    M = molecularMass(molec_id, local_iso_id)

    wave_start = 2000
    wave_end = 3333
    sampling_delta = 0.01
    wing_cut = 25

    sampling_num = int((wave_end - wave_start) / sampling_delta) + 1

    (Nu, sw, gamma_air, gamma_self, n_air, elower) = getColumns(table_name,
                                                                ['nu',
                                                                 'sw', 'gamma_air', 'gamma_self',
                                                                 'n_air', 'elower'])

    sample_point = np.linspace(wave_start, wave_end, sampling_num)
    absor_coef = np.zeros(sampling_num)

    # GPU
    Nu_gpu = cuda.to_device(Nu)
    sample_point_gpu = cuda.to_device(sample_point)
    absor_coef_gpu = cuda.to_device(absor_coef)

    threads_per_block = 32
    blocks_per_grid = int((sampling_num + threads_per_block - 1) / threads_per_block)
    grid_stride = 1
    blocks_per_grid = int((blocks_per_grid + grid_stride - 1) / grid_stride)

    data_prepare_time = time()
    print(f"data prepare cost: {data_prepare_time - begin_time}")

    P_range = np.array([1.9,1.4,1.0,0.7,0.5,0.4,0.4,0.3,0.3,0.2])
    T_range =  np.array([386,1104,583,472,332,839,750,313,294,276])

    nP = len(P_range)
    nT = len(T_range)

    nN = nP

    Pdata = np.zeros(nN)
    Tdata = np.zeros(nN)
    Coef_data = np.zeros((nN, sampling_num))
    cost_time = np.zeros(nN)
    for i in range(nP):
        gpu_calcu_start_time = time()
        P = P_range[i]
        Pdata[i] = P
        T = T_range[i]
        Tdata[i] = T

        unit_k = volumeConcentration(P, T)

        Sw = sw_pt(sw, T, Nu, elower, molec_id, local_iso_id)
        Gamma_D = doppler_hwhm(Nu, T, M)
        Gamma_L = lorentz_hwhm(P, T, gamma_air, gamma_self, n_air, P)
        Sw_gpu = cuda.to_device(Sw)
        Gamma_D_gpu = cuda.to_device(Gamma_D)
        Gamma_L_gpu = cuda.to_device(Gamma_L)
        coef_gpu = cuda.to_device(absor_coef)

        cuLBL_calcu[blocks_per_grid, threads_per_block](Sw_gpu, Gamma_D_gpu, Gamma_L_gpu, Nu_gpu, wave_start,
                                                        wave_end, wing_cut,
                                                        sampling_delta, unit_k, sample_point_gpu, absor_coef_gpu,
                                                        coef_gpu)
        cuda.synchronize()
        output_coef = coef_gpu.copy_to_host()
        Coef_data[i, :] = output_coef
        gpu_calcu_end_time = time()
        cost_time[i] = gpu_calcu_end_time - gpu_calcu_start_time
        print(f"GPU run {i} cost: {cost_time[i]}")

    savemat("a2509_cuUSAR_unpath_cuLBL.mat", {'Pc': Pdata, 'Tc': Tdata, 'Coef': Coef_data})

    print(f"GPU run average cost: {np.mean(cost_time)}")
    end_time = time()
    print(f"total cost: {end_time - begin_time}")

    # savemat('data_coef2.mat',{'Pc': Pdata, 'Tc': Tdata, 'Coef': Coef_data})

    # np.save("P-max.npy", Pdata)
    # np.save("T3.npy", Tdata)
    # np.save("Trans3.npy", Trans_data)

    # figure
    # nnn = np.load("nu2000-3333.npy")
    # ccc = np.load("coef2000-3333.npy")
    # plb.figure()
    # plb.plot(sample_point, output_coef, 'r--', label='GPU', alpha=0.7)
    # plb.plot(nnn, ccc, 'b:', label='Ref', alpha=0.7)
    # plb.xlabel('wavenumber($cm^{-1}$)')
    # plb.ylabel('absorption coefficient($cm^{-1}$)')
    # plb.title('$CO_2$ absorption coefficient @ 0.1atm, 210K')
    # plb.figure()
    # plb.plot(sample_point, (output_coef - ccc), 'r', label='error', alpha=0.7)
    # plb.show()
