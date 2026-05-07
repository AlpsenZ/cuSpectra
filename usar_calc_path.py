# Ultrafast reconstruction of molecular absorption spectra
# Parallelized by wavenumber channels
from time import time
from numba import cuda
import scipy
import numpy as np
from scipy.interpolate import RegularGridInterpolator


@cuda.jit(device=True)
def surf_func(x, y, p):
    return p[0] + p[1] * x + p[2] * y + p[3] * x ** 2 + p[4] * x * y + p[5] * y ** 2 + p[6] * x ** 3 + p[
        7] * x ** 2 * y + p[8] * x * y ** 2 + p[9] * y ** 3 + p[10] * x ** 4 + p[11] * x ** 3 * y + p[
        12] * x ** 2 * y ** 2 + p[13] * x * y ** 3 + p[14] * y ** 4 + p[15] * x ** 5 + p[16] * x ** 4 * y + p[
        17] * x ** 3 * y ** 2 + p[18] * x ** 2 * y ** 3 + p[19] * x * y ** 4 + p[20] * y ** 5


@cuda.jit(device=True)
def surf_fit(P, T, norms, paras, ni):
    if 0.1 <= P <= 0.5 and 250 <= T <= 500:  # I
        x = (P - norms[0, 0]) / norms[0, 1]
        y = (T - norms[0, 2]) / norms[0, 3]
        p = paras[ni, 0:21]
    elif 0.1 <= P <= 0.5 and 1250 >= T > 500:  # II
        x = (P - norms[1, 0]) / norms[1, 1]
        y = (T - norms[1, 2]) / norms[1, 3]
        p = paras[ni, 21:42]
    elif 0.5 < P <= 2 and 1250 >= T > 500:  # III
        x = (P - norms[2, 0]) / norms[2, 1]
        y = (T - norms[2, 2]) / norms[2, 3]
        p = paras[ni, 42:63]
    elif 0.5 < P <= 2 and 250 <= T <= 500:  # IV
        x = (P - norms[3, 0]) / norms[3, 1]
        y = (T - norms[3, 2]) / norms[3, 3]
        p = paras[ni, 63:84]
    else:
        print('out of P/T range!')
        return 0
    return surf_func(x, y, p)


@cuda.jit
def gpu_fs(P, T, norms, paras, out_spec, N):
    idxWithinGrid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    gridStride = cuda.gridDim.x * cuda.blockDim.x
    for idx in range(idxWithinGrid, N, gridStride):
        for i in range(len(P)):
            out_spec[i,idx] = surf_fit(P[i], T[i], norms, paras, idx)


s_channel = 133301
pt_num = 10

data = scipy.io.loadmat('a2509_cuUSAR_paras_nonuniform.mat')
norms = data['norms']  # (4,4)
paras = data['parameters']  # (1334,84)
nu = data['nu'].squeeze()
P = np.array([1.9,1.4,1.0,0.7,0.5,0.4,0.35,0.3,0.25,0.2])
T = np.array([386,1104,583,472,332,839,577,313,957,276])
out_spec = np.zeros((len(P), s_channel))

norms_g = cuda.to_device(norms)
paras_g = cuda.to_device(paras)
out_g = cuda.to_device(out_spec)
P_g = cuda.to_device(P)
T_g = cuda.to_device(T)

threads_per_block = 32
blocks_per_grid = int((pt_num + threads_per_block - 1) / threads_per_block)
grid_stride = 1
blocks_per_grid = int((blocks_per_grid + grid_stride - 1) / grid_stride)

begin_time = time()
gpu_fs[blocks_per_grid, threads_per_block](P_g, T_g, norms_g, paras_g, out_g, s_channel)
print(f'cost{time() - begin_time}')
P = np.array([1.9,1.4,1.0,0.7,0.5,0.4,0.4,0.3,0.3,0.2])
T = np.array([386,1104,583,472,332,839,750,313,294,276])
P_g = cuda.to_device(P)
T_g = cuda.to_device(T)
out_g = cuda.to_device(np.zeros((len(P), s_channel)))
begin_time = time()
gpu_fs[blocks_per_grid, threads_per_block](P_g, T_g, norms_g, paras_g, out_g, s_channel)
cuda.synchronize()
print(f'cost{time() - begin_time}')
scipy.io.savemat("a2509_cuUSAR_unpath_cuUSAR.mat", {"out_spec": out_g.copy_to_host(), 'Px': P, 'Tx':T})
