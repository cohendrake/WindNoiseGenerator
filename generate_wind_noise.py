import os
import numpy as np
import sys
from scipy.io import loadmat
from scipy.signal import hann, fftconvolve
import soundfile as sf
from multiprocessing import Pool
import random


class generate_wind_noise(object):
    def __init__(self, mat_file):
        param = loadmat(mat_file)
        self.exc_pulses = param["exc_pulses"]
        self.transition_mtx = dict()
        self.transition_mtx["gusts"] = param["transition_prob_gusts"]
        self.transition_mtx["const"] = param["transition_prob_const"]
        self.var_excitation_noise = 0.005874
        self.mean_state2 = 0.005
        self.mean_state3 = 0.25
        self.lpc_coeff = [2.4804, -2.0032, 0.5610, -0.0794, 0.0392]
        self.lpc_order = len(self.lpc_coeff)
        self.alpha1 = 0.15
        self.alpha2 = 0.5

    def generate(self, duration, fs, type="gusts"):
        L = int(fs * duration)
        excite_sig_noise = np.random.randn(L) *\
            np.sqrt(self.var_excitation_noise)
        transition_mtx_cum = np.cumsum(self.transition_mtx[type], axis=1)

        # generate state sequence
        state_act = 0
        states_syn = np.zeros(L)
        for k in range(self.lpc_order, L):
            x = np.random.rand(1)[0]
            p = transition_mtx_cum[state_act, :]
            if x < p[0]:
                state_act = 0
            elif p[0] < x <= p[1]:
                state_act = 1
            else:
                state_act = 2
            states_syn[k] = state_act

        # generate gains for long term behaviour from state sequence
        g_apl = np.zeros(L)
        g_apl[states_syn==1] = self.mean_state2
        g_apl[states_syn==2] = self.mean_state3

        # generate gins for short term behavior from random processes
        g_apl_ST = np.random.randn(L)

        # smoothing of gains implemented by hann filters
        win1 = hann(10000)
        win1 = win1 / np.sum(win1)
        g_apl = fftconvolve(g_apl, win1, mode="same")
        g_apl_LT = np.absolute(g_apl)

        win2 = hann(int(fs * 5e-2))
        win2 = win2 / np.sum(win2)
        g_apl_ST = fftconvolve(g_apl_ST, win2, mode="same")
        g_apl_ST = np.absolute(g_apl_ST)

        # Combine LT and ST characteristic by modulation of gains
        g_apl = g_apl_LT * g_apl_ST
        n = np.zeros(L)
        exc_L = 0
        idx_exc = 0
        for k in range(self.lpc_order, L):
            if states_syn[k] != 0:
                if idx_exc < (exc_L - 1):
                    idx_exc = idx_exc + 1
                else:
                    r_pulse = int(self.exc_pulses.shape[0] * np.random.rand(1)[0])
                    exc_L = int(self.exc_pulses[r_pulse, -1])
                    exc_pulse_cur = self.exc_pulses[r_pulse, 0:exc_L]
                    idx_exc = 0

            if states_syn[k] == 0:
                exc_sig = excite_sig_noise[k] / 2
            elif states_syn[k] == 1:
                exc_sig = self.alpha1 * exc_pulse_cur[idx_exc] + (1 - self.alpha1) * excite_sig_noise[k]
            else:
                exc_sig = self.alpha2 * exc_pulse_cur[idx_exc] + (1 - self.alpha2) * excite_sig_noise[k]

            n[k] = np.sum((g_apl[k] * exc_sig + np.flip(n[k-self.lpc_order:k], axis=0) +
                           g_apl_LT[k] * excite_sig_noise[k] * self.var_excitation_noise) * self.lpc_coeff)

        n = n / np.max(np.absolute(n)) * 0.95
        return n


def write_wind_noise(wind_generator, idx, fs, duration, wind_type, out_dir):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    n = wind_generator.generate(duration, fs, wind_type)
    out_file = os.path.join(
                out_dir,
                "{:s}_{:06d}.wav".format(wind_type, idx))
    sf.write(out_file, n, fs)
    return os.path.abspath(out_file)


if __name__ == "__main__":
    if len(sys.argv) == 7:
        mat_file, fs, duration, wind_type, numClip, out_dir = sys.argv[1:]
        fs = int(fs)
        duration = float(duration)
        numClip = int(numClip)
    else:
        print(f"Usage: {sys.argv[0]} mat_file fs duration wind_type numClip out_dir")
        exit(1)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    wind_generator = generate_wind_noise(mat_file)
    num_worker = 20
    pools = Pool(num_worker)
    res_list = []
    from tqdm import tqdm, trange
    for i in range(numClip):
        res = pools.apply_async(write_wind_noise, args=(wind_generator,
                                                        i,
                                                        fs,
                                                        duration,
                                                        wind_type,
                                                        out_dir))
        res_list.append(res)

    file_list = []
    for res in tqdm(res_list):
        file_list.append(res.get())
