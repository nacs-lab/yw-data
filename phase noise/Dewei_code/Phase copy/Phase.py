import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Unitary evolution without detune
def generate_U(omega, phi, dt):
    omega = omega / 2
    return np.array([[np.cos(omega * dt), -1j * np.exp(1j * phi) * np.sin(omega * dt)],
                     [-1j * np.exp(-1j * phi) * np.sin(omega * dt), np.cos(omega * dt)]])


# Unitary evolution with detune
def generate_U_detune(omega, phi, delta, dt):
    omega1 = np.sqrt(omega ** 2 + delta ** 2)  # Altered Omega
    u11 = np.cos(omega1 * dt / 2) + 1j * delta * np.sin(omega1 * dt / 2) / omega1
    u12 = -1j * np.exp(1j * phi) * np.sin(omega1 * dt / 2) * omega / omega1
    u21 = -1j * np.exp(-1j * phi) * np.sin(omega1 * dt / 2) * omega / omega1
    u22 = np.cos(omega1 * dt / 2) - 1j * delta * np.sin(omega1 * dt / 2) / omega1
    return np.array([[u11, u12], [u21, u22]])


# Setting rabi frequency
Omega0 = 1.25e6 * 2
# Setting Monte-Carlo realization times
realization = 50
# Setting noise
# fluctuation of detune: equals to k_eff * delta_v
w1 = 196e3
# importing phase noise data
data_exp = np.array(np.loadtxt('./PHASE1_Node_14 (1).csv', delimiter=',', skiprows=58, usecols=[0, 1]))
f = data_exp[:, 0]
applyFilter = True;

# using phase noise data to generate discrete time sequence
N = len(f)
t_list = np.arange(0, 1 / (f[1] - f[0]) + 1e-10, 1 / (f[-1] - f[0]))
# T = t[-1]
t_spacing = t_list[1] - t_list[0]
# t_list = np.arange(0, T, t_spacing)
dim = len(t_list)

# Generating Phase noise in time series
P_dbm = data_exp[:, 1]  # AC noise in dbm
P_w = [10 ** ((x - 30) / 10) for x in P_dbm]  # AC noise in W
Err_u = [2 * 50 * x for x in P_w]  # AC noise(amplitude square)
Err_f = [x * (1000 / 0.044) ** 2 for x in Err_u]  # laser frequency noise in Hz^2
if applyFilter:
    Err_f = [Err_f[i] / (1 + 4 * (f[i] / (10 ** 5)) ** 2) for i in range(N)]  # Apply filter function
Err_phi = [Err_f[i] / f[i] ** 2 for i in range(N)]  # laser phase noise

# Plot Err_u with f
plt.plot(f, Err_f)
plt.xlabel('frequency (Hz)')
plt.ylabel('frequency noise (Hz^2)')
plt.show()

# plot phase noise vs f
plt.yscale("log")
plt.plot(f, Err_phi)
plt.ylabel("phase noise")
plt.xlabel("frequency (Hz)")
plt.grid()
plt.show()

Err_t = np.zeros(dim)  # correlator
for j in range(dim):
    Err_t[j] = sum([Err_phi[i] * np.cos(2 * np.pi * f[i] * t_list[j]) for i in range(N)]) * (f[1] - f[0])
    # Err_t[j] = sum([Err_phi[i] * np.cos(2*np.pi*f[i] * t[j]) for i in range(N)])*10000

# Making second half to be zero
Err_t[int(len(Err_t) / 2):] = 0
# Plot Err with t
plt.plot(t_list, Err_f)
plt.xlabel('time (s)')
plt.ylabel('Err_t (arb)')
plt.show()

# Calculating covariance
cov = np.zeros([dim, dim])
for i in range(dim):
    for j in range(dim):
        cov[i, j] = Err_t[abs(i - j)] * 2 * np.pi
        # cov[i,j] = func_correlation(abs(i-j)*dt,w0,tau)

# logging state vector
time_trace = np.zeros_like(t_list)
time_trace_err = np.zeros_like(t_list)
for rl in range(realization):
    phi_list = np.random.multivariate_normal([0] * dim, cov)
    delta = np.random.normal(0, w1)
    plt.plot(t_list * 1e6, phi_list)
    state = np.array([1., 0.])
    Omega = Omega0
    for i_phi, phi in enumerate(phi_list):
        time_trace[i_phi] += abs(state[0]) ** 2
        time_trace_err[i_phi] += abs(state[0]) ** 4
        state = generate_U_detune(Omega, phi, delta, t_spacing).dot(state)


time_trace_err -= time_trace ** 2 / realization
time_trace_err[np.where(time_trace_err <= 0)[0]] = 0
time_trace_err = time_trace_err ** 0.5 / realization

time_trace_cut = time_trace[:int(len(Err_t) / 2)]
yerr = time_trace_err[:int(len(Err_t) / 2)]
t_list_cut = t_list[:int(len(Err_t) / 2)]

# print time_trace_err
time_trace /= realization
plt.xlabel("time(us)")
plt.ylabel("phase noise")
plt.title("Typical traces of phase noise")
plt.show()

plt.errorbar(1e6 * t_list_cut, time_trace_cut, yerr, ls='-', marker='.')
plt.xlabel("Time(us)")
plt.ylabel("Survival")
plt.grid()
plt.show()


def fit_func(x, a, b, c, d):
    return a * np.cos(2 * b * x) * np.exp(-x / c) / 2 + d


# fit with exp*cos
popt, pcov = curve_fit(fit_func, 1e6 * t_list_cut, time_trace_cut, bounds=([0.4, 0, 0, 0], [1, 3, 10000, 1]))
# print(params[0])
plt.plot(1e6 * t_list_cut, fit_func(1e6 * t_list_cut, *popt), 'g--',
         label='contrast=%5.3f, Omega=%5.3f(MHz),\n tau=%5.3f(us),offset=%5.3f' % tuple(popt))
plt.legend(loc="lower right")
plt.errorbar(1e6 * t_list_cut, time_trace_cut, yerr, marker='.', linestyle='none')
plt.xlabel("Time(us)")
plt.ylabel("Simulated Survival")
plt.grid()
plt.show()
print(np.sqrt(np.diag(pcov)))
