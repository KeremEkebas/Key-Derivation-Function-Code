import numpy as np
import matplotlib.pyplot as plt

N = 20001
u = np.linspace(0, 4*np.pi, N, endpoint=True)
A = 1 + 0.5*np.cos(u/2)
cosm = A * np.cos(u)

plt.figure(figsize=(10,4))
plt.plot(u, cosm, label="cos_m")
plt.plot(u, np.cos(u), "--", label="cos(u)")
plt.title("cos_m vs cos(u)")
plt.xlabel("u")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()


# fft kısmı
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq

N = 20001
u = np.linspace(0, 4*np.pi, N, endpoint=True)  # 0 → 4π
du = u[1] - u[0]

A = 1 + 0.5*np.cos(u/2)
cosm = A * np.cos(u)
f_cosm = cosm - np.mean(cosm)

spec = np.abs(rfft(f_cosm))
freqs = rfftfreq(N, d=du)  # units: 1/unit u
freqs_cycles_per_2pi = freqs * 2*np.pi  # normalize: cycles per 2π
spec_norm = spec / np.max(spec)

plt.figure(figsize=(8,3))
plt.plot(freqs_cycles_per_2pi, spec_norm)
plt.title("FFT of cos_m (Normalized)")
plt.xlabel("Harmonic index (cycles per 2π)")
plt.ylabel("Normalized amplitude")
plt.grid(True)
plt.xlim(0,3)  # İlk 3 harmonik
plt.tight_layout()
plt.show()


# faz dağılımı
import numpy as np
from numpy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

N = 20001
u = np.linspace(0, 4*np.pi, N, endpoint=True)
du = u[1] - u[0]

A = 1 + 0.5*np.cos(u/2)
cosm = A * np.cos(u)
f_cosm = cosm - np.mean(cosm)

spec_cosm = rfft(f_cosm)
freqs = rfftfreq(N, d=du)
freqs_cycles_per_2pi = freqs * 2*np.pi
phase_cosm = np.angle(spec_cosm)

plt.figure(figsize=(8,3))
plt.plot(freqs_cycles_per_2pi, phase_cosm)
plt.title("Phase Spectrum of cos_m")
plt.xlabel("Harmonic index (cycles per 2π)")
plt.ylabel("Phase (radians)")
plt.grid(True)
plt.xlim(0,3)
plt.tight_layout()
plt.show()
