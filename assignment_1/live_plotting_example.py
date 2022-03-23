import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def sine_curve(x, a, b, c, d):
    return a * np.sin(b * x + c) + d

target_kinetic_energy = 3.
current_step = 50

at, bt, ct, dt = 55., 1 / 25., -11.3, target_kinetic_energy + 70

print([at, bt, ct, dt])

x_long = np.arange(400)

kinetic_energy = sine_curve(x_long, at, bt, ct, dt)

rng = np.random.default_rng()
# noise = rng.normal(0, )



fig, ax = plt.subplots(nrows=1, ncols=1,
                       constrained_layout=True,
                       figsize=(7, 5))

p0 = [(np.max(kinetic_energy[:current_step])
      - np.min(kinetic_energy[:current_step]))/2,
      1 / np.abs(np.argmin(kinetic_energy[:current_step]) - np.argmax(kinetic_energy[:current_step])),
      np.argmax(np.abs(kinetic_energy[:current_step])) * 2.,
      np.mean(kinetic_energy[:current_step])]

print(p0)

para, __ = curve_fit(sine_curve,
                     np.arange(len(kinetic_energy[:current_step]), dtype=float),
                     kinetic_energy[:current_step],
                     p0=p0)

print(para)

estimated = sine_curve(np.arange(len(kinetic_energy[:current_step])),
                            *para)

extrapolated = sine_curve(np.arange(len(kinetic_energy[:current_step]) + 100),
                            *para)

para1, __ = curve_fit(sine_curve,
                      np.arange(40., len(kinetic_energy[:current_step]), dtype=float),
                      kinetic_energy[40:current_step],
                      p0=p0)

print(para1)

estimated1 = sine_curve(np.arange(len(kinetic_energy[:current_step])),
                            *para1)

extrapolated1 = sine_curve(np.arange(len(kinetic_energy[:current_step]) + 100),
                               *para1)

ax.plot(kinetic_energy[:current_step], c="black", label="actual")

ax.plot(estimated + 1, c="red", label="estimated")
ax.plot(extrapolated + 2, c="orange", label="extrapolated")

ax.plot(estimated1 - 1 , c="blue", label="estimated no start")
ax.plot(extrapolated1 - 2, c="lightblue", label="extrapolated no start")


ax.axhline(target_kinetic_energy, ls="dashed", c="gray")
ax.set_xlabel(r'$step$ [-]')
ax.set_ylabel(r'$E_{kin}$ [-]')
ax.set_title(f'Kinetic Energy')
ax.legend()
plt.show()