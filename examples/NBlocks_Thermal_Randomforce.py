import os

import FrictionQPotSpringBlock.Line1d as splib
import h5py
import numpy as np
import prrng
import tqdm

# import matplotlib.pyplot as plt

N = 1000

initstate = np.arange(N)
initseq = np.zeros(N)
generators = prrng.pcg32_array(initstate, initseq)

y = 2.0 * generators.random([20000])
y = np.cumsum(y, 1)
y -= 50.0

system = splib.System_thermal(
    tmp = 1,
    eta=2.0 * np.sqrt(3.0) / 10.0,
    mu=1.0,
    k_neighbours=1.0,
    k_frame=1.0 / N,
    dt=0.1,
    x_yield=y
)

ninc = 1000
ret_x_frame = np.empty([ninc], dtype=float)
ret_f_frame = np.empty([ninc], dtype=float)
ret_S = np.empty([ninc], dtype=int)

for inc in range(ninc):

    # Extract output data.
    i_n = system.m_model.i()

    # Apply event-driven protocol.
    if inc == 0:
        system.m_model.set_x_frame(0.0)  # initial quench
    else:
        system.GenerateThermalRandomForce();
        system.timeStep();

    # Extract output data.
    ret_x_frame[inc] = system.m_model.x_frame()
    ret_f_frame[inc] = np.mean(system.m_model.f_frame())
    ret_S[inc] = np.sum(system.m_model.i() - i_n)

fig, ax = plt.subplots()
ax.plot(ret_x_frame, ret_f_frame)
plt.show()