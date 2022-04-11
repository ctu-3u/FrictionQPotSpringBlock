import os

import FrictionQPotSpringBlock.Line1d as model
import h5py
import numpy as np
import prrng
import tqdm

import matplotlib.pyplot as plt

N = 1000

initstate = np.arange(N)
initseq = np.zeros(N)
generators = prrng.pcg32_array(initstate, initseq)

nchunk = 1500
nbuffer = 200
state = generators.state()

y = 2.0 * generators.random([nchunk])
y = np.cumsum(y, 1)
y -= 50.0

xdelta = 1e-3

system = model.System_thermal(
    tmp = 0,
    m=1.0,
    eta=2.0 * np.sqrt(3.0) / 10.0,
    mu=1.0,
    k_neighbours=1.0,
    k_frame=1.0 / N,
    dt=0.1,
    x_yield=y,
)

istart = system.m_model.istart()