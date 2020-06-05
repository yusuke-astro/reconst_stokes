"""
KEKで計算した再構成したストークパラメータについてloadして解析を行う
"""
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import astropy.io as ap
import time
from joblib import Parallel, delayed

NSIDE = 1024
NPIX = hp.nside2npix(NSIDE)
day = 60*60*24#1日の秒数
year = day*365
day*30
times = year +1
time_array = np.arange(0,times,1)
times

start = time.time()
s = np.load("stokes_vector_KEK.npz")
sp = np.array([s["I"],s["Q"],s["V"],s["U"]])
sp[0][513]
hp.mollview(sp[0], title="LiteBIRD observation {:.4}-days".format(times/(day+1)), unit="mK",cmap="jet")
