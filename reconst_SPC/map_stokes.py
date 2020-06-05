import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import astropy.io as ap
import time
from joblib import Parallel, delayed

NSIDE = 256
NPIX = hp.nside2npix(NSIDE)
NPIX
day = 60*60*24#1日の秒数
year = day*365
day*30
times = year +1
time_array = np.arange(0,times,1)

#file_path1 = "/Users/yusuke/program/py_program/CMB/skymap/data/stokes_parameter0526.npz"
stokes_file = "/Users/yusuke/program/py_program/CMB/skymap/data/stokes_parameter0531.npz"
#stokes_file = "/Users/yusuke/program/py_program/CMB/skymap/data/stokes_param0531.npz"
s = np.load(stokes_file)
sp = np.array([s["I"],s["Q"],s["V"],s["U"]])


#file_path2 = "/Users/yusuke/program/py_program/CMB/skymap/regenerate_s/COM_CompMap_CMB-commrul_0256_R1.00.fits"
obs_map_file = "/Users/yusuke/program/py_program/CMB/skymap/data/LFI_SkyMap_030-BPassCorrected_0256_R2.01_full.fits"

I_planck = hp.read_map(obs_map_file, field=(0,1,2), dtype=np.float32)#PlanckのRING型データ
I_planck

I_planck[]
sp[0]
map_delta = sp[0]- I_planck[0]
map_delta



plt.figure()
plt.title("Delta hist by matplot")
plt.ylabel("Number of pixel")
plt.xlabel("K")
plt.hist(map_delta,bins=100,range=(-0.0025,0.0025), log=True, histtype="step")
plt.grid()

#zero = np.zeros(NPIX-1000)
#map = np.concatenate([sp[0], zero])
#print(sp[1])
#print(I_planck)

hp.mollview(I_planck[0], title="Planck law data plot {:.4}-days".format(times/(day+1)), unit="mK", cmap="jet", max=max(I_planck[0]),min=min(I_planck[0])/10)
hp.mollview(sp[0], title="LiteBIRD observation {:.4}-days".format(times/(day+1)), unit="mK",cmap="jet", max=max(sp[0]),min=min(sp[0])/10)
plt.show()
