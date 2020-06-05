import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import astropy.io as ap

obs_map_file = "/Users/yusuke/program/py_program/CMB/skymap/data/LFI_SkyMap_030-BPassCorrected_0256_R2.01_full.fits"
I_planck = hp.read_map(obs_map_file, field=(0,1,2), dtype=np.float32)#PlanckのRING型データ

sp = []
I = []
Q = []
V = []
U = []

dir = "/Users/yusuke/program/py_program/CMB/skymap/data/output/"
N = 15
for i in range(N):
    sp.append(np.load(dir+"sp_{}.npz".format(i)))
    I.append(np.array([sp[i]["I"]]))
    Q.append(np.array([sp[i]["Q"]]))
    V.append(np.array([sp[i]["V"]]))
    U.append(np.array([sp[i]["U"]]))


I_stokes = np.concatenate([I[i] for i in range(N)],1)[0]
Q_stokes = np.concatenate([Q[i] for i in range(N)],1)[0]
V_stokes = np.concatenate([V[i] for i in range(N)],1)[0]
U_stokes = np.concatenate([U[i] for i in range(N)],1)[0]

I_delta = I_stokes - I_planck[0]
Q_delta = Q_stokes - I_planck[1]
U_delta = U_stokes - I_planck[2]

plt.figure()
plt.title("I_delta histogram")
plt.ylabel("Number of pixel")
plt.xlabel("K")
plt.hist(I_delta,bins=100, range=(-1e-16,1e-16), log=True, histtype="step")
plt.grid()

plt.figure()
plt.title("Q_delta histogram")
plt.ylabel("Number of pixel")
plt.xlabel("K")
plt.hist(Q_delta,bins=100, range=(-1e-16,1e-16), log=True, histtype="step")
plt.grid()

plt.figure()
plt.title("U_delta histogram")
plt.ylabel("Number of pixel")
plt.xlabel("K")
plt.hist(U_delta,bins=100, range=(-1e-16,1e-16), log=True, histtype="step")
plt.grid()


hp.mollview(I_stokes, title="I_STOKES observed by LiteBIRD", unit="mK", cmap="jet")
hp.mollview(I_planck[0], title="I_STOKES observed by Planck", unit="mK", cmap="jet")
hp.mollview(Q_stokes, title="Q_STOKES observed by LiteBIRD", unit="mK", cmap="jet")
hp.mollview(I_planck[1], title="Q_STOKES observed by Planck", unit="mK", cmap="jet")
hp.mollview(U_stokes, title="U_STOKES observed by LiteBIRD", unit="mK", cmap="jet")
hp.mollview(I_planck[2], title="U_STOKES observed by Planck", unit="mK", cmap="jet")
