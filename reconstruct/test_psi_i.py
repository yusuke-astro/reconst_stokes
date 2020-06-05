print("import now...")
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import astropy.io as ap
import time
from joblib import Parallel, delayed

def spin_prec(time):
    alpha = np.radians(45)
    beta = np.radians(50)
    omega_a = np.pi/30/192.348
    omega_b = 0.05*np.pi/30
    vect_prec = np.array([
        [np.cos(alpha)*np.cos(omega_a*time), -np.sin(omega_a*time), np.sin(alpha)*np.cos(omega_a*time)],
        [np.cos(alpha)*np.sin(omega_a*time), np.cos(omega_a*time), np.sin(alpha)*np.sin(omega_a*time)],
        [-np.sin(alpha), 0, np.cos(alpha)]
    ])
    vect_spin = np.array([np.sin(beta)*np.cos(omega_b*time), np.sin(beta)*np.sin(omega_b*time), np.cos(beta)])
    Vect = np.dot(vect_prec,vect_spin)

    omega_orbit = 2*np.pi/(60*60*24*365)
    vect_orbit = np.array([
        [np.cos(omega_orbit*time), 0, np.sin(omega_orbit*time)],
        [0, 1, 0],
        [-np.sin(omega_orbit*time), 0, np.cos(omega_orbit*time)]
    ])
    Vect = np.dot(vect_orbit,Vect)
    Vect = np.array([Vect[2],Vect[0],Vect[1]])
    return Vect.T

start = time.time()
NSIDE = 256
NPIX = hp.nside2npix(NSIDE)
NPIX
day = 60*60*24#1日の秒数
year = day*365
times = day+1
time_array = np.arange(0,times,1)

print("Calcurate orbit...")
orbit = spin_prec(time_array)
orbit = hp.vec2ang(orbit)

#orbit_file = "/Users/yusuke/program/py_program/CMB/skymap/orbit_data/orbit_angle.npz"
#orbit = np.load(orbit_file)
#orbit = np.array([orbit["theta"][:times], orbit["phi"][:times]])
#orbit = np.array([orbit["x"][:times+1], orbit["y"][:times+1], orbit["z"][:times+1]])

orbit

dif = np.diff(orbit)#軌道ベクトルの移動方向
dif
len(dif[0])
n_vec = np.array([np.ones(times-1)*(-1e-3),np.zeros(times-1)])#θが下向きの方向
len(n_vec[0])

inner = (dif*n_vec).T.sum(1)
inner
L_dif = np.sqrt(dif[0]**2 + dif[1]**2)
L_dif
L_n = np.sqrt(n_vec[0]**2 + n_vec[1]**2)
L_n

judge = np.sign(dif[0]*n_vec[1]-dif[1]*n_vec[0])#角度psiの正負は外積で判断
cos_psi = inner/(L_dif*L_n)
cos_psi
psi = np.rad2deg(np.arccos(cos_psi))*judge
psi
#psi[2461]
ang = 180
a = np.where((psi > ang-1) & (psi < ang+1))[0]#-1<ang<1の範囲に軌道ベクトルがきたときの時間[s]
a
pix = hp.ang2pix(NSIDE,orbit[0],orbit[1])
#pix_ = np.concatenate([pix, a,a,a,a,a,a])
#pix_
hit_pix, bins = np.histogram(pix,bins=NPIX)

I_lu = np.zeros(NPIX)
for i in range(times):
    for j in range(len(a)):
        if i == a[j]:
            I_lu[pix[i]] += 10#a[s]の時は+10
    I_lu[pix[i]] += 1
#I_lu[pix[2461]]

hp.mollview(I_lu, title="Hit count map in Ecliptic coordinates", unit="Hit number", cmap="jet")
hp.graticule()
plt.show()
