"""
Pythonだけで高速化を行なったプログラム．
現状，Cythonや並列化よりも高速で，I_luに関するパッケージングループをnp.histogramで除去した．

軌道を予めnumpyのバイナリファイルに保存してloadすることで行列計算を回避．
3次元ベクトルよりも2次元の球座標で保存するほうがデータ量が2/3で済む．

ストークスパラメータの再構成を行うプログラム．
Planckの観測データCOM_CMB_IQU-nilc_1024_R2.02_full.fitsのI成分をLiteBIRDのTrackで観測していく．
最後の観測データはI=[I_1,I_2,I_3,...I_NPIX]のピクセルに対するIの値とLiteBIRDの重複観測ピクセルI_lu=[,,,I_lu_NPIX]の積
I_observeで
"""
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import astropy.io as ap
import time
start = time.time()

def create_mollweide_axes():
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection="mollweide")
    return ax

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

NSIDE = 256
NPIX = hp.nside2npix(NSIDE)
day = 60*60*24#1日の秒数
year = day*365
times = year*3

time_array = np.arange(0,times+1,1)
orbit = spin_prec(time_array)

"""vec or angle"""
orbit = hp.vec2ang(orbit)


file = "orbit_angle_test.npz"
"""save"""
np.savez_compressed(file, theta=orbit[0], phi=orbit[1])

"""load"""
orbit = np.load(file)
orbit = np.array([orbit["theta"],orbit["phi"]])


pix = hp.ang2pix(NSIDE,orbit[0],orbit[1])
I_lu, bins = np.histogram(pix,bins=NPIX)
max(I_lu)
elapsed_time = time.time() - start
print ("計算時間: {0}".format(elapsed_time) + "[sec]")
hp.mollview(I_lu, title="Hit count map in Ecliptic coordinates", unit="Hit number")
hp.graticule()

plt.show()
