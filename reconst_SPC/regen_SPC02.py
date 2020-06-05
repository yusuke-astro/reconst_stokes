"""
regen01.pyからの変更点．
・np.loadで配列読み込み

ストークスパラメータの再構成を行うプログラム．
Planckの観測データCOM_CMB_IQU-nilc_1024_R2.02_full.fitsのI成分をLiteBIRDのTrackで観測していく．
最後の観測データはI=[I_1,I_2,I_3,...I_NPIX]のピクセルに対するIの値とLiteBIRDの重複観測ピクセルI_lu=[,,,I_lu_NPIX]の積
I_observeで
"""
print("import now...")
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import astropy.io as ap
import time
import func_stokes as fs
from joblib import Parallel, delayed
from numpy import vectorize as vec

def detec(t, l):
    det = np.array([np.ones(l),np.sin(2*w*t), 0,np.cos(2*w*t)])
    return det

def mat(t,l):
    D_mat = np.array([
        [np.ones(l).sum(), np.sin(2*w*t).sum(),0,np.cos(2*w*t).sum()],
        [np.sin(2*w*t).sum(),(np.sin(2*w*t)**2).sum(),0,np.cos(2*w*t)*np.sin(2*w*t).sum()],
        [0,0,0,0],
        [np.cos(2*w*t).sum(),np.cos(2*w*t)*np.sin(2*w*t).sum(),0,np.cos(2*w*t)*np.cos(2*w*t).sum()]
        ])
    return D_mat


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

def stokes(i, I_obs):#iには0からNPIXまでのintをいれる
    t = np.where(pix == i)[0]#i番目のPixcelを踏んだときの時間がtに配列ではいる．
    times = t.size
    w = 2*np.pi*(0.3/60)
    det = np.array([np.ones(times),np.sin(2*w*t),np.zeros(times),np.cos(2*w*t)])

    D_mat = np.array([
            [np.ones(times), np.sin(2*w*t),np.zeros(times),np.cos(2*w*t)],
            [np.sin(2*w*t),np.sin(2*w*t)**2,np.zeros(times),np.cos(2*w*t)*np.sin(2*w*t)],
            [np.zeros(times),np.zeros(times),np.zeros(times),np.zeros(times)],
            [np.cos(2*w*t),np.cos(2*w*t)*np.sin(2*w*t),np.zeros(times),np.cos(2*w*t)*np.cos(2*w*t)]
            ])
    sum_D_pinv = np.linalg.pinv(D_mat.sum(2))
    D_E = np.dot(D_mat.sum(2), sum_D_pinv)
    wp = det*I_obs[t]
    sum_wp = wp.sum(1)
    s_tilde = np.dot(sum_D_pinv, sum_wp)
    return s_tilde#0~NPIX番目までのピクセルに対するストークスパラメータが入る．


start = time.time()
NSIDE = 256
NPIX = hp.nside2npix(NSIDE)
NPIX
day = 60*60*24#1日の秒数
year = day*365
times = year+1
time_array = np.arange(0,times,1)

print("Calcurate orbit...")
#orbit = spin_prec(time_array)
orbit_file = "/Users/yusuke/program/py_program/CMB/skymap/regenerate_s/orbit_angle.npz"
orbit = np.load(orbit_file)
orbit = np.array([orbit["theta"][:times], orbit["phi"][:times]])
#orbit = hp.vec2ang(orbit)

pix = hp.ang2pix(NSIDE,orbit[0],orbit[1])


#ヒストグラムの処理は長い
print("Calcurate orbit histogram...")
hit_pix, bins = np.histogram(pix,bins=NPIX)

"""Planckのマップをreadして解析する"""
print("Reading Planck data...")
file_path = "/Users/yusuke/program/py_program/CMB/skymap/data/LFI_SkyMap_030-BPassCorrected_0256_R2.01_full.fits"
I_planck = hp.read_map(file_path, field = 0, dtype=np.float32)#PlanckのRING型データ
I_obs = I_planck[pix]#Planckのデータを観測されるpix順に並び換えた時系列観測データI_obs
I_map = np.zeros(NPIX)
I_map[pix[:]] = I_planck[pix]
print("Complete...")
#0~NPIX番目までのピクセつがもつストークスパラメータが入る．
#例えば，STOK[0]は0番目のピクセルのストークスパラメータ

print("Recomporse stokes parameter...")
print("Start calcuration...")
""""""
n = 100
""""""
#s_par = np.array(fs.stokes_loop(pix, I_obs, n))

np.where(pix == 0)[0]

t = []
length = []
p_i = []
for i in range(2):
    #print(i)
    t.append(np.where(pix == i)[0])
    length.append(len(np.where(pix == i)[0]))
    p_i.append(I_obs[t[i]])

t = np.array(t)
l = np.array(length)
p_i = np.array(p_i)
p_i

w = 2*np.pi*(0.3/60)
l
t


f = np.vectorize(detec)
g = np.vectorize(mat)
det = f(t,l)
D_mat = g(t,l)
D_mat
det.sum()
p_i

sum_D_pinv = np.linalg.pinv(D_mat.sum(2))
D_E = np.dot(D_mat.sum(2), sum_D_pinv)
wp = det*I_obs[t]
sum_wp = wp.sum(1)
s_tilde = np.dot(sum_D_pinv, sum_wp)
#s_par = np.array(Parallel(n_jobs=-1, verbose=5, backend='multiprocessing')( [delayed(stokes)(i,I_obs) for i in range(n)] )).T

hp.mollview(hit_pix, title="Hit count map in Ecliptic coordinates", unit="Hit number")
hp.mollview(I_planck, title="I_STOKES observed by Planck", unit="mK",norm="hist", cmap="jet")
hp.mollview(map, title="LiteBIRD observation {:.4}-days".format(times/(day+1)), unit="mK",norm="hist",cmap="jet")
plt.show()
print ("Shutdown program.")
