print("import now...")
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import astropy.io as ap
import time
from joblib import Parallel, delayed

def stokes(i, I_obs):#iには0からNPIXまでのintをいれる
    t = np.where(pix == i)[0]#i番目のPixcelを踏んだときの時間がtに配列ではいる．
    times = t.size
    psi = Psi[t]
    w_i = (1/2)*np.array([np.ones(times),np.sin(2*psi),np.zeros(times),np.cos(2*psi)])
    p_i = w_i[0]*I_obs[0][t]+w_i[1]*I_obs[1][t]+w_i[3]*I_obs[2][t]

    D_mat = (1/4)*np.array([
            [np.ones(times), np.sin(2*psi),np.zeros(times),np.cos(2*psi)],
            [np.sin(2*psi),np.sin(2*psi)**2,np.zeros(times),np.cos(2*psi)*np.sin(2*psi)],
            [np.zeros(times),np.zeros(times),np.ones(times),np.zeros(times)],
            [np.cos(2*psi),np.cos(2*psi)*np.sin(2*psi),np.zeros(times),np.cos(2*psi)*np.cos(2*psi)]
            ])
    sum_D_inv = np.linalg.inv(D_mat.sum(2))
    D_E = np.dot(D_mat.sum(2), sum_D_inv)
    wp = w_i*p_i
    #print("w_i=",w_i)
    #print("p_i=",p_i)
    #print("wp =",wp)
    sum_wp = wp.sum(1)
    s_tilde = np.dot(sum_D_inv, sum_wp)
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
#orbit = hp.vec2ang(orbit)

orbit_file = "/Users/yusuke/program/py_program/CMB/skymap/orbit_data/orbit_angle.npz"
orbit = np.load(orbit_file)
orbit = np.array([orbit["theta"][:times], orbit["phi"][:times]])
orbit

dif = np.diff(orbit)#軌道ベクトルの移動方向
n_vec = np.array([np.ones(times-1)*(-1e-3),np.zeros(times-1)])#θが下向きの方向

inner = (dif*n_vec).T.sum(1)
L_dif = np.sqrt(dif[0]**2 + dif[1]**2)
L_n = np.sqrt(n_vec[0]**2 + n_vec[1]**2)
judge = np.sign(dif[0]*n_vec[1]-dif[1]*n_vec[0])#角度psiの正負は外積で判断
cos_psi = inner/(L_dif*L_n)
Psi = np.rad2deg(np.arccos(cos_psi))*judge

pix = hp.ang2pix(NSIDE,orbit[0],orbit[1])

hit_pix, bins = np.histogram(pix,bins=NPIX)
"""
I_lu = np.zeros(NPIX)
for i in range(times):
    for j in range(len(a)):
        if i == a[j]:
            I_lu[pix[i]] += 10#a[s]の時は+10
    I_lu[pix[i]] += 1
"""

"""Planckのマップをreadして解析する"""
print("Reading Planck data...")
file_path = "/Users/yusuke/program/py_program/CMB/skymap/data/LFI_SkyMap_030-BPassCorrected_0256_R2.01_full.fits"
I_planck = hp.read_map(file_path, field = (0,1,2), dtype=np.float32)#PlanckのRING型データ
I_obs = [I_planck[0][pix], I_planck[1][pix], I_planck[2][pix]]#Planckのデータを観測されるpix順に並び換えた時系列観測データI_obs

npix_array=np.arange(0,NPIX)
npix_split=np.array_split(npix_array, 10)

s_par = np.array(Parallel(n_jobs=-1, verbose=5)( [delayed(stokes)(i,I_obs) for i in n_s[0] ])).T

Runtime = time.time() - start
print("\nFinish!")
print ("Runtime: {0}".format(Runtime) + "[sec]")

print ("Now, data saving...")
np.savez_compressed("stokes_parameter.npz", I=s_par[0], Q=s_par[1], V=s_par[2], U=s_par[3])
print ("Complete...")

""""""
#s = np.load("stokes_parameter.npz")
#s = np.array([s["I"], s["Q"], s["V"], s["U"]])
zero = np.zeros(NPIX-n)
map = np.concatenate([s_par[0], zero])
""""""
print ("Show plot!")
hp.mollview(hit_pix, title="Hit count map in Ecliptic coordinates", unit="Hit number")
hp.mollview(I_planck[0], title="I_STOKES observed by Planck", unit="mK", cmap="jet")
hp.mollview(map, title="LiteBIRD observation {:.4}-days".format(times/(day+1)), unit="mK",norm="hist",cmap="jet")
plt.show()
