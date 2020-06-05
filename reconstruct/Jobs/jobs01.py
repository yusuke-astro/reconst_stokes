print("import now...")
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import astropy.io as ap
import time
from joblib import Parallel, delayed
import sys

#number = sys.argv
number = 0#int(number[1])
print("number =",number)

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
    sum_wp = wp.sum(1)
    s_tilde = np.dot(sum_D_inv, sum_wp)
    return s_tilde#0~NPIX番目までのピクセルに対するストークスパラメータが入る．

start = time.time()
NSIDE = 256
NPIX = hp.nside2npix(NSIDE)
day = 60*60*24#1日の秒数
year = day*365
times = year+1#fループの数合わせで+1
time_array = np.arange(0,times+100,1)#軌道は参照バグ回避のために100秒多めに求めておく

print("Calcurate orbit...")
orb_Nyear = spin_prec(time_array)#長めに軌道を計算する．
orb_Nyear = hp.vec2ang(orb_Nyear)

#orbit_file = "/Users/yusuke/program/py_program/CMB/skymap/orbit_data/orbit_angle.npz"
#orbit = np.load(orbit_file)
#orbit = np.array([orbit["theta"][:times], orbit["phi"][:times]])

#LiteBIRDとsky-axisのなす角の計算
dif = np.diff(orb_Nyear)#軌道ベクトルの移動方向
n_vec = np.array([np.ones(len(dif[0]))*(-1e-3),np.zeros(len(dif[0]))])#θが下向きの方向
inner = (dif*n_vec).T.sum(1)
L_dif = np.sqrt(dif[0]**2 + dif[1]**2)
L_n = np.sqrt(n_vec[0]**2 + n_vec[1]**2)
judge = np.sign(dif[0]*n_vec[1]-dif[1]*n_vec[0])#角度psiの正負は外積で判断
cos_psi = inner/(L_dif*L_n)
Psi = np.rad2deg(np.arccos(cos_psi))*judge#なす角

theta = orb_Nyear[0][:times]#実際のLiteBIRDのフライト時間に配列をスライス．timesはフライト時間
phi = orb_Nyear[1][:times]
pix = hp.ang2pix(NSIDE, theta, phi)
hit_pix, bins = np.histogram(pix,bins=NPIX)

print("Reading Planck data...")
file_path = "/Users/yusuke/program/py_program/CMB/skymap/data/LFI_SkyMap_030-BPassCorrected_0256_R2.01_full.fits"
I_planck = hp.read_map(file_path, field = (0,1,2), dtype=np.float32)#PlanckのRING型データ
I_obs = [I_planck[0][pix], I_planck[1][pix], I_planck[2][pix]]#Planckのデータを観測されるpix順に並び換えた時系列観測データI_obs

npix_array=np.arange(0,NPIX)
npix_split=np.array_split(npix_array, 10)
#s_par = np.array(Parallel(n_jobs=-1, verbose=5)( [delayed(stokes)(i,I_obs) for i in npix_split[number][0:10] ])).T
#s_par = np.array([stokes(i,I_obs) for i in range(507976, 550000) ]).T
s_par = np.array([stokes(i,I_obs) for i in npix_split[number][0:10] ]).T

Runtime = time.time() - start
print("\nFinish!")
print ("Runtime: {0}".format(Runtime) + "[sec]")

print ("Now, data saving...")
path = "/Users/yusuke/program/py_program/CMB/skymap/stokes/reconstruct/Jobs"
np.savez_compressed(path+"/output/sp_{}.npz".format(number), I=s_par[0], Q=s_par[1], V=s_par[2], U=s_par[3])
print ("Complete...")
s_par[0]
I_planck[0]
print ("Show plot!")
hp.mollview(hit_pix, title="Hit count map in Ecliptic coordinates", unit="Hit number")
#hp.mollview(I_planck[0], title="I_STOKES observed by Planck", unit="mK", cmap="jet")
#hp.mollview(map, title="LiteBIRD observation {:.4}-days".format(times/(day+1)), unit="mK",norm="hist",cmap="jet")
#plt.show()
print ("Shut down.")
