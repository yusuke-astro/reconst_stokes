"""
regen01.pyからの変更点．
・np.loadで配列読み込み

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
from joblib import Parallel, delayed
import func_stokes as fs

NSIDE = 256
NPIX = hp.nside2npix(NSIDE)

day = 60*60*24#1日の秒数
year = day*365
times = year*3 +1
time_array = np.arange(0,times,1)
times

orbit_file = "orbit_angle.npz"

#orbit_fileからtimes秒まで読み込み，times秒間フライトしたデータを取得する．
orbit = np.load(orbit_file)
orbit = np.array([orbit["theta"][:times], orbit["phi"][:times]])

pix = hp.ang2pix(NSIDE,orbit[0],orbit[1])
pix
#ヒストグラムの処理は長い
#hit_pix, bins = np.histogram(pix,bins=NPIX)

"""Planckのマップをreadして解析する"""
file_path = "COM_CompMap_CMB-commrul_0256_R1.00.fits"
I_planck = hp.read_map(file_path,field = 0)#PlanckのRING型データ
I_obs = I_planck[pix]#Planckのデータを観測されるpix順に並び換えた時系列観測データI_obs
I_obs

I_map = np.zeros(NPIX)
I_map[pix[:]] = I_planck[pix]
start = time.time()
#0~NPIX番目までのピクセつがもつストークスパラメータが入る．
#例えば，STOK[0]は0番目のピクセルのストークスパラメータ
n = 100
S_para = fs.parallel(pix, I_obs)
np.savez_compressed("stokes_parameter.npz", I=S_para[0], Q=S_para[1], V=S_para[2], U=S_para[3])
zero = np.zeros(NPIX-n)
print(S_para[0])
elapsed_time0 = time.time() - start
print ("\n計算時間: {0}".format(elapsed_time0) + "[sec]")
#map = np.concatenate([S_para[0], zero])

#hit_time(0)
#pix_hit_timing = Parallel(n_jobs=-1, verbose=5, backend="threading")( [delayed(hit_time)(i) for i in range(NPIX)] )
#pix_hit_timing


#hp.mollview(hit_pix, title="Hit count map in Ecliptic coordinates", unit="Hit number")
#hp.mollview(I_planck, title="I_STOKES observed by Planck", unit="mK",norm="hist", cmap="jet")
#hp.mollview(map, title="LiteBIRD observation {:.4}-days".format(times/(day+1)), unit="mK",norm="hist",cmap="jet")
#plt.show()
