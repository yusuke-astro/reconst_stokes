# %% codecell
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import math
import healpy as hp
from scipy.optimize import curve_fit
from astropy.io import fits
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

__all__ = ['CoordinateTransformation', 'Transformations', 'EpochPropagation']

from pygaia.astrometry.vectorastrometry import sphericalToCartesian, cartesianToSpherical, \
        elementaryRotationMatrix, normalTriad
from pygaia.utils import enum, degreesToRadians, radiansToDegrees
from pygaia.astrometry.constants import auKmYearPerSec

from numpy import ones_like, array, pi, cos, sin, zeros_like, zeros, arccos, select
from numpy import dot, transpose, cross, vstack, diag, sqrt, identity, tile, sum, arctan
from numpy import matmul, newaxis, squeeze
from numpy.linalg import norm
from scipy import isscalar


# %% codecell
def create_mollweide_axes():
    fig = plt.figure(figsize=(10,20))
    ax = fig.add_subplot(111, projection="mollweide")

    return ax


# %% codecell
tmps = 60*60*24*365
temps1 = 2*np.pi/600
temps2 = 2*np.pi/5760
temps3 = 2*np.pi/(365*24*60*60)
Vect_anti_sun = []
for t in range(tmps):
    Vect_anti_sun.append(temps3*t)

# %% codecell
#tourne autour de l
def rot_l(Vect, theta_t):
    mat_rot_l = np.array([
        [np.cos(theta_t), -np.sin(theta_t)],
        [np.sin(theta_t), np.cos(theta_t)]
    ])
    New_vect = mat_rot_l.dot(Vect)
    return New_vect

#tourne autour de 45,45
def rot_45(Vect,theta_45):
    mat_rot_45 = np.array([
        [np.cos(theta_45), -np.sin(theta_45)],
        [np.sin(theta_45), np.cos(theta_45)]
    ])
    Vect_tr= [Vect[0],Vect[1]-np.radians(45)]
    New_vect_tr = mat_rot_45.dot(Vect_tr)
    New_vect= New_vect_tr + [0,np.radians(45)]
    return New_vect
# %% codecell
Vectinit=[np.radians(0), np.radians(95)]
theta_s = []
theta_l = []
theta_45 = []
for t in range(tmps):
    theta_l.append(2*np.pi*t/5760)
    theta_s.append(Vect_anti_sun[t])
    theta_45.append(2*np.pi*t/600)

U1=[]
U2=[]
U3=[]
j = 0
Vect1deg=[[],[]]
Vect2deg=[[],[]]
for i in range(tmps):
    U1 = rot_45(Vectinit,theta_45[i])
    U2 = rot_l(U1,theta_l[i])
    U3 = U2
    U3[0] = U3[0] + theta_s[i]
    while U3[0] > np.pi:
        U3[0] = -np.pi + (U3[0] - np.pi)
    if U3[1] > np.pi/2:
        U3[1] = np.pi/2 - (U3[1] - np.pi/2)
    if U3[1] < -np.pi/2:
        U3[1] = -np.pi/2 - (U3[1] + np.pi/2)
    Vect2deg[0].append(U3[0])
    Vect2deg[1].append(U3[1])
    j = j + 1


# %% codecell
print(j)
# %% codecell

# %% codecell
print(max(Vect2deg[0]), min(Vect2deg[0]), max(Vect2deg[1]), min(Vect2deg[1]))
# %% codecell
nmb_pix = hp.nside2npix(1024)

Pixel1 = []
data1 = np.zeros((nmb_pix, 2))
data_line1 = [0]*nmb_pix
for k in range(nmb_pix):
    data_line1[k] = []


for i in range(tmps):
    pix = hp.ang2pix(1024, (Vect2deg[1][i] + np.pi/2), Vect2deg[0][i], nest=False, lonlat=False)
    data1[pix][1] = data1[pix][1] + 1
    data_line1[pix].append(tmps)
    Pixel1.append(pix)


# %% codecell
sky = fits.open("COM_CMB_IQU-nilc_1024_R2.02_full.fits")
sky2 = hp.read_map("COM_CMB_IQU-nilc_1024_R2.02_full.fits")
I = sky[1].data['I_STOKES']
U = sky[1].data['U_STOKES']
Q = sky[1].data['Q_STOKES']
# %% codecell
I_lu1 = np.zeros(nmb_pix)
U_lu = []
Q_lu = []
I_lu = np.zeros(nmb_pix)
j = 0

for i in range(nmb_pix):
    I_lu1[i] = data1[i][1]
    if data1[i][1]>1:
        I_lu[i] = I[i]
        j = j + 1


print(j)
# %% codecell
print(max(I_lu), min(I_lu))
# %% codecell
hp.mollview( I_lu, coord=[ "E"],  title="Histogram equalized Galactic", unit="mK", norm="hist",min=min(I_lu), max=max(I_lu))
#hp.graticule()
# %% codecell
In_out = I - I_lu
# %% codecell
print(max(In_out), min(In_out))
# %% codecell
hp.mollview( In_out, title="Histogram equalized Galactic", unit="mK", norm="hist",min=min(In_out), max=max(In_out))
hp.graticule()
# %% codecell
LMAX = 1024
cl = hp.anafast(In_out, lmax=LMAX)
ell = np.arange(len(cl))
# %% codecell
#Calculating the beam window function
ls = np.arange(1024)
beam_arcmin = 30.0

def B_l(beam_arcmin, ls):
    theta_fwhm = ((beam_arcmin/60.0)/180.0)*math.pi #angle in radians
    theta_s = theta_fwhm/(math.sqrt(8*math.log(2)))
    return np.exp(-2*(ls + 0.5)**2*(math.sin(theta_s/2.0))**2)

pl = hp.pixwin(1024)


#Deconvolve the beam and the pixel window function
dl = cl/(B_l(10.0, ell)**2*pl[0:1025]**2)




#Apply scaling factors for plotting (convention)
dl = (ell * (ell + 1) * dl/ (2*math.pi))



# %% codecell
fig = plt.figure(3)
ax = fig.add_subplot(111)

ax.scatter(ell, ell * (ell + 1) * dl/ (2*math.pi) ,
           s=4, c='red', lw=0,
           label='I_nicl')


ax.set_xscale('log')
#ax.set_yscale('log')


ax.set_xlabel('$\ell$')
ax.set_ylabel('$\ell(\ell+1)C_\ell/2\pi \,\,(\mu K^2)$ ')
ax.set_title('Angular Power Spectra')
ax.legend(loc='upper right')
ax.grid()

ax.set_xlim(2, 2000)
#ax.set_ylim(2, 1000000)

plt.show()
# %% codecell
from scipy.stats import norm
import matplotlib.mlab as mlab

# best fit of data
(mu, sigma) = norm.fit(In_out)

# the histogram of the data
n, bins, patches = plt.hist(In_out, 60, density=1, facecolor='r', alpha=0.25)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)

plt.xlabel('Temperature/K')
plt.ylabel('Frequency')
plt.title(r'Histogram of $12N_{side}^2$ pixels from the Planck SMICA map', y=1.08)
plt.xlim(-0.001, 0.001)

plt.legend()

plt.show()
# %% codecell
moy = np.mean(In_out)
sigma = np.std(In_out)
# %% codecell
print(moy,sigma)
# %% codecell
nmb_pix2 = hp.nside2npix(128)

Pixel2 = []
data2 = np.zeros((nmb_pix2, 2))
data_line2 = [0]*nmb_pix2
for k in range(nmb_pix2):
    data_line2[k] = []


for i in range(tmps):
    pix = hp.ang2pix(128, (Vect2deg[1][i] + np.pi/2), Vect2deg[0][i], nest=False, lonlat=False)
    data2[pix][1] = data2[pix][1] + 1
    data_line2[pix].append(tmps)
    Pixel2.append(pix)


# %% codecell
print(nmb_pix2, len(Pixel2), max(Pixel2),min(Pixel2))
# %% codecell
I_moins = hp.ud_grade(I, 128)

# %% codecell
I_lu2 = np.zeros(nmb_pix2)
j = 0

for i in range(nmb_pix2):
    if data2[i][1]>1:
        I_lu2[i] = I_moins[i]
        j = j + 1


print(j)
# %% codecell
hp.mollview( I_lu2, coord=[ "E"],  title="Histogram equalized Galactic", unit="mK", norm="hist",min=min(I_moins), max=max(I_moins))
hp.graticule()
# %% codecell
In_out2 = I_moins - I_lu2
hp.mollview( In_out2, title="Histogram equalized Galactic", unit="mK", norm="hist",min=min(In_out2), max=max(In_out2))
hp.graticule()
# %% codecell
LMAX2 = 128
cl2 = hp.anafast(In_out2, lmax=LMAX2)
ell2 = np.arange(len(cl2))
# %% codecell
#Calculating the beam window function
ls2 = np.arange(128)
beam_arcmin = 30.0

def B_l(beam_arcmin, ls):
    theta_fwhm = ((beam_arcmin/60.0)/180.0)*math.pi #angle in radians
    theta_s = theta_fwhm/(math.sqrt(8*math.log(2)))
    return np.exp(-2*(ls2 + 0.5)**2*(math.sin(theta_s/2.0))**2)

pl2 = hp.pixwin(128)


#Deconvolve the beam and the pixel window function
dl2 = cl2/(B_l(10.0, ell2)**2*pl2[0:128]**2)




#Apply scaling factors for plotting (convention)
dl2 = (ell2 * (ell2 + 1) * dl2/ (2*math.pi))


# %% codecell
fig = plt.figure(3)
ax = fig.add_subplot(111)

ax.scatter(ell2, ell2 * (ell2 + 1) * cl2/ (2*math.pi) / 1e-12,
           s=4, c='red', lw=0,
           label='I_nicl')


ax.set_xscale('log')
#ax.set_yscale('log')


ax.set_xlabel('$\ell$')
ax.set_ylabel('$\ell(\ell+1)C_\ell/2\pi \,\,(\mu K^2)$ ')
ax.set_title('Angular Power Spectra')
ax.legend(loc='upper right')
ax.grid()

ax.set_xlim(2, 500)
ax.set_ylim(-1e-16,1e-16)

plt.show()
# %% codecell
print(min(In_out2), max(In_out2))
# %% codecell
from scipy.stats import norm
import matplotlib.mlab as mlab

# best fit of data
(mu, sigma) = norm.fit(In_out2)

# the histogram of the data
#n, bins, patches = plt.hist(In_out2, 60, density=1, facecolor='r', alpha=0.25)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)

plt.xlabel('Temperature/K')
plt.ylabel('Frequency')
plt.title(r'Histogram of $12N_{side}^2$ pixels from the Planck SMICA map', y=1.08)
plt.xlim(-1e-16, 1e16)
plt.ylim(0, 1e-16)


plt.legend()

plt.show()
# %% codecell
