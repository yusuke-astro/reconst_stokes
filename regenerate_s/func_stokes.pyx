import numpy as np
import cython
cimport numpy as cnp
from joblib import Parallel, delayed

def stokes(int i, cnp.ndarray pix, cnp.ndarray I_obs):#iには0からNPIXまでのintをいれる
    cdef:
        int times
        cnp.ndarray  D_mat, sum_D_pinv, D_E, wp, sum_wp, s_tilde, t
        double w

    t = np.where(pix == i)[0]#i番目のPixcelを踏んだときの時間がtに配列ではいる．

    times = t.size
    #print(times)
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

def stokes_loop(cnp.ndarray pix, cnp.ndarray I_obs, int N):#iには0からNPIXまでのintをいれる
    cdef:
      int times, i
      cnp.ndarray  D_mat, sum_D_pinv, D_E, wp, sum_wp, t
      list s_tilde
      double w
    w = 2*np.pi*(0.3/60)
    s_tilde = []
    for i in range(N):
      t = np.where(pix == i)[0]#i番目のPixcelを踏んだときの時間がtに配列ではいる．
      times = t.size

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
      s_tilde.append(np.dot(sum_D_pinv, sum_wp))

    return np.array(s_tilde).T#0~NPIX番目までのピクセルに対するストークスパラメータが入る．
