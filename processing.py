import load_j as lj
import numpy as np
import scipy.misc as sm
import scipy.ndimage as nd

fname = '/home/billy/Documents/Research/data/ccore/train.json/data/processed/train.json'

labs, hh, hv = lj.load_j(fname)
#to save normalized images
hh_norm = []
hv_norm = []
hh_smoo = []
hv_smoo = []
hh_maxes = []
hv_maxes = []

for i in range(0, (len(hh) ) ):
    #hh
    temp = (hh[i] - np.mean(hh[i]) )/(np.sqrt(np.var(hh[i])))
    hh_norm.append(temp+ np.abs(np.min(temp)) )
    hh_smoo.append( nd.gaussian_filter(hh_norm[i], sigma=1.5) )
    #hh_maxes.append(np.max(hh_norm[i]))
    #hv
    temp = (hv[i] - np.mean(hv[i]) )/(np.sqrt(np.var(hv[i])))
    hv_norm.append(temp+ np.abs(np.min(temp)) )
    hv_smoo.append( nd.gaussian_filter(hv_norm[i], sigma=1.5) )



sm.imsave('hh_norm_1.png', hh_norm[0])
sm.imsave('hh_smoo_1.png', hh_smoo[0])

#have similar values for the max of each image (used mean, max, and sd)
#taking the average of the two images should be a reasonable decision

h_ave = np.divide(np.add(hh_norm,0.0000001),np.add(hv_norm,0.0000001) )
mask = (h_ave[0]<3)+0.000
mask2 = (h_ave[0]>0.9)+0.000
rslt = mask*h_ave[0]
sm.imsave("temp_rat.png", rslt)

sm.imsave('temp_ave.png',h_ave[0])

thresh = (h_ave[0]>7)+0.0

sm.imsave('temp_thresh.png', thresh)

#
