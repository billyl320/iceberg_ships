#python functions to load and convert all of the images
import json
from pprint import pprint
import numpy as np
import scipy.misc as sm
import scipy.ndimage as nd

def load_j(fname):
    #load json file
    files = json.load(open(fname))
    #value of bands
    dim = len(files[0]["band_1"])
    #create list to hold images
    imgs_hh = []
    imgs_hv = []
    id = []
    y = []
    #for loop to extract all the images
    for i in range(0, (len(files) ) ):
        id.append(files[i]["id"])
        y.append(files[i]["is_iceberg"])
        imgs_hh.append(np.array(files[i]["band_1"]).reshape(75, 75))
        imgs_hv.append(np.array(files[i]["band_2"]).reshape(75, 75))
    #clearing cache
    files = None
    #return imgs
    return id, y, imgs_hh, imgs_hv

def norm_j(fname):
    files = load_j(fname)
    id, labs, hh, hv = load_j(fname)
    #to save normalized images
    hh_norm = []
    hv_norm = []
    h_ave = []
    hh_smoo = []
    hv_smoo = []
    ave_smoo = []
    thresh_hh = []
    thresh_hv = []
    thresh_ave = []
    for i in range(0, (len(hh) ) ):
        #hh
        print(id[i])
        temp = (hh[i] - np.mean(hh[i]) )/(np.sqrt(np.var(hh[i])))
        hh_norm.append(temp+ np.abs(np.min(temp)) )
        hh_smoo.append( nd.gaussian_filter(hh_norm[i], sigma=1) )
        #hh_maxes.append(np.max(hh_norm[i]))
        #hv
        temp = (hv[i] - np.mean(hv[i]) )/(np.sqrt(np.var(hv[i])))
        hv_norm.append(temp+ np.abs(np.min(temp)) )
        hv_smoo.append( nd.gaussian_filter(hv_norm[i], sigma=1) )
        thresh_hh.append((hh_smoo[i]>5.25)+0.0)
        thresh_hv.append((hv_smoo[i]>5.25)+0.0)
        h_ave.append(hh_norm[i]+hv_norm[i])
        h_ave[i] = h_ave[i]/2.0
        ave_smoo.append( nd.filters.convolve(input=h_ave[i], weights=np.full((3, 3), 1.0/9.0), mode='reflect' ) )
        thresh_ave.append( (ave_smoo[i]>5.50)+0.0)
        #getting largest object
        #inspired by https://stackoverflow.com/questions/15283849/isolate-greatest-smallest-labeled-patches-from-numpy-array
        labeled_array, numpatches = nd.label(thresh_ave[i]) # labeling
        sizes = nd.sum(thresh_ave[i],labeled_array,range(1,numpatches+1))
        # To get the indices of all the min/max patches.
        map = np.where(sizes==sizes.max())[0] + 1
        # inside the largest, respecitively the smallest labeled patches with values
        max_index = np.zeros(numpatches + 1, np.uint8)
        max_index[map] = 1
        thresh_ave[i] = (max_index[labeled_array]+0.0)
        sm.imsave('images/hh/hh_'+str(id[i])+'.png', thresh_hh[i])
        sm.imsave('images/hv/hv_'+str(id[i])+'.png', thresh_hv[i])
        sm.imsave('images/ave/ave_'+str(id[i])+'.png', thresh_ave[i])
        #hv_maxes.append(np.max(hv_norm[i]))
    #have similar values for the max of each image (used mean, max, and sd)
    #taking the average of the two images should be a reasonable decision
    #originally 6 = 0.68125; 5 is less
    return labs, thresh_hh, thresh_hv, thresh_ave

#
