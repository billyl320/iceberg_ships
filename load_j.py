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

#loads images, processes images, and performs data augementation
def norm_j(fname, deg=90.0):
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
    labs2 = []
    #rotations for rotated images
    turns = (360.0)/(deg+0.0)
    for i in range(0, (len(hh) ) ):
        #hh
        print(id[i])
        print(i)
        #new labs
        labs2.append(labs[i])
        # original image
        temp = (hh[i] - np.mean(hh[i]) )/(np.sqrt(np.var(hh[i])))
        temp_norm_hh = temp+ np.abs(np.min(temp))
        hh_norm.append( temp_norm_hh )
        temp_smoo_hh = nd.gaussian_filter(temp_norm_hh, sigma=1)
        hh_smoo.append( temp_smoo_hh )
        #hh_maxes.append(np.max(hh_norm[len(hh_norm)]))
        #hv
        temp = (hv[i] - np.mean(hv[i]) )/(np.sqrt(np.var(hv[i])))
        temp_norm_hv = temp+ np.abs(np.min(temp))
        hv_norm.append( temp_norm_hv )
        temp_smoo_hv = nd.gaussian_filter(temp_norm_hv, sigma=1)
        hv_smoo.append( temp_smoo_hv )
        thresh_hh.append((temp_smoo_hh>5.25)+0.0)
        thresh_hv.append((temp_smoo_hv>5.25)+0.0)
        temp_ave = (temp_norm_hh+temp_norm_hv)/2.0
        h_ave.append(temp_ave)
        temp_ave_smo = nd.filters.convolve(input=temp_ave, weights=np.full((3, 3), 1.0/9.0) )
        ave_smoo.append( temp_ave_smo ) #, mode='constant', cval=0.0 ) )
        temp_ave_thresh = (temp_ave_smo>5.50)+0.0
        #getting largest object
        #inspired by https://stackoverflow.com/questions/15283849/isolate-greatest-smallest-labeled-patches-from-numpy-array
        labeled_array, numpatches = nd.label(temp_ave_thresh) # labeling
        sizes = nd.sum(temp_ave_thresh,labeled_array,range(1,numpatches+1))
        # To get the indices of all the min/max patches.
        map = np.where(sizes==sizes.max())[0] + 1
        # inside the largest, respecitively the smallest labeled patches with values
        max_index = np.zeros(numpatches + 1, np.uint8)
        max_index[map] = 1
        temp_ave_thresh = (max_index[labeled_array]+0.0)
        thresh_ave.append(temp_ave_thresh)
        #sm.imsave('images/hh/hh_'+str(id[i])+'_turn_'+str(j)+'_.png', thresh_hh[len(thresh_hh)-1])
        #sm.imsave('images/hv/hv_'+str(id[i])+'_turn_'+str(j)+'_.png', thresh_hv[len(thresh_hv)-1])
        sm.imsave('images/ave/ave_'+str(id[i])+'.png', temp_ave_thresh)
        #data augementation (rotations)
        for j in range(1, int(turns)):
            print(j)
            #new labs
            labs2.append(labs[i])
            temp = nd.rotate(temp_ave_thresh, deg*j, reshape=True, cval=0)
            temp_ave_thresh2 = (temp>0.5)+0.0
            thresh_ave.append( temp_ave_thresh2 )
            #sm.imsave('images/hh/hh_'+str(id[i])+'_turn_'+str(j)+'_.png', thresh_hh[len(thresh_hh)-1])
            #sm.imsave('images/hv/hv_'+str(id[i])+'_turn_'+str(j)+'_.png', thresh_hv[len(thresh_hv)-1])
            sm.imsave('images/ave/ave_'+str(id[i])+'_turn_'+str(j)+'_.png', temp_ave_thresh2)
            #hv_maxes.append(np.max(hv_norm[len(hv_norm)]))
    #have similar values for the max of each image (used mean, max, and sd)
    #taking the average of the two images should be a reasonable decision
    #originally 6 = 0.68125; 5 is less
    return labs2, thresh_hh, thresh_hv, thresh_ave

#
