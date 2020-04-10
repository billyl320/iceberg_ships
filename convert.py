import scipy.misc as sm
import mgcreate as mgc
import scipy.ndimage as nd
import scipy.signal as ss
import numpy as np
import skimage as sk
from sklearn import cluster
from skimage import filters
from skimage import feature
from skimage.color import label2rgb
from skimage.feature import corner_harris, corner_peaks
from skimage.measure import regionprops as rp
import os
import math
import colorsys
import load_j as lj

#convert image to square centered at center of image
#assuming input has the shape as white
def convert(adata):
    #plopping into square centered at center of image with max of original image
    d = max(adata.shape)*2
    pic = mgc.Plop(adata, (d, d), 0)
    #swapping black and white (if needed [shape needs to be white])
    new = pic
    #new = 1-new
    # find the center of mass
    v = nd.center_of_mass(new)
    v = (int(v[0]), int(v[1]))
    #finding cetner of new image
    n = (d/2, d/2)
    #shift in horizontal
    hori = v[1]-n[1]
    #shift in vertical
    vert = v[0]-n[0]
    #shift the image
    ultima = nd.shift(new, (-vert, -hori), cval=0)
    return ultima

#finding minimum encompassing circle - needs to be binary (1 and 0)
def enc_circ(pic):
    ultima = pic + 0
    v,h = ultima.shape
    z = np.ones((v,h))+0
    z[v//2,h//2] = 0
    dist = nd.distance_transform_edt(z)
    vals = dist*ultima
    r = vals.max()
    r = math.ceil(r)
    ultima = ultima[(v//2 - r) : r + v//2, h//2 -r : r + h//2]
    return ultima

#from page 266 of Kinser (2018)
#gives some shape metrics
#this version only provides eccentricity
def Metrics(orig):
    v, h = orig.nonzero()
    mat = np.zeros((2, len(v)))
    mat[0] = v
    mat[1] = h
    evls, evcs = np.linalg.eig(np.cov(mat))
    eccen = evls[0]/evls[1]
    if eccen < 1: eccen = 1/eccen
    return eccen, evls[0], evls[1]

def Shapes(pic):
    #clean image (if needed)
    #pic = nd.gaussian_filter(pic, sigma=1.5)
    #pic = (pic > 0.99) +0.0
    #pic = (nd.binary_erosion(pic , iterations=20))+0.0
    #pic = (nd.binary_dilation(pic , iterations=20))+0.0
    #setup
    shapes = np.zeros((1,14))
    #obtain circularity
    circ = ( sum(sum((nd.binary_dilation(pic , iterations=1) - pic ))) **2) /(4*np.pi*sum(sum(pic)))
    shapes[0][0] = circ
    #provides eccentricity, eigen1 and eigen2
    eccen, e1, e2 = Metrics(pic)
    shapes[0][1] = eccen
    shapes[0][2] = e1
    shapes[0][3] = e2
    #number of corners
    corners = corner_harris(pic, k=0.1, sigma=0.01)
    shapes[0][4] = corner_peaks(corners, min_distance=1).shape[0]
    #white and black pixel counts for min bounding box
    theta = rp( (pic>0.5) +0)[0]['orientation']
    rot_pic = nd.rotate(pic, angle=theta*180/np.pi)
    rot_pic = (rot_pic>0.5) +0.00
    slice_x, slice_y = nd.find_objects(rot_pic==1)[0]
    roi = rot_pic[slice_x, slice_y]
    shapes[0][5] = np.unique(roi, return_counts=True)[1][1]
    shapes[0][6] = np.unique(roi, return_counts=True)[1][0]
    #calculting moments from data
    m = sk.measure.moments(pic)
    #centroid
    centroid = (m[0, 1] / m[0, 0], m[1, 0] / m[0, 0])
    #central moments
    mu = sk.measure.moments_central(pic, centroid)
    #normalizing moments
    nu = sk.measure.moments_normalized(mu)
    #calculting hu moments
    shapes[0][7:14] = sk.measure.moments_hu(nu)
    return shapes


#performs needed setup for other functions
#also provides binary edges info
#must use first
def bin_edges (fname, bname):#, thresh=2 ):
    #read in image as color
    adata = sm.imread(fname, flatten=True)
    #find edges
    gdata= np.abs(np.gradient( adata ))
    val = 0.0
    ddata = ((gdata[0]>val) + (gdata[1]>val) +0.0)+0.0
    ############
    #fill holes
    ############
    shape = nd.binary_fill_holes(ddata+0.0) + 0.0
    ##################
    #get largest shape
    ###################
    b, n = nd.label(shape+0.0)
    #finding biggest shapes (except 0s)
    simp = np.hstack(b)
    locs= np.nonzero( simp )
    counts=np.bincount(simp [locs] )
    vals=counts.argsort()[-3:][::-1]
    clist = list(map(lambda x: b==x, (0,vals[0]) ))
    shape = (clist[1]+0)
    ############
    #fill holes
    ############
    #shape = nd.binary_fill_holes(shape+0.0) + 0.0
    #shape = (shape>0) + 0.0
    sm.imsave(bname+fname[4:-3]+'png', shape+0.0)
    return shape

#gets binary image histogram
def BinaryHist(fname):
    labs, imgs_hh, imgs_hv, imgs_ave = lj.norm_j(fname)
    hist_hh = np.zeros( (len(imgs_hh),2) )
    hist_hv = np.zeros( (len(imgs_hv),2) )
    hist_ave = np.zeros( (len(imgs_hv),2) )
    #appling SPEI
    for i in range(0, len(imgs_hh)):
        print(i)
        b = convert(imgs_hh[i])
        c = enc_circ(b)
        hist_hh[i][0] = imgs_hh[i].sum()
        hist_hh[i][1] = (imgs_hh[i].shape[0]*imgs_hh[i].shape[1]) - hist_hh[i][0]
    #appling SPEI
    for i in range(0, len(imgs_hv)):
        print(i)
        b = convert(imgs_hv[i])
        c = enc_circ(b)
        hist_hv[i][0] = imgs_hv[i].sum()
        hist_hv[i][1] = (imgs_hv[i].shape[0]*imgs_hv[i].shape[1]) - hist_hv[i][0]
    #appling SPEI
    for i in range(0, len(imgs_ave)):
        print(i)
        b = convert(imgs_ave[i])
        c = enc_circ(b)
        hist_ave[i][0] = imgs_ave[i].sum()
        hist_ave[i][1] = (imgs_ave[i].shape[0]*imgs_ave[i].shape[1]) - hist_ave[i][0]
    #return vals
    return labs, hist_hh, hist_hv, hist_ave


#obtaining images for shape metrics except EIs
def GetAllImages_Shapes(fname):
    labs, imgs_hh, imgs_hv, imgs_ave = lj.norm_j(fname)
    vals_hh = []
    vals_hv = []
    vals_ave = []
    for i in range(0, len(imgs_hh)):
        c = Shapes(imgs_hh[i])
        vals_hh.append( c )
        c = Shapes(imgs_hv[i])
        vals_hv.append(c)
        c = Shapes(imgs_ave[i])
        vals_ave.append(c)
    return vals_hh, vals_hv, vals_ave

#save histogram as .txt where first column is white counts
#and second column is black counts
def BinaryHistTXT(tname, fname):
    #obtain histogram and labels
    labs, hist_hh, hist_hv, hist_ave = BinaryHist(fname)
    #save as txt
    np.savetxt("HH_"+tname, hist_hh, delimiter=',', header="white,black", comments='')
    np.savetxt("HV_"+tname, hist_hv, delimiter=',', header="white,black", comments='')
    np.savetxt("AVE_"+tname, hist_ave, delimiter=',', header="white,black", comments='')
    np.savetxt("LABS_"+tname, np.asarray(labs), delimiter=',', header="labs", comments='', fmt="%s")

#save histogram as .txt where first column is white counts
#and second column is black counts
def BinaryShapesTXT(tname, fname):
    #obtain shape metrics
    shapes_hh, shapes_hv, shapes_ave = GetAllImages_Shapes(fname)
    #get image names
    name6 = tname + "_HH_SHAPES.txt"
    name7 = tname + "_HV_SHAPES.txt"
    name8 = tname + "_AVE_SHAPES.txt"
    #save as txt
    np.savetxt(name6, np.vstack(shapes_hh), delimiter=',', header="Shape_circ, Shape_eccent, Shape_e1, Shape_e2, Shape_corn, White_box, Black_box, Hu1, Hu2, Hu3, Hu4, Hu5, Hu6, Hu7", comments='')
    np.savetxt(name7, np.vstack(shapes_hv), delimiter=',', header="Shape_circ, Shape_eccent, Shape_e1, Shape_e2, Shape_corn, White_box, Black_box, Hu1, Hu2, Hu3, Hu4, Hu5, Hu6, Hu7", comments='')
    np.savetxt(name8, np.vstack(shapes_ave), delimiter=',', header="Shape_circ, Shape_eccent, Shape_e1, Shape_e2, Shape_corn, White_box, Black_box, Hu1, Hu2, Hu3, Hu4, Hu5, Hu6, Hu7", comments='')

#
