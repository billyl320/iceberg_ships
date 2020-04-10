#importing directories
import convert as cvt

fname = '/home/billy/Documents/Research/data/ccore/train.json/data/processed/train.json'

#name of .txt file
name = 'ice_boat.txt'

#converting images
cvt.BinaryHistTXT(name, fname)
cvt.BinaryShapesTXT(name[-0:-4], fname)



#
