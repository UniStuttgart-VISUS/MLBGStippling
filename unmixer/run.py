# pip install numpy scipy Cython Pillow
# pip install cvxopt ... is tricky.
import numpy as np
import scipy
import Additive_mixing_layers_extraction
from scipy.spatial import ConvexHull, Delaunay
import scipy.sparse
import scipy
import PIL.Image as Image
from PIL import ImageFilter
from numpy import *
import os
import shutil
import time

# Modify image settings here.
image = {"path": "../examples/fire/source.png", "palette": ["#000000", "#404040", "#400000", "#800000", "#ff8000", "#ffff40"], "filter": True}
#image = {"path": "../examples/toucan/source.jpg", "palette": ["#ffffff", "#000000", "#fff786", "#8bcaab", "#2b88a9", "#963044", "#74ae5a", "#e18401"], "filter": False}

print("Loading", image["path"])
im1 = Image.open(image["path"])
img=np.asfarray(im1.convert('RGB'))/255.0
palette_rgb = []
for color in image["palette"]:
    if isinstance(color, str):
        color = color.lstrip('#')
        lv = len(color)
        palette_rgb.append([int(color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)])
    else:
        palette_rgb.append(color)
palette_rgb = np.array(palette_rgb) / 255.0

if image["filter"]:
    print("Filtering...")
    im1 = im1.filter(ImageFilter.MedianFilter(5))

print("Decomposing...")
arr=img.copy()
X,Y=np.mgrid[0:img.shape[0], 0:img.shape[1]]
XY=np.dstack((X*1.0/img.shape[0],Y*1.0/img.shape[1]))
data=np.dstack((img, XY))
print("...Convex Hull", data.shape)
data_hull=ConvexHull(data.reshape((-1,5))) #, qhull_options = 'Qx C-0.0001 C0.001') # Stay below 1.0 / 255.0 ~= 0.0039
print("...Mixing Layers")
Additive_mixing_layers_extraction.DEMO=True
mixing_weights_1 = Additive_mixing_layers_extraction.Get_ASAP_weights_using_Tan_2016_triangulation_and_then_barycentric_coordinates(img.reshape((-1,3))[data_hull.vertices].reshape((-1,1,3)), palette_rgb, None, order=0)
mixing_weights_2 = Additive_mixing_layers_extraction.recover_ASAP_weights_using_scipy_delaunay(data_hull.points[data_hull.vertices], data_hull.points, option=3)
mixing_weights = mixing_weights_2.dot(mixing_weights_1.reshape((-1,len(palette_rgb))))
mixing_weights = mixing_weights.reshape((img.shape[0],img.shape[1],-1))

def drop_lower_bound(weights, min_T):
    return weights - weights.clip(0.0, min_T)

# Clip small values and renormalize
min_weight = 8.0 / 255.0
mixing_weights = (mixing_weights.clip(min_weight, 1.0) - min_weight) / (1.0 - min_weight)

# Invert
mixing_weights = 1.0 - mixing_weights

print("Cleaning...")
try:
    shutil.rmtree('out', ignore_errors = True)
    time.sleep(1)
    os.mkdir('out')
except FileExistsError:
    None

print("Saving palette...")
c=50
palette2=np.ones((1*c, len(palette_rgb)*c, 3))
for i in range(len(palette_rgb)):
    palette2[:,i*c:i*c+c,:]=palette_rgb[i,:].reshape((1,1,-1))
Image.fromarray((palette2*255).round().astype(np.uint8)).save("out/color.png")

print("Saving layers...")
for i in range(0, mixing_weights.shape[-1]):
    final = (mixing_weights[:,:,i]*255).round().clip(0, 255).astype(np.uint8)
    im_filename = "out/layer-{}_{:02x}{:02x}{:02x}.png" .format(i, int(palette_rgb[i][0] * 255.0), int(palette_rgb[i][1] * 255.0), int(palette_rgb[i][2] * 255.0))
    im = Image.fromarray(final)
    im.save(im_filename)

print("Done")
