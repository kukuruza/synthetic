import os, os.path as op
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

''' This file is for better placement of the distorted logo on background. 
    The logo is put on a uniform area. A big uniform area is picked as
    as a big uniform superpixel.
'''


def select_segments(image, segments):

  for i in range(segments.max()+1):
    segment = np.nonzero(segments == i)
    
    r_segment = image[:,:,0][segment]
    g_segment = image[:,:,1][segment]
    b_segment = image[:,:,2][segment]
    
    center = (segment[0].mean(), segment[1].mean())
    avgsize = sqrt(len(binary_segment[0]))
    halfsize = avgsize / 2
    if center[0] - halfsize < 0 or center[0] + halfsize >= image.shape[0] or \
       center[1] - halfsize < 0 or center[1] + halfsize >= image.shape[1]:
       print 'logo wont fit into the segment %d with the standard scale'
       continue
    scaled_logo = np.zeros(shape=segment, dtype=bool)
    scaled_logo[center[0] - halfsize : center[0] + halfsize,
                center[1] - halfsize : center[1] + halfsize] = True
    (logo_shape[0] * )

    print np.std(r_segment), np.std(g_segment), np.std(b_segment)

  return segments



if __name__ == "__main__":

  ''' test with an example image '''

  background = cv2.imread('background.jpg').astype(np.float32) / 255
 
  segments = slic(background, n_segments = 100, sigma = 5)
  segments = select_segments(background, segments)
 
  #fig = plt.figure("Superpixels")
  #ax = fig.add_subplot(1, 1, 1)
  #ax.imshow(segments == 0) #mark_boundaries(background, segments))
  #plt.axis("off")
  #plt.show()