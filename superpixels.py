import sys
import os, os.path as op
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float, img_as_ubyte
from skimage import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import logging

''' This file is for better placement of the distorted logo on background. 
    The logo is put on a uniform area. A big uniform area is picked as
    as a big uniform superpixel.
'''


THRESHOLD_SEGMENT_IS_SQUARISH = 0.65
THRESHOLD_SEGMENT_IS_UNIFORM = 0.15


def is_segment_squarish(image, segment):

  nnz = np.nonzero(segment)
  center = (nnz[0].mean(), nnz[1].mean())
  avgsize = sqrt(len(nnz[0]))
  halfsize = avgsize / 2
  if center[0] - halfsize < 0 or center[0] + halfsize >= image.shape[0] or \
     center[1] - halfsize < 0 or center[1] + halfsize >= image.shape[1]:
     logging.debug('logo wont fit into the segment with the standard scale')
     return False
  scaled_logo = np.zeros(shape=image.shape[:2], dtype=bool)
  scaled_logo[center[0] - halfsize : center[0] + halfsize,
              center[1] - halfsize : center[1] + halfsize] = True
  I = np.bitwise_and (segment, scaled_logo)
  U = np.bitwise_or  (segment, scaled_logo)
  IoU = float(np.count_nonzero(I)) / float(np.count_nonzero(U))

  # print 'IoU: %f' % IoU
  # test = np.zeros(shape=image.shape, dtype=np.uint8)
  # test[:,:,0] = img_as_ubyte(segment)
  # test[:,:,1] = img_as_ubyte(scaled_logo)
  # cv2.imshow('is_segment_squarish', test)
  # key = cv2.waitKey(-1)
  # if key == 27: sys.exit()

  return IoU > THRESHOLD_SEGMENT_IS_SQUARISH


def is_segment_uniform(image, segment):

  r_segment = image[:,:,2][segment]
  g_segment = image[:,:,1][segment]
  b_segment = image[:,:,0][segment]
  uniformity = np.std(r_segment) + np.std(g_segment) + np.std(b_segment)
  
  # test = image.copy()
  # test[np.invert(segment)] = 0
  # print 'uniformity: %f' % uniformity
  # cv2.imshow('is_segment_uniform', img_as_ubyte(test))
  # key = cv2.waitKey(-1)
  # if key == 27: sys.exit()

  return uniformity < THRESHOLD_SEGMENT_IS_UNIFORM




def select_segments(image, segments):

  good_segments = []

  for i in range(segments.max()+1):
    segment = (segments == i)

    if is_segment_squarish(image, segment) and is_segment_uniform(image, segment): 
      good_segments.append(i)
    

  return segments



if __name__ == "__main__":

  ''' test with an example image '''

  logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

  background = img_as_float(cv2.imread('background.jpg'))
 
  segments = slic(background, n_segments = 100, sigma = 5)
  segments = select_segments(background, segments)
 
  #fig = plt.figure("Superpixels")
  #ax = fig.add_subplot(1, 1, 1)
  #ax.imshow(segments == 0) #mark_boundaries(background, segments))
  #plt.axis("off")
  #plt.show()