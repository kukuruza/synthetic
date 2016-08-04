import sys
import os, os.path as op
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float, img_as_ubyte
from skimage import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import sqrt

''' This file is for better placement of the distorted logo on background. 
    The logo is put on a uniform area. A big uniform area is picked as
    as a big uniform superpixel.
'''


N_SEGMENTS = 40  # the higher this number, the less are segments sizes
THRESHOLD_SEGMENT_IS_SQUARISH = 0.65
THRESHOLD_SEGMENT_IS_UNIFORM = 0.2
THRESHOLD_SEGMENT_IS_SIMILAR_COLOR = 0.3
THRESHOLD_SPARSITY = 2.


def _is_bgra(im):
  return len(im.shape) == 3 and im.shape[2] == 4

def _is_bgr(im):
  return len(im.shape) == 3 and im.shape[2] == 3


def is_segment_squarish(image, segment):
  ''' returns roi in the format [y1,x1,y2,x2] '''

  nnz = np.nonzero(segment)
  center = (nnz[0].mean(), nnz[1].mean())
  avgsize = sqrt(len(nnz[0]))
  halfsize = avgsize / 2
  if center[0] - halfsize < 0 or center[0] + halfsize >= image.shape[0] or \
     center[1] - halfsize < 0 or center[1] + halfsize >= image.shape[1]:
     # logo wont fit into the segment with the standard scale
     return False, None
  scaled_logo = np.zeros(shape=image.shape[:2], dtype=bool)
  roi = [center[0] - halfsize, center[1] - halfsize,
         center[0] + halfsize, center[1] + halfsize]
  roi = [int(x) for x in roi]
  scaled_logo[roi[0]:roi[2], roi[1]:roi[3]] = True
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

  return IoU > THRESHOLD_SEGMENT_IS_SQUARISH, roi



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



def is_segment_similar_color(image, segment, logo):

  # get the color distribution of the logo (yes, again every time for now)
  alpha  = logo[:,:,3]
  r_logo = img_as_float(logo[:,:,2])[alpha]
  g_logo = img_as_float(logo[:,:,1])[alpha]
  b_logo = img_as_float(logo[:,:,0])[alpha]
  r_mean = np.mean(r_logo)
  g_mean = np.mean(g_logo)
  b_mean = np.mean(b_logo)
  logo_mean = (r_mean, g_mean, b_mean)
  print 'logo_mean: ', logo_mean

  r_segment = image[:,:,2][segment]
  g_segment = image[:,:,1][segment]
  b_segment = image[:,:,0][segment]
  print (np.mean(r_segment), np.mean(g_segment), np.mean(b_segment))

  difference = abs(np.mean(r_segment) - mean_logo[2]) \
             + abs(np.mean(g_segment) - mean_logo[1]) \
             + abs(np.mean(b_segment) - mean_logo[0])
  
  # test = image.copy()
  # test[np.invert(segment)] = 0
  # print 'difference: %f' % difference
  # cv2.imshow('is_segment_similar_color', img_as_ubyte(test))
  # key = cv2.waitKey(-1)
  # if key == 27: sys.exit()

  return difference > THRESHOLD_SEGMENT_IS_SIMILAR_COLOR


def keep_sparse_rois(rois):
  ''' Keep only rois which are not close to each other '''

  def is_far_from_others (roi, old_rois):
    new_center = ((roi[0]+roi[2]) / 2, (roi[1]+roi[3]) / 2)
    new_size = (roi[2]-roi[0] + roi[3]-roi[1]) / 2
    for old_roi in old_rois:
      old_center = ((old_roi[0]+old_roi[2]) / 2, (old_roi[1]+old_roi[3]) / 2)
      dist = np.linalg.norm (np.asarray(old_center)-np.asarray(new_center))
      if dist < new_size * THRESHOLD_SPARSITY:
        return False
    return True

  good_rois = []
  for roi in rois:
    if is_far_from_others (roi, good_rois):
      good_rois.append (roi)
  return good_rois


def find_good_rois (background, logo, display=False):
  ''' returns rois in the format [y1,x1,y2,x2] '''

  assert _is_bgr (background)
  assert _is_bgra (logo)

  background = img_as_float(background)

  segments = slic(background, N_SEGMENTS, sigma = 5)

  good_segment_ids = []
  good_rois = []

  for i in range(segments.max()+1):
    segment = (segments == i)

    #good = is_segment_similar_color(background, segment, logo)
    #if not good: continue
    good, roi = is_segment_squarish(background, segment)
    if not good: continue
    good = is_segment_uniform(background, segment)
    if not good: continue
    good_segment_ids.append(i)
    good_rois.append(roi)

  good_rois = keep_sparse_rois (good_rois)

  if display:
    for i in range(segments.max()+1):
      if i not in good_segment_ids: 
        segments[segments == i] = 0
    fig = plt.figure("Superpixels")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(background[:,:,::-1], segments))
    for roi in good_rois:
      x1, y1, width, height = roi[1], roi[0], roi[3]-roi[1], roi[2]-roi[0]
      print x1, y1, width, height
      ax.add_patch( patches.Rectangle((x1, y1), width, height, fill=False) )
    plt.axis("off")
    plt.show()

  return good_rois



if __name__ == "__main__":

  ''' test with an example image '''

  background = cv2.imread('background.jpg')
  logo = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)

  find_good_rois (background, logo, display=True)
