import numpy as np
import cv2
from math import cos, sin, exp

'''
Image (e.g.) logo is distorted in different ways, then overlayed on background
'''

## coefficients to regulate geometric and color distortions

# image is transformed as shear -> rotation -> shear
COEF_SHEAR = 0.5
COEF_ROT = 0.3

# hue is shifted, intensity histogram is pushed either to blacks or whites
COEF_HUE = 10.0
COEF_INTENSITY = 3.

# image size is SIZE_BASE +- scale distortions
SC_MAX = 1.0
SIZE_BASE = 50

# image is first downsized to SIZE_TEMP for speed of distortions
SIZE_TEMP = 100




def is_bgra(im):
  return len(im.shape) == 3 and im.shape[2] == 4

def is_bgr(im):
  return len(im.shape) == 3 and im.shape[2] == 3


def warp_patch (image0, M, K):
  assert is_bgra(image0), image0.shape
  assert image0.shape[0] == image0.shape[1]
  newshape = image0.shape[0] * K
  mid      = image0.shape[0] / 2
  offset_x = newshape / 2 - mid * (M[0,0] + M[0,1])
  offset_y = newshape / 2 - mid * (M[1,0] + M[1,1])
  #print M
  #print image0.shape
  #print offset_x, offset_y
  M = np.hstack([M, np.asarray([[offset_x],[offset_y]])])
  image = np.zeros((newshape,newshape,4), image0.dtype)
  cv2.warpAffine(image0, M, (newshape,newshape), image,
                 borderMode=cv2.BORDER_TRANSPARENT)
  assert image.shape[2] == 4
  return image


def geometric_distort (image0):

  assert image0.shape[0] == image0.shape[1], 'need a square on input'
  assert is_bgra(image0), image0.shape

  # warp
  shear1 = exp((np.random.rand()-0.5) * COEF_SHEAR)
  rot    = np.random.randn() * COEF_ROT
  shear2 = exp((np.random.rand()-0.5) * COEF_SHEAR)
  Shear1 = np.asarray([[shear1, 0], [0, 1.0/shear1]])
  Rot    = np.asarray([[cos(rot), sin(rot)], [-sin(rot), cos(rot)]])
  Shear2 = np.asarray([[shear2, 0], [0, 1.0/shear2]])    
  #print shear1, rot, shear2
  M = np.matmul(np.matmul(Shear2, Rot), Shear1)
  image = warp_patch (image0, M, 2)

  # crop to roi
  nnz = np.nonzero(image[:,:,3])
  # roi = [y1 x1 y2 x2)
  roi = (min(nnz[0].tolist()), min(nnz[1].tolist()),
         max(nnz[0].tolist()), max(nnz[1].tolist()))
  #print roi
  image = image[roi[0]:roi[2],roi[1]:roi[3],:]

  return image



def color_distort (image0):

  # save alpha channel separately
  assert is_bgra(image0), image0.shape
  b, g, r, a = cv2.split(image0)
  image0 = cv2.merge((b, g, r))

  hsv = cv2.cvtColor(image0, cv2.COLOR_BGR2HSV)

  def make_lut(a, n=256):
    '''create a lokup table that will saturate the image historgram,
       either into white (for a > 1), or to black (for 0 < a < 1) '''
    assert a > 0
    if a >= 1:
      lut = np.power(np.arange(n, dtype=float) / (n-1), a) * (n-1)
    else:
      lut = (n-1) - make_lut(1/a, n)[::-1]
    return lut.astype(np.uint8)

  # change hue pixelwise
  dhue = (np.random.rand() - 0.5) * COEF_HUE
  hsv[:,:,0] = np.mod(hsv[:,:,0].astype(int) + dhue, 255).astype(np.uint8)
  # change histogram of values
  dval = np.exp((np.random.rand() - 0.5) * COEF_INTENSITY)
  lut = make_lut(dval)
  hsv[:,:,2] = cv2.LUT(hsv[:,:,2], lut)

  image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

  # restore saved alpha channel
  b, g, r = cv2.split(image)
  image = cv2.merge((b, g, r, a))

  return image



def distort_logo (logo0):
  ''' apply all kind of distortions to logo0 '''

  # first downscale for faster computations (it's better to have a square)
  logo = cv2.resize(logo0, (SIZE_TEMP,SIZE_TEMP))
  logo = geometric_distort (logo)
  logo = color_distort (logo)
  # random scaling
  scale = exp((np.random.rand()-0.5) * SC_MAX)
  logo = cv2.resize(logo, None, None, scale, scale)
  return logo


def overlay_logo (background, logo):
  ''' Overlay semi-transparent logo onto image center '''

  # crop background center, same size as logo
  if logo.shape[0] > background.shape[0] or logo.shape[1] > background.shape[1]:
    raise Exception('logo too large')
  y1 = (background.shape[0] - logo.shape[0]) / 2
  x1 = (background.shape[1] - logo.shape[1]) / 2
  h,w = logo.shape[:2]
  crop = background[y1:y1+h, x1:x1+w, :]

  def blend (background, logo):
    ''' opencv lacks functionality to put blend images with alpha-channel
        In this case only logo has alpha-channel '''
    assert is_bgra(logo), logo.shape
    assert is_bgr(background), background.shape
    assert logo.shape[0] == background.shape[0]
    assert logo.shape[1] == background.shape[1]

    src_B, src_G, src_R, src_A = cv2.split(logo.astype(float))
    dst_B, dst_G, dst_R        = cv2.split(background.astype(float))

    dst_R = (src_R*src_A + dst_R*(255 - src_A)) / 255;
    dst_G = (src_G*src_A + dst_G*(255 - src_A)) / 255;
    dst_B = (src_B*src_A + dst_B*(255 - src_A)) / 255;

    return cv2.merge((dst_B, dst_G, dst_R)).astype(np.uint8)

  crop = blend(crop, logo)
  blended = background.copy()
  blended[y1:y1+h, x1:x1+w, :] = crop

  return blended, [x1, y1, x1+w, y1+h]



if __name__ == "__main__":
  ''' test on one logo and background '''

  background = cv2.imread('background.jpg')
  assert background is not None
  assert is_bgr(background), \
    'has to have THREE channels. Now shape is %s' % str(background.shape)

  logo = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)
  assert logo is not None
  assert is_bgra(logo), \
    'has to have FOUR channels (with alpha). Now shape is %s' % str(logo.shape)

  for i in range(20):
    blended, _ = overlay_logo(background, distort_logo(logo))
    cv2.imshow('test', blended)
    cv2.waitKey(-1)
