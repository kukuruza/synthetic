import numpy as np
import cv2
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation

''' Remove background from images in case it's uniform '''


# background color must be in a large portion of the image
DOMINANT_TO_WHOLE = 0.  # doesn't really work
# background color must be much more widespread than other colors
DOMINANT_TO_NEXT = 2.5
# number of dilations to remove jpg artifacts
DILATE_ITER = 2


def _is_bgr(im):
  return len(im.shape) == 3 and im.shape[2] == 3


def get_dominant_colors (image, bin_size=np.array([4,4,4])):
  ''' check if there is a uniform wide background in an image '''

  assert _is_bgr (image)

  ranges = [0,256,0,256,0,256]
  # bookkeeping
  hist_size = np.divide(np.array([256,256,256]), bin_size)

  hist_flat = cv2.calcHist([image], [0, 1, 2], None, hist_size, ranges)
  hist = hist_flat.reshape((np.prod(hist_size),))
  ind = np.argsort(-hist)  # "-hist" to sort highest-first

  if hist[ind[1]] == 0:
    print 'BAD: nothing in the image'
    return None
  # check that there is much more dominant color than any other color
  elif hist[ind[0]] / hist[ind[1]] < DOMINANT_TO_NEXT:
    print 'BAD: hist[ind[0]] / hist[ind[1]] = %f' % (hist[ind[0]] / hist[ind[1]])
    return None
  # check that the dominant color takes a large part of the image
  elif hist[ind[0]] / np.sum(hist) < DOMINANT_TO_WHOLE:
    print 'BAD: hist[ind[0]] / np.sum(hist) = %f' % (hist[ind[0]] / np.sum(hist))
    return None
  else:
    color = np.array(np.unravel_index(ind[0], hist_size))
    color = np.multiply(color + 0.5, bin_size)
    print 'OK. Dominant color: ', color
    return color


def remove_background (image):

  # remove frame if any
  image = image[5:-5, 5:-5, :]

  thresh = np.array([10,10,10]).astype(int)
  c = get_dominant_colors(image, bin_size=thresh/4)
  if c is None: return None

  b, g, r = np.rollaxis(image, axis=-1)
  c = c.astype(int)
  mask = np.ones(r.shape, dtype=bool)
  mask = np.bitwise_and(mask, b.astype(int) >= c[0]-thresh[0])
  mask = np.bitwise_and(mask, b.astype(int) <= c[0]+thresh[0])
  mask = np.bitwise_and(mask, g.astype(int) >= c[1]-thresh[1])
  mask = np.bitwise_and(mask, g.astype(int) <= c[1]+thresh[1])
  mask = np.bitwise_and(mask, r.astype(int) >= c[2]-thresh[2])
  mask = np.bitwise_and(mask, r.astype(int) <= c[2]+thresh[2])

  mask = np.invert(binary_fill_holes(np.invert(mask)))
  for i in range(DILATE_ITER):
    mask = binary_dilation(mask)

  r[mask] = 0
  g[mask] = 0
  b[mask] = 0
  image = np.dstack((b, g, r, np.invert(mask.astype(np.uint8)*255)))
  return image



if __name__ == "__main__":

  ''' test with an example image '''

  image = cv2.imread('logo.png')
  image = remove_background(image)
  if image is None:
    print 'background is not clear enough'
  else:
    cv2.imshow('image', image[:,:,:3])
    cv2.imshow('mask', image[:,:,-1])
    cv2.waitKey(-1)
