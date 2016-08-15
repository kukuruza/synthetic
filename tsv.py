import os, os.path as op
import sys
import cv2
import numpy as np
import base64
import argparse


def read_line (tsv_path, image_col, num = 10000000000):

  i = 0
  assert op.exists(tsv_path), tsv_path
  with open(tsv_path) as f:
    for line in f:

      # find image and label strings
      lineparts = line.split('\t')
      imagestring = lineparts[image_col]

      # decode image string
      jpgbytestring = base64.b64decode(imagestring)
      nparr = np.fromstring(jpgbytestring, np.uint8)
      image = cv2.imdecode(nparr, -1)

      if i >= num: return
      i += 1

      yield image, lineparts


def encode_image (image, ext='.png'):
  # end image string
  retval, nparr = cv2.imencode(ext, image)
  jpgbytestring = nparr.tostring()
  imagestring = base64.b64encode(jpgbytestring)
  return imagestring


if __name__ == "__main__":

  ''' Tsv viewer '''

  path_in = 'data/Pokemon151.tsv'
  path_out = '/dev/null'
  image_col = 2
  target_size = 1000
  
  with open(path_out, 'w') as f_out:
    for image, lineparts in read_line(path_in, image_col):

      # write
      imagestr = encode_image(image)
      lineparts[image_col] = imagestr
      f_out.write('%s\n' % '\t'.join(lineparts))

      # display
      print 'image shape:', image.shape
      f = float(target_size) / max(image.shape[0], image.shape[1])
      image = cv2.resize(image, None, None, f, f)
      cv2.imshow('test', image)
      key = cv2.waitKey(-1)
      if key == 27: break  # if Esc is pressed
    

