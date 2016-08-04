import os, os.path as op
import sys
import cv2
import numpy as np
import base64
import argparse


def read_line (tsv_path, image_col, label_col = None, num = 10000000000):

  i = 0
  assert op.exists(tsv_path), tsv_path
  with open(tsv_path) as f:
    for line in f:

      # find image and label strings
      lineparts = line.split('\t')
      imagestring = lineparts[image_col]
      label = lineparts[label_col] if label_col is not None else None

      # decode image string
      jpgbytestring = base64.b64decode(imagestring)
      nparr = np.fromstring(jpgbytestring, np.uint8)
      image = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)

      if i >= num: return
      i += 1

      yield image, label



if __name__ == "__main__":

  ''' Tsv viewer '''

  path = 'data/Pokemon151.tsv'
  image_col = 2
  label_col = 0
  target_size = 1000
  
  for image, label in read_line(path, image_col, label_col):
    f = float(target_size) / max(image.shape[0], image.shape[1])
    image = cv2.resize(image, None, None, f, f)
    cv2.imshow('test', image)
    key = cv2.waitKey(-1)
    if key == 27: sys.exit()  # if Esc is pressed
    
