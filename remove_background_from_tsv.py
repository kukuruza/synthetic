import sys
import numpy as np
import cv2
import argparse
from tsv import read_line, encode_image
from remove_background import remove_background


'''
Remove background from images written as a TSV file
Images will be read from TSV file, analyzed, and written back to TSV file.
Images for which background could not be found and filtered are skipped.
Recorded images are encoded as png with alpha-channel, rather than jpg
'''

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--path_in', required=True, 
                      help='path of input TSV file')
  parser.add_argument('--path_out', required=True, 
                      help='path of output TSV file to be written')
  parser.add_argument('--image_col', required=True, type=int,
                      help='column id in TSV with encoded image')
  parser.add_argument('--display', required=False, action='store_true', 
                      help='display result on screen. Press Esc to exit.')
  parser.add_argument('-N', required=False, type=int, default=1000000000, 
                      help='if given, use that number of images (debugging)')
  args = parser.parse_args()


  with open(args.path_out, 'w') as f_out:
    for i, (image, lineparts) in enumerate(read_line(args.path_in, args.image_col)):
      if i >= args.N: break
      sys.stdout.write ('%d: ' % i)

      if args.display:
        f = float(500) / max(image.shape[0], image.shape[1])
        image0 = cv2.resize(image, None, None, f, f)
        cv2.imshow('src', image0)

      image = remove_background(image)

      if image is None: 
        if args.display:
          # show dummy.
          cv2.imshow('image', np.ones(image0.shape[:2], dtype=np.uint8) * 128)
          cv2.imshow('mask',  np.ones(image0.shape[:2], dtype=np.uint8) * 128)

      else:
        if args.display:
          f = float(500) / max(image.shape[0], image.shape[1])
          image = cv2.resize(image, None, None, f, f)
          cv2.imshow('image', image[:,:,:3])
          cv2.imshow('mask', image[:,:,-1])

        # write
        imagestr = encode_image(image)
        lineparts[args.image_col] = imagestr
        f_out.write('%s\n' % '\t'.join(lineparts))

      key = cv2.waitKey(-1)
      if key == 27: sys.exit()  # if Esc is pressed

