import os, os.path as op
import sys
import shutil
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import argparse
from synthesize import distort_logo, overlay_logo
from superpixels import find_good_rois
import traceback

'''
Distort a logo in multiple ways on overlay it onto the center of a VOC db
'''


do_write = True
do_show = False


def _reinit_dataset (set_root, set_name):
  ''' remove if exists and recreate VOC folder structure '''
  #if op.exists(set_root): shutil.rmtree(set_root)
  if not op.exists(op.join(set_root, 'ImageSets', 'Main')):
    os.makedirs(op.join(set_root, 'ImageSets', 'Main'))
  if not op.exists(op.join(set_root, 'Annotations')):
    os.makedirs(op.join(set_root, 'Annotations'))
  if not op.exists(op.join(set_root, 'JPEGImages')):
    os.makedirs(op.join(set_root, 'JPEGImages'))
  if op.exists(op.join(set_root, 'ImageSets', 'Main', set_name)):
    os.remove(op.join(set_root, 'ImageSets', 'Main', set_name))


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--logo_class', required=True, 
                      help='class name for VOC imdb annotations, e.g. "ups"')
  parser.add_argument('-l', '--logo_path', required=True, action='append', 
                      help='path to the canonical logo(s). e.g. "logo.png". '
                           'Can use this argument multiple times.')
  parser.add_argument('--in_imdb_root', required=True, 
                      help='root of input VOC imdb')
  parser.add_argument('--out_imdb_root', required=True, 
                      help='root of output VOC imdb')
  parser.add_argument('--set_name', required=True, 
                      help='set name, e.g. "train", "test"')
  parser.add_argument('-N', required=False, type=int, 
                      help='if given, use that number of images from imdb (debugging)')
  args = parser.parse_args()

  logos = []
  for imagefile in args.logo_path:
      assert op.exists(imagefile), imagefile
      logo = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED)
      assert logo is not None, imagefile
      assert len(logo.shape) == 3 and logo.shape[2] == 4
      logos.append(logo)

  # load val. ImageSet
  with open(op.join(args.in_imdb_root, 'ImageSets', 'Main', args.set_name)) as f:
    imids = f.read().splitlines()

  # recreate output dataset
  if do_write:
    _reinit_dataset (args.out_imdb_root, args.set_name)

  if args.N is not None: imids = imids[:args.N]
  for imid in imids:

    try:

      # pick a random logo from the set
      logo = logos[np.random.randint(low=0, high=len(logos))]
      #print logo.shape

      in_jpg_path = op.join(args.in_imdb_root, 'JPEGImages', '%s.jpg' % imid)
      background = cv2.imread(in_jpg_path)
      if len(background.shape) == 1:   # grayscale to color 
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
      elif background.shape[2] == 4:   # strip alpha channel
        background = background[:,:,:3]

      in_xml_path = op.join(args.in_imdb_root, 'Annotations', '%s.xml' % imid)
      xml_tree = ET.parse(in_xml_path)

      # if logo is already in the image, skip it
      def skip_image(xml_tree, logo_class):
        objs = xml_tree.findall('object')
        for obj in objs:
          cls = obj.find('name').text.lower().strip()
          if cls == logo_class: return True
        return False
      if skip_image(xml_tree, args.logo_class):
        print 'skipping image which already has objects of the same class'
        continue

      rois = find_good_rois(background, logo)
      print 'found %d good rois' % len(rois)

      blended = background.copy()
      for roi in rois:
        blended = overlay_logo (blended, logo, roi)

      if do_write:
        ## save image
        out_jpg_path = op.join(args.out_imdb_root, 'JPEGImages', '%s.jpg' % imid)
        cv2.imwrite(out_jpg_path, blended)

        ## save annotation
        # rename imageset
        root = xml_tree.getroot()
        root.find('folder').text = 'flickrlogo1_%s' % args.logo_class
        # remove all existing objects
        for obj in root.findall('object'):
          root.remove(obj)
        for roi in rois:
          # add a new synthesized object
          def add_object (root, roi, name):
            obj = ET.SubElement(root, 'object')
            child = ET.SubElement(obj, 'name')
            child.text = name
            child = ET.SubElement(obj, 'bndbox')
            grandchild = ET.SubElement(child, 'xmin')
            grandchild.text = str(roi[1])
            grandchild = ET.SubElement(child, 'ymin')
            grandchild.text = str(roi[0])
            grandchild = ET.SubElement(child, 'xmax')
            grandchild.text = str(roi[3])
            grandchild = ET.SubElement(child, 'ymax')
            grandchild.text = str(roi[2])
          add_object(root, roi, args.logo_class)
        # write to file
        out_xml_path = op.join(args.out_imdb_root, 'Annotations', '%s.xml' % imid)
        xml_tree.write(out_xml_path)

        ## save image index to imageset (yes, open the file again for every image)
        out_set_path = op.join(args.out_imdb_root, 'ImageSets', 'Main', args.set_name)
        with open(out_set_path, 'a') as f:
          f.write('%s\n' % imid)

      if do_show:
        ## test if it's written properly
        # for obj in xml_tree.findall('object'):
        #   bndbox = obj.find('bndbox')
        #   x1 = int(bndbox.find('xmin').text)
        #   y1 = int(bndbox.find('ymin').text)
        #   x2 = int(bndbox.find('xmax').text)
        #   y2 = int(bndbox.find('ymax').text)
        #   cv2.rectangle(blended, (x1,y1), (x2,y2), (255,255,255)) 
        for roi in rois:
         cv2.rectangle(blended, (roi[1],roi[0]), (roi[3],roi[2]), (255,255,255))
        cv2.imshow('test', blended)
        key = cv2.waitKey(-1)
        if key == 27: sys.exit()  # stop on Esc key

      if do_write:
        with open(op.join(args.out_imdb_root, 'ImageSets', 'Main', args.set_name)) as f:
          print 'wrote %d images out of %d original' % (len(f.readlines()), len(imids))

    except:
      traceback.print_exc()
