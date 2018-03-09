import numpy as np
import sys
import os
import cv2
import face_recognition
from scipy.spatial import ConvexHull
from tqdm import tqdm
from .collector import Collector
from . import imutils
from . import utils

class Photo_Collector(Collector):

    def __init__(self, target_faces, tolerance=0.5, min_face_size=256, crop_size=512, min_luminosity=10, max_luminosity=245, one_face=False, mask_faces=False):
        Collector.__init__(self, target_faces, tolerance, min_face_size, crop_size, min_luminosity, max_luminosity, one_face, mask_faces )


    def processPhotos(self, files, outdir, sample_height=500):
        total = len(files)
        counter = 0

        for file in tqdm(files):
            counter += 1
            #progress = (counter/total)*100;
            #sys.stdout.write("\r{0:.3g}% \t".format(progress))
            
            img = cv2.imread(file)
            rgb_img = img[:, :, ::-1]
            sample = imutils.downsampleToHeight(rgb_img, sample_height)

            outfile = os.path.join(outdir, "img_{0}.jpg".format(counter))
            self.processImage(img, sample, outfile)


    def cleanDuplicates(self, files, outdir, tolerance):
        hashes = dict()

        total = len(files)
        counter = 0

        for file in tqdm(files):
            counter += 1
            #progress = (counter/total)*100;
            #sys.stdout.write("\r{0:.3g}% \t".format(progress))
            
            img = cv2.imread(file)
            try:
                hs = imutils.dhash(img)
            except:
                #remove corrupted file
                os.remove(file)
                continue
            h, w = imutils.cv_size(img)
            size = int(round(h*w))

            smallest = False
            duplicate_name = ""
            for key, value in list(hashes.items()):
                if utils.get_num_bits_different(value[0], hs) <= tolerance:
                    if value[1] < size:
                        del hashes[key]
                        name = "duplicateof_{0}".format(os.path.basename(file))
                        os.remove(file)
                        #os.replace(key, os.path.join(outdir,name))
                        smallest = False
                    else:
                        smallest = True
                        duplicate_name = "duplicateof_{0}".format(os.path.basename(key))
                        
            if smallest:
                os.remove(file)
                #os.replace(file, os.path.join(outdir,duplicate_name))
            else:
                hashes[file] = [hs, size]
                
        return list(hashes.keys())