import numpy as np
import os
import cv2
import face_recognition
from scipy.spatial import ConvexHull
from tqdm import tqdm
from .collector import Collector
from . import imutils
from . import utils

class Photo_Collector(Collector):

    def __init__(self, target_faces, tolerance=0.5, min_face_size=256, crop_size=512, min_luminosity=10, max_luminosity=245, laplacian_threshold=0, one_face=False, mask_faces=False, save_invalid=False):
        Collector.__init__(self, target_faces, tolerance, min_face_size, crop_size, min_luminosity, max_luminosity, laplacian_threshold, one_face, mask_faces, save_invalid )


    def processPhotos(self, files, outdir, sample_height=500):
        total = len(files)
        counter = 0

        for file in tqdm(files):
            counter += 1
            
            img = cv2.imread(file)
            rgb_img = img[:, :, ::-1]
            sample = imutils.downsampleToHeight(rgb_img, sample_height)

            outfile = os.path.join(outdir, "img_{0}.jpg".format(counter))
            self.processImage(img, sample, outfile)


    def cleanDuplicates(self, files, tolerance):
        hashes = dict()

        duplicate_dir = os.path.join(self._invalid_dir, "duplicates")
        corrupted_dir = os.path.join(self._invalid_dir, "corrupted")

        os.makedirs(duplicate_dir, exist_ok=True)
        os.makedirs(corrupted_dir, exist_ok=True)

        total = len(files)
        counter = 0

        for file in tqdm(files):
            counter += 1
            
            img = cv2.imread(file)
            try:
                hs = imutils.dhash(img)
            except:
                #remove corrupted file
                if self.save_invalid:
                    os.rename(file, os.path.join(corrupted_dir, os.path.basename(file)))
                else:
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
                        name = "duplicateof_{0}_{1}".format(os.path.basename(file), os.path.basename(key))
                        if self.save_invalid:
                            os.rename(key, os.path.join(duplicate_dir, name))
                        else:
                            os.remove(file)
                        smallest = False
                    else:
                        smallest = True
                        duplicate_name = "duplicateof_{0}_{1}".format(os.path.basename(key), os.path.basename(file))
                        
            if smallest:
                if self.save_invalid:
                    os.rename(file, os.path.join(duplicate_dir, duplicate_name))
                else:
                    os.remove(file)
            else:
                hashes[file] = [hs, size]
                
        return list(hashes.keys())