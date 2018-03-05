import sys
import os
import cv2
import distance
import face_recognition
from tqdm import tqdm
from .imutils import IMutils

class Photo_Collector:

    def __init__(self, target_faces, tolerance=0.5, min_face_size=256, min_luminosity=10, max_luminosity=245):
        self.target_faces = target_faces
        self.tolerance = float(tolerance)
        self.min_face_size = int(round(min_face_size))
        self.min_luminosity = min_luminosity
        self.max_luminosity = max_luminosity

        
    def get_num_bits_different(self, hash1, hash2):
        return bin(hash1 ^ hash2).count('1')


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
              hs = IMutils.dhash(img)
            except:
              continue
            h, w = IMutils.cv_size(img)
            size = int(round(h*w))

            smallest = False
            duplicate_name = ""
            for key, value in list(hashes.items()):
                if self.get_num_bits_different(value[0], hs) <= tolerance:
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
                los = 1
                os.remove(file)
                #os.replace(file, os.path.join(outdir,duplicate_name))
            else:
                hashes[file] = [hs, size]
                
        return list(hashes.keys())
        
    
    def processPhotos(self, files, outdir, sample_height=500):
        total = len(files)
        counter = 0
        
        for file in tqdm(files):
            counter += 1
            #progress = (counter/total)*100;
            #sys.stdout.write("\r{0:.3g}% \t".format(progress))
            
            img = cv2.imread(file)
            rgb_img = img[:, :, ::-1]
            sample = IMutils.downsampleToHeight(rgb_img, sample_height)

            face_locations = face_recognition.face_locations(sample, number_of_times_to_upsample=0, model="cnn")
            face_encodings = face_recognition.face_encodings(sample, face_locations)

            for fenc, floc in zip(face_encodings, face_locations):
                result = face_recognition.compare_faces(self.target_faces, fenc, self.tolerance)

                #if the face found matches the target
                if any(result):
                    top, right, bottom, left = IMutils.scaleCoords(floc, IMutils.cv_size(sample), IMutils.cv_size(img))

                    size = int(round(min(bottom-top, right-left)))
                    face = img[int(round(top)):int(round(bottom)), int(round(left)):int(round(right))]
                    luminance = int(round(IMutils.getLuminosity(face)))

                    if (size >= self.min_face_size):
                        cropped = IMutils.cropAsPaddedSquare(img, top, bottom, left, right)

                        if bottom-top > 512 or right-left > 512:
                            try:
                              cropped = cv2.resize(cropped, (512, 512))
                            except:
                              continue

                        #disqualified_dir = os.path.join(outdir,"2Dark_or_2Bright")
                        #os.makedirs(disqualified_dir, exist_ok=True)
                        #outfile = os.path.join(outdir, "img_{0}.jpg".format(counter)) if luminance in range(self.min_luminosity,self.max_luminosity) else os.path.join(disqualified_dir, "img_{0}.jpg".format(counter))
                        if luminance in range(self.min_luminosity,self.max_luminosity):
                            outfile = os.path.join(outdir, "img_{0}.jpg".format(counter))
                        else:
                            continue
                        
                        IMutils.saveImage(cropped, outfile)
