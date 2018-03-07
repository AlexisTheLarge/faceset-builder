import cv2
import math
import numpy as np

class IMutils:

    def isbw(img):
        rgb_img = img[:, :, ::-1]
        height, width = IMutils.cv_size(img)
        if len(img.shape) > 2:
            for x in range(0, width):
                for y in range(0, height):
                    if not (img[x, y, 0] == img[x, y, 1] == img[x, y, 2]):
                        return False 
            return True
        else:
            return True

    def minMaxMeanMedian(gray):
        #h, w = IMutils.cv_size(image);
        min_val = gray.min()
        max_val = gray.max()
        mean = gray.mean()
        median = np.median(gray)
        return (min_val, max_val, mean, median)

    def getLuminosity(image):
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(img_yuv)
        mean = y.mean()
        return mean
        

    def getCannySharpness(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        minimum, maximum, mean, median = IMutils.minMaxMeanMedian(gray)
        edges = cv2.Canny(gray, mean*0.66, mean*1.33)    
        
        #Count the number of pixel representing an edge
        nCountCanny = cv2.countNonZero(edges);
        #print(nCountCanny)

        # Compute a sharpness grade:
        # < 1.5 = blurred, in movement
        # between 1.5 and 6 = acceptable
        # > 6 =stable, sharp
        h, w = IMutils.cv_size(image)
        dSharpness = (nCountCanny * 1000.0 / (h * w));
        return dSharpness

    def getLaplacianVariance(image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.max(cv2.convertScaleAbs(cv2.Laplacian(gray,3)))

    def cropAsPaddedSquare(image, y1, y2, x1, x2):
        h, w = IMutils.cv_size(image)

        y1 = int(round(y1))
        y2 = int(round(y2))
        x1 = int(round(x1))
        x2 = int(round(x2))

        # make square
        crop_w = x2-x1
        crop_h = y2-y1
        if crop_h > crop_w:
            x2 = x1 + crop_w
        elif crop_w > crop_h:
            y2 = y1 + crop_w

        diagonal = math.sqrt(2*((y2 - y1)**2))
        padding = int(round(diagonal/4))

        y1 = y1-padding
        y2 = y2+padding
        x1 = x1-padding
        x2 = x2+padding

        # keep within image bounds
        if y1 < 0:
            y2 += (0-y1)
            y1 += (0-y1)
        elif y2 > h:
            y1 -= (y2-h)
            y2 -= (y2-h)

        if x1 < 0:
            x2 += (0-x1)
            x1 += (0-x1)
        elif x2 > h:
            x1 -= (x2-w)
            x2 -= (x2-w)
            
        crop_img = image[y1:y2, x1:x2]

        h, w = IMutils.cv_size(crop_img)
        if h > w:
            padding = h-w
            crop_img = cv2.copyMakeBorder(crop_img,padding,0,0,0,cv2.BORDER_CONSTANT, 0)
        elif w > h:
            padding = w-h
            crop_img = cv2.copyMakeBorder(crop_img,0,0,padding,0,cv2.BORDER_CONSTANT, 0)


        return crop_img
        
    def cv_size(img):
        return tuple(img.shape[1::-1])

    def downsampleToHeight(img, height):
        newHeight = height
        oldHeight = img.shape[0]
        oldWidth = img.shape[1]
        r = newHeight / oldHeight
        newWidth = int(oldWidth * r)
        dim = (int(newWidth), int(newHeight))
        downsampled = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return downsampled

    def scaleCoords(coords, oldres, newres):
        top, right, bottom, left = coords

        old_h, old_w = oldres
        new_h, new_w = newres

        h_mult = float(new_h/old_h)
        w_mult = float(new_w/old_w)

        new_top = top*h_mult
        new_bottom = bottom*h_mult
        new_left = left*w_mult
        new_right = right*w_mult

        return (new_top,new_right,new_bottom,new_left)

    def saveImage(image, outfile):
        cv2.imwrite(outfile,image)


    def dhash(image, hashSize=8):
        # resize the input image, adding a single column (width) so we
        # can compute the horizontal gradient
        resized = cv2.resize(image, (hashSize + 1, hashSize))
 
        # compute the (relative) horizontal gradient between adjacent
        # column pixels
        diff = resized[:, 1:] > resized[:, :-1]
 
        # convert the difference image to a hash
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


        
        

    
