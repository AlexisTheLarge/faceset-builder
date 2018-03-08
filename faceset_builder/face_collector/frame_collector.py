import numpy as np
import cv2
import os
import sys
import face_recognition
from scipy.spatial import ConvexHull
from tqdm import tqdm
from . import imutils
from .photo_collector import Photo_Collector

class Frame_Collector:

    def __init__(self, target_faces, tolerance=0.5, min_face_size=256, crop_size=512, min_luminosity=10, max_luminosity=245, one_face=False, mask_faces=False):
        self.target_faces = target_faces
        self.tolerance = float(tolerance)
        self.min_face_size = int(round(min_face_size))
        self.crop_size = crop_size
        self.min_luminosity = min_luminosity
        self.max_luminosity = max_luminosity
        self.one_face = one_face
        self.mask_faces = mask_faces

    def processVideoFile(self, file, outdir, scanrate=0.2, capturerate=5, sample_height=500, batch_size=32, buffer_size=-1):
        vCap = cv2.VideoCapture(file)

        video_h = vCap.get(4)
        video_w = vCap.get(3)
        video_fps = vCap.get(cv2.CAP_PROP_FPS)

        video_total_frames = int(vCap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        scan_mult = int(round((1/scanrate)*video_fps))
        capture_mult = int(round((1/capturerate)*video_fps))

        buffer_size = int(round((scan_mult if (buffer_size == -1) else buffer_size)))

        scan_buffer = []

        # Batch arrays
        lowres_frames = []
        raw_frames = []
        
        frame_count = 0
        frame_count_all = 0

        target_found = False
        pbar = tqdm(total=video_total_frames)
        while vCap.isOpened():
            # Get frame number
            frameId = int(round(vCap.get(1)))
            
            # Grab a single frame of video
            ret, frame = vCap.read()

            # Bail out when the video file ends
            if not ret:
                break
                
            frame_count_all+=1

            #progress = (frame_count_all/video_total_frames)*100;
            #sys.stdout.write("\r{0:.3g}% \t".format(progress))
            pbar.update(1)


            # Convert to RGB for face_recognition
            rgb_frame = frame[:, :, ::-1]
            
            # Downsample frame to increase face_recognition speed, can result in fewer detections but we get plenty of frames from video anyway.
            lowres_frame = imutils.downsampleToHeight(rgb_frame, sample_height)
            
            if not target_found:
                if len(scan_buffer) == buffer_size:
                    scan_buffer = []
                scan_buffer.append(frame)
                if frameId % scan_mult == 0:
                    target_found = self.scanFrame(lowres_frame)
            else:
                # Process buffer
                if len(scan_buffer) > 0:
                    buffer_batch_rgb = []
                    buffer_batch_raw = []
                    for idx, buffered_frame in enumerate(scan_buffer):
                        if idx % capture_mult == 0:
                            buf_frame_rgb = buffered_frame[:, :, ::-1]
                            buf_frame_lowres = imutils.downsampleToHeight(buf_frame_rgb, sample_height)
                            buffer_batch_rgb.append(buf_frame_lowres)
                            buffer_batch_raw.append(buffered_frame)

                        if len(buffer_batch_rgb) == batch_size:
                            target_found = self.processBatch(buffer_batch_raw, buffer_batch_rgb, frame_count_all, outdir)
                            buffer_batch_rgb=[]
                            buffer_batch_raw=[]

                    if len(buffer_batch_rgb) > 0:
                        target_found = self.processBatch(buffer_batch_raw, buffer_batch_rgb, frame_count_all, outdir)
                        buffer_batch_rgb=[]
                        buffer_batch_raw=[]

                    # Clear buffer until face no longer detected
                    scan_buffer=[]
                
                if frameId % capture_mult == 0:
                    lowres_frames.append(lowres_frame)
                    raw_frames.append(frame)

                    
                    # Every n frames (batch size), batch process the list of frames to find faces
                    if len(lowres_frames) == batch_size:
                        target_found = self.processBatch(raw_frames, lowres_frames, frame_count_all, outdir)

                        # Clear the frames arrays to start the next batch
                        lowres_frames = []
                        raw_frames = []
        pbar.close()
        vCap.release()


    def scanFrame(self, rgb_frame):
        
        face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=0, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for fenc, floc in zip(face_encodings, face_locations):
            result = face_recognition.compare_faces(self.target_faces, fenc, self.tolerance)

            #if the face found matches the target
            if any(result):
                return True


    def processBatch(self, raw_frames, rgb_frames, frame_count, outdir):
        target_found = False
        batch_of_face_locations = face_recognition.batch_face_locations(rgb_frames, number_of_times_to_upsample=0)

        for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):

            frame_number = frame_count - len(rgb_frames) + frame_number_in_batch

            raw_frame = raw_frames[frame_number_in_batch]
            rgb_frame = rgb_frames[frame_number_in_batch]

            s_height, s_width = imutils.cv_size(rgb_frame)
            o_height, o_width = imutils.cv_size(raw_frame)
            scale_factor = o_height/s_height

            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)

            face_overlay = np.zeros((o_width, o_height), np.uint8)
            face_overlay.fill(255)
            tgt_face_poly = None

            crop_points = None
            outfile = ""
            for fenc, floc, flan in zip(face_encodings, face_locations, face_landmarks):
                result = face_recognition.compare_faces(self.target_faces, fenc, self.tolerance)

                #if the face found matches the target
                if any(result):
                    target_found = True
                    tgt_face_poly = Photo_Collector.get_face_mask(flan, (1.2, 1.2, 1.2), scale_factor)

                    crop_points = imutils.scaleCoords(floc, imutils.cv_size(rgb_frame), imutils.cv_size(raw_frame))
                    top, right, bottom, left = crop_points

                    size = int(round(min(bottom-top, right-left)))
                    face = raw_frame[int(round(top)):int(round(bottom)), int(round(left)):int(round(right))]
                    luminance = int(round(imutils.getLuminosity(face)))

                    if (size >= self.min_face_size):

                        #disqualified_dir = os.path.join(outdir,"2Dark_or_2Bright")
                        #os.makedirs(disqualified_dir, exist_ok=True)
                        #outfile = os.path.join(outdir, "frame_{0}.jpg".format(frame_number)) if luminance in range(self.min_luminosity,self.max_luminosity) else os.path.join(disqualified_dir, "frame_{0}.jpg".format(frame_number))
                        if luminance in range(self.min_luminosity,self.max_luminosity):
                            outfile = os.path.join(outdir, "frame_{0}.jpg".format(frame_number))
                else:
                    face_polygon = Photo_Collector.get_face_mask(flan, (0.8, 0.95, 0.95), scale_factor)
                    cv2.fillConvexPoly(face_overlay, face_polygon.astype(int), 0)

            if outfile != "":
                cv2.fillConvexPoly(face_overlay, tgt_face_poly.astype(int), 255)

                if (self.mask_faces):
                    raw_frame = cv2.bitwise_and(raw_frame, raw_frame, mask=face_overlay)

                top, right, bottom, left = crop_points
                cropped = imutils.cropAsPaddedSquare(raw_frame, top, bottom, left, right)
                h, w = imutils.cv_size(cropped)
                if h > self.crop_size or w > self.crop_size:
                    try:
                        cropped = cv2.resize(cropped, (self.crop_size, self.crop_size))
                    except:
                      continue
                
                if not imutils.isbw(cropped):
                    if self.one_face and Photo_Collector.has_multiple_faces(cropped):
                        break
                    imutils.saveImage(cropped, outfile)

        return target_found
