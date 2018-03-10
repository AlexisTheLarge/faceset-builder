import numpy as np
import cv2
import os
import face_recognition
from scipy.spatial import ConvexHull
from tqdm import tqdm
from .collector import Collector
from . import imutils

class Frame_Collector(Collector):

    def __init__(self, target_faces, tolerance=0.5, min_face_size=256, crop_size=512, min_luminosity=10, max_luminosity=245, laplacian_threshold=0, one_face=False, mask_faces=False, save_invalid=False):
        Collector.__init__(self, target_faces, tolerance, min_face_size, crop_size, min_luminosity, max_luminosity, laplacian_threshold, one_face, mask_faces, save_invalid )

    def processVideoFile(self, file, outdir, scanrate=0.2, capturerate=5, sample_height=500, batch_size=32, buffer_size=-1, greedy=False):
        vCap = cv2.VideoCapture(file)

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
                    target_found = self.scanFrame(lowres_frame, frame, greedy)
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


    def scanFrame(self, rgb_frame, raw_frame, greedy):
        
        face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=0, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for fenc, floc in zip(face_encodings, face_locations):
            result = face_recognition.compare_faces(self.target_faces, fenc, self.tolerance)

            #if the face found matches the target
            if any(result):
                if not greedy:
                    top, right, bottom, left = imutils.scaleCoords(floc, imutils.cv_size(rgb_frame), imutils.cv_size(raw_frame))
                    if (int(round(min(bottom-top, right-left))) < self.min_face_size):
                        return False

                return True


    def processBatch(self, raw_frames, rgb_frames, frame_count, outdir):
        target_found = False
        batch_of_face_locations = face_recognition.batch_face_locations(rgb_frames, number_of_times_to_upsample=0)

        for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):

            frame_number = frame_count - len(rgb_frames) + frame_number_in_batch

            raw_frame = raw_frames[frame_number_in_batch]
            rgb_frame = rgb_frames[frame_number_in_batch]

            outfile = os.path.join(outdir, "frame_{0}.jpg".format(frame_number))
            self.processImage(raw_frame, rgb_frame, outfile)

        return target_found
